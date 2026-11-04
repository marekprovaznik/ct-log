import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ann_pipeline.timber.data import decode_bitmap
import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from camera_slider.frame_agreement import piece_frame_agreement

LOGGER = logging.getLogger(__name__)

SCALE = 4
WOOD_LEVEL = 25
CLOSE_PX = 9
BAND_ROW_FRACTION = 0.3
BAND_MIN_MM = 8.0
EDGE_INSET_PX = 2
EDGE_STRIPS_PX = (4, 10)
FINE_BG_PX = 61
COARSE_BG_PX = 301
END_MARGIN_MM = 40.0
ORDER_LAG_PX = 8
SHIFT_WINDOW_MM = 300.0
SHIFT_MAX_MM = 40.0
SHIFT_MIN_NCC = 0.5
STRAY_ELONGATION = 4
STRAY_COMPACT_MM2 = 200.0
SHIFT_MIN_WINDOWS = 3
SHIFT_MAX_MAD_MM = 5.0
STEP_WINDOW_MM = 50.0
KNOT_CLASSES = ("dead_knot", "sound_knot", "wing_knot", "knot_unclassified")
TYPES = ("face_order", "face_count", "stray_segment", "edge_step", "face_shift", "bad_seam")
ORDER_COLUMNS = (
    "order_margin",
    "order_cycle",
    "board_order_margin",
    "board_order_cycle",
    "build_order_cycle",
    "build_order_wins",
    "build_order_fraction",
)


@dataclasses.dataclass
class Thresholds:
    order_piece: float = 0.08
    order_board: float = 0.02
    order_build_fraction: float = 0.8
    stray_mm2: float = 40.0
    edge_step_mm: float = 5.0
    shift_mm: float = 10.0
    knot_dx_mm: float = 20.0
    stretch_ratio: float = 1.03
    seam_px: float = 3.5
    frame_p90_px: float = 12.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag incorrectly stitched CameraSlider pieces and boards.")
    parser.add_argument("--pieces_dir", type=Path, required=True, help="directory with <board>_x<start>.jpg pieces")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ann_dir", type=Path, default=None, help="Supervisely ann/ for the same pieces")
    parser.add_argument("--boards_dir", type=Path, default=None, help="builder boards/ with report.json, mapping.json")
    parser.add_argument("--raw_root", type=Path, default=None, help="raw frame root; enables the frame agreement check")
    parser.add_argument("--raw_prefix", default="/home/jovyan/work/ctlog-eval/camera_slider_data/")
    parser.add_argument("--frame_cache", type=Path, default=None, help="default: <out_dir>/frame_agreement")
    parser.add_argument("--faces", nargs="+", default=["spruce=4", "larch=2"], help="species=number of faces")
    parser.add_argument("--px_per_mm", nargs="+", default=["spruce=7.7", "larch=13.7"], help="fallback per species")
    parser.add_argument("--order", default="0123", help="cycle of rows whose bottom edge meets the next row's top edge")
    parser.add_argument("--labels", type=Path, default=None, help="CSV piece,types (';'-separated, empty = clean)")
    parser.add_argument(
        "--reference_dir", type=Path, default=None, help="stitch_qa output of the builder pieces, to tell edits apart"
    )
    parser.add_argument("--workers", type=int, default=8)
    for field in dataclasses.fields(Thresholds):
        parser.add_argument(f"--{field.name}", type=float, default=field.default)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = Thresholds(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Thresholds)})
    faces = {k: int(v) for k, v in (s.split("=") for s in args.faces)}
    fallback_ppmm = {k: float(v) for k, v in (s.split("=") for s in args.px_per_mm)}
    jobs = []
    for path in sorted(args.pieces_dir.glob("*_x*.jpg")):
        board = board_of(path.stem)
        species = board.split("_")[0]
        report_path = args.boards_dir / board / "report.json" if args.boards_dir else None
        ppmm = load_json(report_path)["px_per_mm"] if report_path and report_path.exists() else fallback_ppmm[species]
        ann_path = args.ann_dir / f"{path.name}.json" if args.ann_dir else None
        jobs.append((path, ann_path, faces[species], ppmm, args.order))
    LOGGER.info("Measuring %d pieces", len(jobs))
    with ProcessPoolExecutor(args.workers) as pool:
        pieces = pd.DataFrame(list(pool.map(measure_piece_job, jobs, chunksize=2)))
    if args.boards_dir:
        pieces = pieces.merge(report_signals(args.boards_dir, pieces.piece.tolist()), on="piece", how="left")
        if args.raw_root:
            LOGGER.info("Checking raw frame agreement")
            cache = args.frame_cache or args.out_dir / "frame_agreement"
            agreement = frame_signals(args.boards_dir, pieces.board.unique(), args, cache)
            pieces = pieces.merge(agreement, on="piece", how="left")
    pieces = classify(pieces, thresholds, rotate_to_zero([int(c) for c in args.order]))
    if args.reference_dir:
        pieces = add_stray_origin(pieces, pd.read_csv(args.reference_dir / "pieces.csv"), thresholds)
    boards = board_table(pieces)
    pieces.to_csv(args.out_dir / "pieces.csv", index=False)
    boards.to_csv(args.out_dir / "boards.csv", index=False)
    summary = {"thresholds": dataclasses.asdict(thresholds), "args": {k: str(v) for k, v in vars(args).items()}}
    summary["pieces"] = len(pieces)
    summary["counts"] = {t: int(pieces.types.str.split(";").apply(lambda ts, t=t: t in ts).sum()) for t in TYPES}
    summary["flagged_pieces"] = int((pieces.types != "").sum())
    summary["flagged_boards"] = int((boards.types != "").sum())
    if args.labels:
        summary["metrics"] = evaluate(pieces, pd.read_csv(args.labels, keep_default_na=False))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("Flagged %d of %d pieces, %d boards", summary["flagged_pieces"], len(pieces), summary["flagged_boards"])


def measure_piece_job(job: Tuple[Path, Optional[Path], int, float, str]) -> Dict:
    path, ann_path, n_faces, ppmm, order = job
    ann = load_json(ann_path) if ann_path and ann_path.exists() else None
    return measure_piece(path, ann, n_faces, ppmm, [int(c) for c in order])


def measure_piece(path: Path, ann: Optional[Dict], n_faces: int, ppmm: float, order: List[int]) -> Dict:
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    raw = (image.max(axis=2) > WOOD_LEVEL).astype(np.uint8)
    wood = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((CLOSE_PX, CLOSE_PX), np.uint8))
    mm = ppmm / SCALE
    bands = find_bands(wood > 0, int(BAND_MIN_MM * mm))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, connectivity=8)
    mains = [main_component(labels, band, n_labels) for band in bands]
    edges = [face_edges(labels == k) for k in mains]
    row = {
        "piece": path.stem,
        "board": board_of(path.stem),
        "species": path.stem.split("_")[0],
        "x0": int(path.stem.rsplit("_x", 1)[1]),
        "width": image.shape[1] * SCALE,
        "px_per_mm": ppmm,
        "n_faces": n_faces,
        "n_bands": len(bands),
        "bands": ";".join(f"{a * SCALE}-{b * SCALE}" for a, b in bands),
    }
    row.update(stray_segments(stats, centroids, set(mains), edges, mm))
    row.update(edge_steps(edges, mm))
    margin = int(END_MARGIN_MM * mm)
    if len(bands) == n_faces == len(order) and n_faces >= 3:
        profiles = [edge_profiles(src, edge, margin) for src in (gray, saturation) for edge in edges]
        row.update(face_order(profiles, n_faces))
        pairs = [(order[i], order[(i + 1) % n_faces]) for i in range(n_faces)]
        dark = profiles[:n_faces]
        row.update(face_shift([(dark[a][1][0], dark[b][0][0]) for a, b in pairs], mm, SHIFT_MAX_MAD_MM))
    elif len(bands) == n_faces == 2:
        whole = [face_profile(gray, labels == k, margin) for k in mains]
        row.update(face_shift([(whole[0], whole[1])], mm, SHIFT_MAX_MAD_MM))
    if ann is not None:
        row.update(annotation_signals(ann, bands, order, ppmm))
    return row


def find_bands(wood: np.ndarray, min_rows: int) -> List[Tuple[int, int]]:
    rows = np.where(wood.mean(axis=1) > BAND_ROW_FRACTION)[0]
    if len(rows) == 0:
        return []
    runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    return [(int(r[0]), int(r[-1])) for r in runs if len(r) >= min_rows]


def main_component(labels: np.ndarray, band: Tuple[int, int], n_labels: int) -> int:
    counts = np.bincount(labels[band[0] : band[1] + 1].ravel(), minlength=n_labels)
    counts[0] = 0
    return int(counts.argmax())


def stray_segments(
    stats: np.ndarray,
    centroids: np.ndarray,
    mains: set,
    edges: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    mm: float,
) -> Dict:
    strays = []
    for k in range(1, len(stats)):
        area_mm2 = stats[k, cv2.CC_STAT_AREA] / mm**2
        cx, cy = int(centroids[k][0]), centroids[k][1]
        inside = any(has[cx] and top[cx] <= cy <= bottom[cx] for has, top, bottom in edges)
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        sliver = w >= STRAY_ELONGATION * h or area_mm2 >= STRAY_COMPACT_MM2
        if k not in mains and not inside and sliver and area_mm2 >= 1.0:
            strays.append((area_mm2, [int(v) * SCALE for v in stats[k, :4]]))
    strays.sort(key=lambda s: -s[0])
    return {
        "n_strays": len(strays),
        "stray_max_mm2": round(strays[0][0], 1) if strays else 0.0,
        "stray_boxes": json.dumps([s[1] for s in strays[:5]]),
    }


def face_edges(component: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    has = component.any(axis=0)
    top = component.argmax(axis=0)
    bottom = component.shape[0] - 1 - component[::-1].argmax(axis=0)
    return has, top, bottom


def edge_steps(edges: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], mm: float) -> Dict:
    w = max(2, int(STEP_WINDOW_MM * mm))
    margin = int(END_MARGIN_MM * mm)
    best, where = 0.0, ""
    for r, (has, top, bottom) in enumerate(edges):
        xs = np.where(has)[0][margin:-margin]
        if len(xs) < 2 * w + 2:
            continue
        for side, edge in (("top", top), ("bottom", bottom)):
            steps = step_profile(edge[xs].astype(np.float32), w)
            k = int(np.abs(steps).argmax())
            value = abs(float(steps[k])) / mm
            if value > best:
                best, where = value, f"row {r} {side} x={int(xs[k + w]) * SCALE}"
    return {"edge_step_mm": round(best, 2), "edge_step_at": where}


def step_profile(edge: np.ndarray, w: int) -> np.ndarray:
    med = np.median(np.lib.stride_tricks.sliding_window_view(edge, w), axis=1)
    return med[w:] - med[: len(med) - w]


def edge_profiles(
    src: np.ndarray, edge: Tuple[np.ndarray, np.ndarray, np.ndarray], margin: int
) -> List[List[np.ndarray]]:
    has, top, bottom = edge
    out = []
    for first in (True, False):
        variants = []
        for strip in EDGE_STRIPS_PX:
            start = top + EDGE_INSET_PX if first else bottom - EDGE_INSET_PX - strip
            rows = np.clip(start[None] + np.arange(strip)[:, None], 0, src.shape[0] - 1)
            prof = src[rows, np.arange(src.shape[1])[None]].mean(axis=0)[None]
            fine = cv2.blur(prof, (FINE_BG_PX, 1))[0]
            variants.append(np.clip(fine - prof[0], 0, None))
            variants.append(cv2.blur(prof, (9, 1))[0] - cv2.blur(prof, (COARSE_BG_PX, 1))[0])
        for v in variants:
            v[~has] = 0
            v[:margin] = 0
            v[-margin:] = 0
        out.append(variants)
    return out


def face_profile(gray: np.ndarray, component: np.ndarray, margin: int) -> np.ndarray:
    bg = cv2.blur(gray, (FINE_BG_PX, FINE_BG_PX))
    dark = np.where(component, np.clip(bg - gray - 10.0, 0, None), 0).mean(axis=0)
    dark[:margin] = 0
    dark[-margin:] = 0
    return dark


def face_order(profiles: List[List[List[np.ndarray]]], n: int) -> Dict:
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            values = []
            for src in range(len(profiles) // n):
                bottom = profiles[src * n + i][1]
                top = profiles[src * n + j][0]
                values += [max_ncc(a, b, ORDER_LAG_PX) for a, b in zip(bottom, top)]
            sim[i, j] = np.mean(values)
    scores = {}
    for perm in itertools.permutations(range(1, n)):
        cycle = (0, *perm)
        scores["".join(map(str, cycle))] = float(np.mean([sim[cycle[k], cycle[(k + 1) % n]] for k in range(n)]))
    return {"order_scores": json.dumps({k: round(v, 4) for k, v in scores.items()})}


def rotate_to_zero(order: List[int]) -> str:
    k = order.index(0)
    return "".join(map(str, order[k:] + order[:k]))


def face_shift(pairs: List[Tuple[np.ndarray, np.ndarray]], mm: float, max_mad: float) -> Dict:
    window = int(SHIFT_WINDOW_MM * mm)
    max_lag = int(SHIFT_MAX_MM * mm)
    lags = []
    for p, (a, b) in enumerate(pairs):
        for x in range(max_lag, len(a) - window - max_lag, window // 2):
            seg = a[x : x + window]
            if seg.std() == 0:
                continue
            values = [ncc(seg, b[x + d : x + d + window]) for d in range(-max_lag, max_lag + 1)]
            k = int(np.argmax(values))
            if values[k] >= SHIFT_MIN_NCC and 0 < k < 2 * max_lag:
                lags.append((p, round((k - max_lag) / mm, 1), (x + window // 2) * SCALE))
    row: Dict = {"shift_n": len(lags), "shift_lags": json.dumps(lags)}
    best = None
    for p in range(len(pairs)):
        values = np.array([v for q, v, _ in lags if q == p])
        if len(values) < SHIFT_MIN_WINDOWS:
            continue
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        if mad <= max_mad and (best is None or abs(med) > abs(best[1])):
            best = (p, med, mad)
    if best is not None:
        row.update({"shift_pair": best[0], "shift_med_mm": round(best[1], 2), "shift_mad_mm": round(best[2], 2)})
    return row


def annotation_signals(ann: Dict, bands: List[Tuple[int, int]], order: List[int], ppmm: float) -> Dict:
    knots = []
    groups: Dict[int, List[Tuple[float, float]]] = {}
    for obj in ann["objects"]:
        if obj["geometryType"] != "bitmap" or obj["classTitle"] not in KNOT_CLASSES:
            continue
        x0, y0 = obj["bitmap"]["origin"]
        mask = decode_bitmap(obj["bitmap"]["data"])
        knots.append((x0, y0, x0 + mask.shape[1], y0 + mask.shape[0]))
        ids = [t["value"] for t in obj["tags"] if t["name"] == "knot_id"]
        if ids:
            ys, xs = np.nonzero(mask)
            groups.setdefault(int(ids[0]), []).append((x0 + xs.mean(), y0 + ys.mean()))
    dx = []
    for members in groups.values():
        members.sort(key=lambda p: p[1])
        dx += [(q[0] - p[0]) / ppmm for p, q in zip(members, members[1:])]
    row = {
        "tags": ";".join(sorted({t["name"] for t in ann["tags"]})),
        "knot_pairs": len(dx),
        "knot_dx_med_mm": round(float(np.median(dx)), 1) if len(dx) >= 2 else np.nan,
        "knot_dx_mad_mm": round(float(np.median(np.abs(np.array(dx) - np.median(dx)))), 1) if len(dx) >= 2 else np.nan,
    }
    if len(bands) == len(order) >= 3:
        row.update(knot_edge_matches(knots, bands, order))
    return row


def knot_edge_matches(knots: List[Tuple[int, int, int, int]], bands: List[Tuple[int, int]], order: List[int]) -> Dict:
    tol = 25
    touching: List[Tuple[int, str, Tuple[int, int]]] = []
    for x0, y0, x1, y1 in knots:
        cy = (y0 + y1) / 2 / SCALE
        for r, (top, bottom) in enumerate(bands):
            if top <= cy <= bottom:
                if y0 <= top * SCALE + tol:
                    touching.append((r, "t", (x0, x1)))
                if y1 >= bottom * SCALE - tol:
                    touching.append((r, "b", (x0, x1)))
    n = len(bands)
    expected = {(order[i], order[(i + 1) % n]) for i in range(n)}
    matches = {"expected": 0, "other": 0}
    for (ra, sa, xa), (rb, sb, xb) in itertools.combinations(touching, 2):
        if ra == rb or sa == sb or interval_iou(xa, xb) <= 0.3:
            continue
        pair = (ra, rb) if sa == "b" else (rb, ra)
        matches["expected" if pair in expected else "other"] += 1
    return {"knot_edge_expected": matches["expected"], "knot_edge_other": matches["other"]}


def report_signals(boards_dir: Path, pieces: List[str]) -> pd.DataFrame:
    rows = []
    for board in sorted({board_of(p) for p in pieces}):
        report_path, mapping_path = boards_dir / board / "report.json", boards_dir / board / "mapping.json"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        stretch = list(report["face_stretch"].values())
        board_row = {
            "stretch_ratio": round(max(stretch) / min(stretch), 4),
            "board_seam_max_px": round(max(f["seam_max_px"] for f in report["faces"]), 2),
            "uncovered_columns": int(sum(f["uncovered_columns"] for f in report["faces"])),
        }
        per_piece = {p["name"][:-4]: {"seam_max_px": np.nan, "flagged_frames": 0} for p in report["pieces"]}
        if mapping_path.exists():
            for name, seam, flagged in frame_pieces(report, load_json(mapping_path)):
                entry = per_piece[name]
                entry["seam_max_px"] = round(np.nanmax([entry["seam_max_px"], seam]), 2)
                entry["flagged_frames"] += int(flagged)
        for name, entry in per_piece.items():
            rows.append({"piece": name, **board_row, **entry})
    return pd.DataFrame(rows)


def frame_pieces(report: Dict, mapping: Dict) -> List[Tuple[str, float, bool]]:
    out = []
    for face_m, face_r in zip(mapping["faces"], report["faces"]):
        for frame_m, frame_r in zip(face_m["frames"], face_r["frames"]):
            if frame_r["seam_px"] is None:
                continue
            xu = frame_centre_x(mapping, face_m, frame_m)
            for piece in mapping["pieces"]:
                if piece["x0"] <= xu < piece["x1"]:
                    out.append((piece["name"][:-4], float(frame_r["seam_px"]), bool(frame_r["flagged"])))
    return out


def frame_centre_x(mapping: Dict, face: Dict, frame: Dict) -> float:
    fh, fw = face["frame_shape"]
    p = np.array(frame["homography"]) @ np.array([fw / 2, fh / 2, 1.0])
    crop_x = (p[0] / p[2] - face["render_origin"][0]) * mapping["px_per_unit"] - face["crop_box"][0]
    placement = face["placement"]
    return (crop_x + 0.5) * placement["scale_x"] - 0.5 + placement["x"]


def frame_signals(boards_dir: Path, boards: List[str], args: argparse.Namespace, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [
        (boards_dir / b, args.raw_root, args.raw_prefix, cache / f"{b}.json")
        for b in sorted(boards)
        if (boards_dir / b / "mapping.json").exists()
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(piece_frame_agreement, jobs))
    return pd.DataFrame([row for rows in results for row in rows])


def classify(pieces: pd.DataFrame, t: Thresholds, expected: str) -> pd.DataFrame:
    pieces = pieces.copy()
    optional = ("order_scores", "shift_med_mm", "shift_mad_mm", "knot_dx_med_mm", "knot_dx_mad_mm", "stretch_ratio")
    optional += ("seam_max_px", "frame_p90_px")
    for column in optional + ("uncovered_columns", "tags"):
        if column not in pieces:
            pieces[column] = np.nan
    pieces = pieces.join(order_levels(pieces, expected, t))
    ratios = pd.DataFrame(index=pieces.index)
    ratios["face_order"] = pd.concat(
        [
            pieces.order_margin / t.order_piece,
            pieces.board_order_margin / t.order_board,
            pieces.build_order_fraction / t.order_build_fraction,
        ],
        axis=1,
    ).max(axis=1)
    ratios["stray_segment"] = pieces.stray_max_mm2 / t.stray_mm2
    ratios["face_count"] = (pieces.n_bands != pieces.n_faces) * 1.0
    ratios["edge_step"] = pieces.edge_step_mm / t.edge_step_mm
    ratios["face_shift"] = pd.concat(
        [
            pieces.shift_med_mm.abs() / t.shift_mm,
            (pieces.knot_dx_med_mm.abs() / t.knot_dx_mm).where(
                pieces.knot_dx_mad_mm <= 0.5 * pieces.knot_dx_med_mm.abs()
            ),
            (pieces.stretch_ratio - 1) / (t.stretch_ratio - 1),
        ],
        axis=1,
    ).max(axis=1)
    ratios["bad_seam"] = pd.concat(
        [pieces.seam_max_px / t.seam_px, pieces.frame_p90_px / t.frame_p90_px, (pieces.uncovered_columns > 0) * 1.0],
        axis=1,
    ).max(axis=1)
    ratios = ratios.fillna(0.0)
    pieces["types"] = ratios.apply(lambda r: ";".join(k for k in TYPES if r[k] >= 1.0), axis=1)
    pieces["score"] = ratios.max(axis=1).round(2)
    for k in TYPES:
        pieces[f"r_{k}"] = ratios[k].round(2)
    pieces["evidence"] = pieces.apply(evidence, axis=1)
    return pieces.sort_values(["score", "piece"], ascending=[False, True])


def order_levels(pieces: pd.DataFrame, expected: str, t: Thresholds) -> pd.DataFrame:
    out = pd.DataFrame(index=pieces.index, columns=ORDER_COLUMNS, dtype=object)
    has = pieces.order_scores.notna()
    if not has.any():
        return out
    margins = pieces.order_scores[has].apply(lambda s: margins_vs(json.loads(s), expected))
    out.loc[has, "order_margin"] = margins.apply(lambda m: max(m.values()))
    out.loc[has, "order_cycle"] = margins.apply(lambda m: max(m, key=m.get))
    table = pd.DataFrame(margins.tolist(), index=margins.index)
    table["board"] = pieces.board[has]
    table["species"] = pieces.species[has]
    board_mean = table.groupby(["species", "board"]).mean()
    best_cycle = board_mean.idxmax(axis=1)
    for (species, board), cycle in best_cycle.items():
        rows = has & (pieces.board == board)
        out.loc[rows, "board_order_margin"] = board_mean.loc[(species, board), cycle]
        out.loc[rows, "board_order_cycle"] = cycle
    for species, group in board_mean.groupby(level=0):
        cycle = best_cycle[group.index].mode().iloc[0]
        wins = int((group[cycle] > 0).sum())
        p = binomtest(wins, len(group), 0.5, alternative="greater").pvalue
        fraction = wins / len(group) if p < 1e-3 else 0.0
        rows = has & (pieces.species == species)
        out.loc[rows, "build_order_cycle"] = cycle
        out.loc[rows, "build_order_wins"] = f"{wins}/{len(group)}"
        out.loc[rows, "build_order_fraction"] = fraction
    return out.astype({c: float for c in ("order_margin", "board_order_margin", "build_order_fraction")})


def margins_vs(scores: Dict[str, float], expected: str) -> Dict[str, float]:
    return {k: v - scores[expected] for k, v in scores.items() if k != expected}


def add_stray_origin(pieces: pd.DataFrame, reference: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    pieces = pieces.copy()
    ref = reference.set_index("piece").stray_max_mm2
    flagged = pieces.types.str.split(";").apply(lambda ts: "stray_segment" in ts)
    in_ref = pieces.piece.isin(ref.index)
    builder = in_ref & (pieces.piece.map(ref).fillna(0.0) >= t.stray_mm2)
    pieces["stray_origin"] = ""
    pieces.loc[flagged & builder, "stray_origin"] = "builder"
    pieces.loc[flagged & in_ref & ~builder, "stray_origin"] = "hand_edit"
    pieces.loc[flagged & ~in_ref, "stray_origin"] = "no_reference"
    pieces.loc[flagged, "evidence"] = pieces.loc[flagged].apply(
        lambda r: "%s (origin: %s)" % (r.evidence, r.stray_origin), axis=1
    )
    return pieces


def evidence(row: pd.Series) -> str:
    parts = []
    types = row.types.split(";") if row.types else []
    if "face_order" in types:
        parts.append(
            "edge continuity favours row cycle %s: piece margin %.3f, board %s %.3f, build %s on %s boards"
            % (
                row.order_cycle,
                row.order_margin,
                row.board_order_cycle,
                row.board_order_margin,
                row.build_order_cycle,
                row.build_order_wins,
            )
        )
    if "stray_segment" in types:
        parts.append(
            "%d stray wood segments, largest %.0f mm2, boxes %s" % (row.n_strays, row.stray_max_mm2, row.stray_boxes)
        )
    if "face_count" in types:
        parts.append("%d wood bands (rows %s) for %d faces" % (row.n_bands, row.bands, row.n_faces))
    if "edge_step" in types:
        parts.append("edge step %.1f mm at %s" % (row.edge_step_mm, row.edge_step_at))
    if "face_shift" in types:
        parts.append(
            "face lag median %.1f mm; knot pair dx median %.1f mm; face length ratio %.3f"
            % (row.shift_med_mm, row.knot_dx_med_mm, row.stretch_ratio)
        )
    if "bad_seam" in types:
        parts.append(
            "tie seam max %.2f px, frame disagreement p90 %.2f px, uncovered columns %s"
            % (row.seam_max_px, row.frame_p90_px, row.uncovered_columns)
        )
    return " | ".join(parts)


def board_table(pieces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for board, group in pieces.groupby("board"):
        types = sorted({t for ts in group.types for t in ts.split(";") if t}, key=TYPES.index)
        flagged = group[group.types != ""]
        rows.append(
            {
                "board": board,
                "species": group.species.iloc[0],
                "pieces": len(group),
                "flagged_pieces": len(flagged),
                "types": ";".join(types),
                "score": group.score.max(),
                "stretch_ratio": group.stretch_ratio.iloc[0],
                "board_order_margin": group.board_order_margin.iloc[0],
                "flagged": " ".join(f"{p}[{t}]" for p, t in zip(flagged.piece, flagged.types)),
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "board"], ascending=[False, True])


def evaluate(pieces: pd.DataFrame, labels: pd.DataFrame) -> Dict:
    merged = labels.merge(pieces[["piece", "types"]], on="piece", how="inner", suffixes=("_true", "_pred"))
    metrics: Dict = {"n_labelled": len(merged)}
    for k in TYPES + ("any",):
        true = merged.types_true.apply(lambda s, k=k: bool(s) if k == "any" else k in s.split(";"))
        pred = merged.types_pred.apply(lambda s, k=k: bool(s) if k == "any" else k in s.split(";"))
        tp, fp, fn = int((true & pred).sum()), int((~true & pred).sum()), int((true & ~pred).sum())
        metrics[k] = {
            "positives": int(true.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "missed": merged.piece[true & ~pred].tolist(),
            "false_alarms": merged.piece[~true & pred].tolist(),
        }
    return metrics


def max_ncc(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    best = -1.0
    for d in range(-max_lag, max_lag + 1):
        best = max(best, ncc(a[max(0, d) : len(a) + min(0, d)], b[max(0, -d) : len(b) + min(0, -d)]))
    return best


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def interval_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(max(a[1], b[1]) - min(a[0], b[0]), 1)


def board_of(piece: str) -> str:
    return piece.rsplit("_x", 1)[0]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
