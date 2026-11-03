import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import dataclasses
import json
import logging
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

from ann_pipeline.timber.data import decode_bitmap
import cv2
import numpy as np
from src.utils.mask import mask_to_base64
import torch

from camera_slider.compose import NO_OWNER
from camera_slider.mapping import (
    MAPPING_FILE,
    BoardMap,
    crop_to_frame,
    face_to_unrolled,
    frame_to_crop,
    load_mapping,
    owners,
    unrolled_to_face,
)

LOGGER = logging.getLogger(__name__)
LAYOUT_VERSION = 1
SUBSAMPLES = np.array([[0.0, 0.0], [-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]])
FIT_MARGIN_PX = 60
FIT_REFINE_PX = 8
FIT_SCALE = 4
WOOD_GRAY = 20
WOOD_ROW_FRACTION = 0.1
DROP_FIELDS = ("id", "smartToolInput", "objectId")


@dataclasses.dataclass
class RowShift:
    row: int
    band: Tuple[int, int]
    dx: int
    dy: int
    mad: float
    ok: bool


@dataclasses.dataclass
class FramePoints:
    paths: List[str]
    gid: np.ndarray
    raw: np.ndarray
    lost_src: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move Supervisely annotations between two stitches of a board through the source frames."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-layout", help="fit per-piece face-row shifts of a hand-edited project")
    fit.add_argument("--edited_dir", type=Path, required=True, help="dataset dir of the edited project (img/)")
    fit.add_argument("--builder_dir", type=Path, required=True, help="dataset dir with the builder pieces (img/)")
    fit.add_argument("--mapping_root", type=Path, required=True, help="boards dir with <board>/mapping.json")
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--boards", nargs="*", default=None)
    fit.add_argument("--max_mad", type=float, default=2.0, help="mean abs grey difference of an ok row fit")
    fit.add_argument("--workers", type=int, default=8)
    move = sub.add_parser("transfer", help="map annotations of stitch A to stitch B")
    move.add_argument("--src_dir", type=Path, required=True, help="dataset dir of stitch A (ann/)")
    move.add_argument("--src_mapping_root", type=Path, required=True)
    move.add_argument("--src_layout", type=Path, default=None, help="fit-layout output when A was edited by hand")
    move.add_argument("--dst_mapping_root", type=Path, required=True)
    move.add_argument("--dst_layout", type=Path, default=None, help="fit-layout output to reproduce on B")
    move.add_argument("--out_dir", type=Path, required=True, help="dataset dir to write B's ann/ into")
    move.add_argument("--frames_out", type=Path, default=None, help="also save frame coordinates per board")
    move.add_argument("--boards", nargs="*", default=None)
    move.add_argument("--max_jump_px", type=float, default=1.5, help="seam jump above which an object is flagged")
    move.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "fit-layout":
        fit_layout(args)
    else:
        transfer(args)


def fit_layout(args: argparse.Namespace) -> None:
    pieces = sorted(p.name for p in (args.edited_dir / "img").iterdir())
    jobs = []
    for name in pieces:
        board = board_of(name)
        if args.boards is not None and board not in args.boards:
            continue
        if not (args.mapping_root / board / MAPPING_FILE).exists():
            LOGGER.warning("%s: no mapping for board %s, skipped", name, board)
            continue
        jobs.append((name, args.edited_dir / "img" / name, args.builder_dir / "img" / name, args.mapping_root / board))
    with ProcessPoolExecutor(args.workers) as pool:
        fitted = list(pool.map(_fit_piece, jobs, [args.max_mad] * len(jobs)))
    layout = {"version": LAYOUT_VERSION, "max_mad": args.max_mad, "pieces": dict(fitted)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(layout, indent=1))
    moved = sum(any(r["dx"] or r["dy"] for r in p["rows"]) for p in layout["pieces"].values())
    bad = [n for n, p in layout["pieces"].items() if not all(r["ok"] for r in p["rows"]) or p["unexplained_px"]]
    LOGGER.info("%d pieces fitted, %d with moved rows, %d need a look: %s", len(fitted), moved, len(bad), bad)


def fit_piece(edited: np.ndarray, builder: np.ndarray, board: BoardMap, max_mad: float) -> Dict:
    if edited.shape != builder.shape:
        msg = f"Edited piece {edited.shape} and builder piece {builder.shape} differ in size"
        raise ValueError(msg)
    h, w = builder.shape
    rows = []
    covered = np.zeros(h, np.int32)
    explained = np.zeros_like(edited, bool)
    for row, face in enumerate(board.faces):
        p = face.placement
        wood = np.nonzero(((builder[p.y : p.y + p.height] > WOOD_GRAY).mean(1)) > WOOD_ROW_FRACTION)[0]
        if len(wood) == 0:
            continue
        y0, y1 = p.y + int(wood[0]), p.y + int(wood[-1]) + 1
        dx, dy, mad = _match_band(edited, builder[y0:y1, FIT_MARGIN_PX : w - FIT_MARGIN_PX], y0)
        rows.append(dataclasses.asdict(RowShift(row, (y0, y1), dx, dy, mad, mad <= max_mad)))
        covered[max(0, y0 + dy) : min(h, y1 + dy)] += 1
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        band = np.zeros_like(builder)
        band[p.y : p.y + p.height] = builder[p.y : p.y + p.height]
        explained |= cv2.warpAffine(band, shift, (w, h), flags=cv2.INTER_NEAREST) > WOOD_GRAY
    explained = cv2.dilate(explained.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    unexplained = (edited > WOOD_GRAY) & ~explained
    unexplained = cv2.morphologyEx(unexplained.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return {
        "rows": rows,
        "overlap_rows": int((covered > 1).sum()),
        "unexplained_px": int(unexplained.sum()),
    }


def transfer(args: argparse.Namespace) -> None:
    src_layout = read_layout(args.src_layout)
    dst_layout = read_layout(args.dst_layout)
    anns: Dict[str, Dict[str, Dict]] = {}
    for path in sorted((args.src_dir / "ann").glob("*.json")):
        piece = path.name[: -len(".json")]
        if args.boards is None or board_of(piece) in args.boards:
            anns.setdefault(board_of(piece), {})[piece] = json.loads(path.read_text())
    (args.out_dir / "ann").mkdir(parents=True, exist_ok=True)
    meta = args.src_dir.parent / "meta.json"
    if meta.exists():
        shutil.copyfile(meta, args.out_dir.parent / "meta.json")
    if args.frames_out is not None:
        args.frames_out.mkdir(parents=True, exist_ok=True)
    jobs = [(board, pieces, src_layout, dst_layout, args) for board, pieces in sorted(anns.items())]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_transfer_board, jobs))
    records = [r for board_records in results for r in board_records]
    write_report(args.out_dir / "transfer_report.csv", records)
    flagged = [r for r in records if r["flagged"]]
    LOGGER.info(
        "%d boards, %d objects moved, %d flagged (seam jump > %.1f px, lost pixels or unreliable layout)",
        len(results),
        len(records),
        len(flagged),
        args.max_jump_px,
    )


def transfer_board(
    board: str,
    pieces: Dict[str, Dict],
    src_layout: Dict[str, List[RowShift]],
    dst_layout: Dict[str, List[RowShift]],
    args: argparse.Namespace,
) -> List[Dict]:
    src = load_mapping(args.src_mapping_root / board)
    dst = load_mapping(args.dst_mapping_root / board)
    out = {p["name"]: [] for p in dst.pieces}
    tags = {name: pieces[name]["tags"] for name in out if name in pieces}
    records = []
    saved: Dict[str, List] = {"obj": [], "gid": [], "raw": []}
    objects_meta = []
    for piece_name, ann in sorted(pieces.items()):
        shifts = src_layout.get(piece_name, [])
        x0 = src.piece(piece_name)["x0"]
        for obj in ann["objects"]:
            mask, origin = object_mask(obj, (ann["size"]["height"], ann["size"]["width"]))
            ys, xs = np.nonzero(mask)
            pix = np.stack([xs + origin[0], ys + origin[1]], 1).astype(np.float64)
            samples = (pix[:, None, :] + SUBSAMPLES[None]).reshape(-1, 2)
            frames = to_frames(src, shifts, x0, samples)
            names, dst_pts, dst_unrolled = from_frames(dst, dst_layout, frames)
            record = {
                "board": board,
                "src_piece": piece_name,
                "class": obj["classTitle"],
                "geometry": obj["geometryType"],
                "src_px": int(len(pix)),
                "lost_src_px": int(frames.lost_src.reshape(-1, len(SUBSAMPLES))[:, 0].sum()),
                "layout_ok": all(s.ok for s in shifts),
            }
            record.update(
                seam_jumps(pix, frames.gid.reshape(-1, len(SUBSAMPLES))[:, 0], dst_unrolled[:: len(SUBSAMPLES)])
            )
            placed = place_objects(obj, names, dst_pts, dst, out)
            record.update(placed)
            record["src_components"] = n_components(mask)
            record["flagged"] = bool(
                record["lost_src_px"]
                or record["lost_dst_px"]
                or not record["layout_ok"]
                or record["seam_gap_px"] > args.max_jump_px
                or record["seam_overlap_px"] > args.max_jump_px
                or record["dst_components"] > record["src_components"]
            )
            records.append(record)
            if args.frames_out is not None:
                centre = frames.gid.reshape(-1, len(SUBSAMPLES))[:, 0]
                saved["obj"].append(np.full(len(centre), len(objects_meta), np.int32))
                saved["gid"].append(centre.astype(np.int16))
                saved["raw"].append(frames.raw.reshape(-1, len(SUBSAMPLES), 2)[:, 0].astype(np.float32))
                objects_meta.append({"src_piece": piece_name, **_clean(obj)})
    for p in dst.pieces:
        ann = {
            "description": "",
            "tags": tags.get(p["name"], []),
            "size": {"height": dst.unrolled_shape[0], "width": p["x1"] - p["x0"]},
            "objects": out[p["name"]],
        }
        (args.out_dir / "ann" / f"{p['name']}.json").write_text(json.dumps(ann))
    if args.frames_out is not None:
        np.savez_compressed(
            args.frames_out / f"{board}.npz",
            paths=np.array([f.path for face in src.faces for f in face.frames]),
            **{k: np.concatenate(v) if v else np.zeros(0) for k, v in saved.items()},
        )
        (args.frames_out / f"{board}.objects.json").write_text(json.dumps(objects_meta))
    LOGGER.info("%s: %d objects", board, len(records))
    return records


def to_frames(board: BoardMap, shifts: List[RowShift], x0: int, pts: np.ndarray) -> FramePoints:
    rows, builder = edited_to_builder(shifts, pts)
    unrolled = builder + np.array([x0, 0.0])
    rows, crop = unrolled_to_face(board, unrolled, rows)
    paths = [f.path for face in board.faces for f in face.frames]
    offsets = np.cumsum([0] + [len(face.frames) for face in board.faces])
    gid = np.full(len(pts), -1, np.int64)
    raw = np.full((len(pts), 2), np.nan)
    for k, face in enumerate(board.faces):
        sel = np.nonzero(rows == k)[0]
        own = owners(face, crop[sel])
        for index in np.unique(own[own != NO_OWNER]):
            hit = sel[own == index]
            raw[hit] = crop_to_frame(face, crop[hit], face.frames[index])
            gid[hit] = offsets[k] + index
    return FramePoints(paths, gid, raw, gid < 0)


def from_frames(
    board: BoardMap, layout: Dict[str, List[RowShift]], frames: FramePoints
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = board.frame_index()
    unrolled = np.full((len(frames.gid), 2), np.nan)
    rows = np.full(len(frames.gid), -1)
    for gid in np.unique(frames.gid[frames.gid >= 0]):
        path = frames.paths[gid]
        if path not in lookup:
            continue
        k, index = lookup[path]
        hit = frames.gid == gid
        face = board.faces[k]
        unrolled[hit] = face_to_unrolled(face, frame_to_crop(face, frames.raw[hit], face.frames[index]))
        rows[hit] = k
    starts = np.array([p["x0"] for p in board.pieces])
    ends = np.array([p["x1"] for p in board.pieces])
    valid = np.isfinite(unrolled[:, 0])
    rx = np.where(valid, np.round(np.nan_to_num(unrolled[:, 0])), -1)
    which = np.searchsorted(starts, rx, side="right") - 1
    valid &= (which >= 0) & (rx < ends[np.clip(which, 0, None)])
    names = np.array([""] * len(rows), dtype=object)
    pts = np.full_like(unrolled, np.nan)
    for w in np.unique(which[valid]):
        name = board.pieces[w]["name"]
        hit = valid & (which == w)
        names[hit] = name
        pts[hit] = builder_to_edited(layout.get(name, []), rows[hit], unrolled[hit] - np.array([starts[w], 0.0]))
    return names, pts, unrolled


def edited_to_builder(shifts: List[RowShift], pts: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if not shifts:
        return None, pts
    starts = np.array([s.band[0] + s.dy for s in shifts], np.float64)
    ends = np.array([s.band[1] + s.dy for s in shifts], np.float64)
    y = pts[:, 1:2]
    distance = np.maximum(starts[None] - 0.5 - y, 0) + np.maximum(y - (ends[None] - 0.5), 0)
    nearest = distance.argmin(1)
    move = np.array([[s.dx, s.dy] for s in shifts], np.float64)
    return np.array([s.row for s in shifts])[nearest], pts - move[nearest]


def builder_to_edited(shifts: List[RowShift], rows: np.ndarray, pts: np.ndarray) -> np.ndarray:
    move = np.zeros((len(pts), 2))
    for s in shifts:
        move[rows == s.row] = (s.dx, s.dy)
    return pts + move


def seam_jumps(pix: np.ndarray, gid: np.ndarray, dst: np.ndarray) -> Dict:
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(pix)}
    gaps, overlaps, jumps = [0.0], [0.0], [0.0]
    n_pairs = 0
    for step, axis in (((1, 0), 0), ((0, 1), 1)):
        a, b = [], []
        for (x, y), i in index.items():
            j = index.get((x + step[0], y + step[1]))
            if j is not None:
                a.append(i)
                b.append(j)
        if not a:
            continue
        a, b = np.array(a), np.array(b)
        ok = np.isfinite(dst[a, 0]) & np.isfinite(dst[b, 0])
        a, b = a[ok], b[ok]
        delta = dst[b] - dst[a]
        same = gid[a] == gid[b]
        expected = np.median(delta[same], 0) if same.any() else np.array(step, np.float64)
        cross = ~same
        n_pairs += int(cross.sum())
        if cross.any():
            off = delta[cross] - expected
            gaps.append(float(off[:, axis].max()))
            overlaps.append(float(-off[:, axis].min()))
            jumps.append(float(np.linalg.norm(off, axis=1).max()))
    return {
        "seam_pairs": n_pairs,
        "seam_jump_px": round(max(jumps), 3),
        "seam_gap_px": round(max(gaps), 3),
        "seam_overlap_px": round(max(overlaps), 3),
    }


def place_objects(obj: Dict, names: np.ndarray, pts: np.ndarray, board: BoardMap, out: Dict[str, List]) -> Dict:
    n = len(SUBSAMPLES)
    lost = names.reshape(-1, n)[:, 0] == ""
    height = board.unrolled_shape[0]
    pieces = []
    dst_px = 0
    components = 0
    for name in sorted(set(names[names != ""])):
        width = board.piece(name)["x1"] - board.piece(name)["x0"]
        p = np.round(pts[names == name]).astype(int)
        keep = (p[:, 0] >= 0) & (p[:, 0] < width) & (p[:, 1] >= 0) & (p[:, 1] < height)
        p = np.unique(p[keep], axis=0)
        if len(p) == 0:
            continue
        mask = np.zeros((height, width), bool)
        mask[p[:, 1], p[:, 0]] = True
        new = _clean(obj)
        if obj["geometryType"] == "rectangle":
            new["points"] = {"exterior": [p.min(0).tolist(), p.max(0).tolist()], "interior": []}
        else:
            new["bitmap"] = bitmap_with_origin(mask)
        out[name].append(new)
        pieces.append(name)
        dst_px += int(mask.sum())
        components += n_components(mask)
    return {
        "dst_pieces": " ".join(pieces),
        "dst_px": dst_px,
        "lost_dst_px": int(lost.sum()),
        "dst_components": components,
    }


def object_mask(obj: Dict, size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    if obj["geometryType"] == "bitmap":
        return decode_bitmap(obj["bitmap"]["data"]), tuple(obj["bitmap"]["origin"])
    if obj["geometryType"] == "rectangle":
        (x0, y0), (x1, y1) = obj["points"]["exterior"]
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        return np.ones((y1 - y0 + 1, x1 - x0 + 1), bool), (x0, y0)
    msg = f"Unsupported geometry {obj['geometryType']}"
    raise ValueError(msg)


def bitmap_with_origin(mask: np.ndarray) -> Dict:
    ys, xs = np.nonzero(mask)
    y0, x0 = int(ys.min()), int(xs.min())
    patch = mask[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1]
    return {"origin": [x0, y0], "data": mask_to_base64(torch.from_numpy(patch))}


def n_components(mask: np.ndarray) -> int:
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)


def read_layout(path: Optional[Path]) -> Dict[str, List[RowShift]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if data["version"] != LAYOUT_VERSION:
        msg = f"{path}: layout version {data['version']}, expected {LAYOUT_VERSION}"
        raise ValueError(msg)
    return {
        name: [RowShift(r["row"], tuple(r["band"]), r["dx"], r["dy"], r["mad"], r["ok"]) for r in piece["rows"]]
        for name, piece in data["pieces"].items()
    }


def write_report(path: Path, records: List[Dict]) -> None:
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    LOGGER.info("Report written to %s", path)


def board_of(piece: str) -> str:
    return piece.rsplit("_x", 1)[0]


def _match_band(edited: np.ndarray, template: np.ndarray, y0: int) -> Tuple[int, int, float]:
    small = cv2.resize(edited, None, fx=1 / FIT_SCALE, fy=1 / FIT_SCALE, interpolation=cv2.INTER_AREA)
    small_t = cv2.resize(template, None, fx=1 / FIT_SCALE, fy=1 / FIT_SCALE, interpolation=cv2.INTER_AREA)
    _, _, _, loc = cv2.minMaxLoc(
        -cv2.matchTemplate(small.astype(np.float32), small_t.astype(np.float32), cv2.TM_SQDIFF)
    )
    h, w = template.shape
    wx0 = max(0, loc[0] * FIT_SCALE - FIT_REFINE_PX)
    wy0 = max(0, loc[1] * FIT_SCALE - FIT_REFINE_PX)
    window = edited[wy0 : wy0 + h + 2 * FIT_REFINE_PX, wx0 : wx0 + w + 2 * FIT_REFINE_PX]
    _, _, _, best = cv2.minMaxLoc(
        -cv2.matchTemplate(window.astype(np.float32), template.astype(np.float32), cv2.TM_SQDIFF)
    )
    x, y = wx0 + best[0], wy0 + best[1]
    mad = float(np.abs(edited[y : y + h, x : x + w].astype(np.float32) - template).mean())
    return int(x - FIT_MARGIN_PX), int(y - y0), round(mad, 3)


def _fit_piece(job: Tuple[str, Path, Path, Path], max_mad: float) -> Tuple[str, Dict]:
    name, edited_path, builder_path, board_dir = job
    edited = cv2.imread(str(edited_path), cv2.IMREAD_GRAYSCALE)
    builder = cv2.imread(str(builder_path), cv2.IMREAD_GRAYSCALE)
    return name, fit_piece(edited, builder, load_mapping(board_dir), max_mad)


def _transfer_board(job: Tuple) -> List[Dict]:
    return transfer_board(*job)


def _clean(obj: Dict) -> Dict:
    return {k: v for k, v in obj.items() if k not in DROP_FIELDS and k not in ("bitmap", "points")}


if __name__ == "__main__":
    main()
