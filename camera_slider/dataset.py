import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import logging
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_slider.compose import layout_board
from camera_slider.registration import FrameObs, detect_frame, make_detector
from camera_slider.scan_index import BoardMatch, index_frames, match_boards, read_logs, segment_frames
from camera_slider.shots import PassShots, RailFit, assign_pass, camera_id, fit_rail
from camera_slider.stitch import load_image, stitch_faces
from camera_slider.wood import make_sift, wood_features

LOGGER = logging.getLogger(__name__)
JPEG_MAX_SIDE = 65500
FACE_ORDER = {4: "ADCB", 2: "AB"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the stitched-board dataset from CameraSlider scans.")
    parser.add_argument("--scan_root", type=Path, nargs="+", required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[], help="block=label pairs, e.g. SKEN_2810=spruce")
    parser.add_argument("--boards", nargs="*", default=None, help="only these <label>_<board> ids")
    parser.add_argument("--project_name", default="camera_slider")
    parser.add_argument("--dataset_name", default="boards")
    parser.add_argument("--pieces", type=int, default=4)
    parser.add_argument("--overlap_px", type=int, default=0)
    parser.add_argument("--gap_px", type=int, default=40)
    parser.add_argument("--margin_px", type=int, default=40, help="black border around the unrolled board")
    parser.add_argument("--board_gap_s", type=float, default=15.0)
    parser.add_argument("--pass_gap_s", type=float, default=3.0)
    parser.add_argument("--clock_offset_s", type=float, default=3600.0)
    parser.add_argument("--csv_delay_s", type=float, nargs=2, default=(20.0, 80.0))
    parser.add_argument("--min_markers", type=int, default=2)
    parser.add_argument("--outlier_px", type=float, default=8.0)
    parser.add_argument("--max_seam_px", type=float, default=3.0)
    parser.add_argument("--max_reject_seam_px", type=float, default=5.0)
    parser.add_argument("--no_height_correction", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--index_only", action="store_true")
    parser.add_argument(
        "--no_stretch_faces", action="store_true", help="centre the faces instead of stretching them to one length"
    )
    parser.add_argument(
        "--save_renders", action="store_true", help="also write each face's stitched frames before masking"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset(args)


def build_dataset(args: argparse.Namespace) -> None:
    labels = dict(pair.split("=", 1) for pair in args.labels)
    args.out_root.mkdir(parents=True, exist_ok=True)
    frames = index_frames(args.scan_root, args.exclude, args.workers)
    logs = read_logs(args.scan_root)
    segments = segment_frames(frames, args.board_gap_s, args.pass_gap_s)
    matches = match_boards(segments, logs, args.clock_offset_s, tuple(args.csv_delay_s))
    boards = [m for m in matches if m.segment is not None and m.log is not None and not m.log.is_calibration]
    write_index(args.out_root / "index.csv", matches, labels)
    write_contact_sheet(args.out_root / "board_ids.jpg", boards, labels)
    if args.index_only:
        return
    if args.boards is not None:
        boards = [m for m in boards if board_name(m, labels) in set(args.boards)]

    accepted = [m for m in boards if m.reason == "ok"]
    observations = detect_all(accepted, args.out_root / "cache", args.workers)
    rails = calibrate_rails(accepted, observations, labels)
    manifest_path = args.out_root / "manifest.csv"
    rows = _read_manifest(manifest_path)
    for match in boards:
        name = board_name(match, labels)
        if args.skip_existing and name in rows:
            continue
        if match.reason != "ok":
            rows[name] = {"board": name, "status": f"rejected: {match.reason}", "session": match.log.session}
            _write_manifest(manifest_path, rows)
            continue
        try:
            rows[name] = build_board(match, name, observations, rails.get(match.block), args)
        except Exception as error:
            LOGGER.exception("%s failed", name)
            rows[name] = {"board": name, "status": f"failed: {error}"}
        _write_manifest(manifest_path, rows)
    write_supervisely_meta(args.out_root / "supervisely" / args.project_name)
    LOGGER.info("Dataset written to %s", args.out_root)


def build_board(
    match: BoardMatch,
    name: str,
    observations: Dict[Path, FrameObs],
    rail: Optional[RailFit],
    args: argparse.Namespace,
) -> Dict:
    segment, log = match.segment, match.log
    LOGGER.info("%s: session %d, %d frames, %d passes", name, log.session, segment.n_frames, len(segment.passes))
    groups = [[segment.frames[i] for i in p] for p in segment.passes][: log.passes]
    pass_numbers = sorted(log.rows["pass"].unique())[: len(groups)]
    areas = [str(log.rows.loc[log.rows["pass"] == p, "area"].iloc[0]) for p in pass_numbers]
    paths = [p for g in groups for p in g]
    with ThreadPoolExecutor(args.workers) as pool:
        features = dict(zip(paths, pool.map(lambda p: wood_features(load_image(p), make_sift()), paths)))
    board_dir = args.out_root / "boards" / name
    board_dir.mkdir(parents=True, exist_ok=True)
    renders = [board_dir / f"render_{area}.jpg" for area in areas] if args.save_renders else None
    geometries, crops, faces, px_per_unit = stitch_faces(groups, observations, features, args, render_paths=renders)
    del features

    shots = []
    t0 = segment.start
    for group, pass_number, idx in zip(groups, pass_numbers, [p for p in segment.passes]):
        frame_ms = np.array([(segment.times[i] - t0).total_seconds() * 1000 for i in idx])
        cams = np.array([camera_id(observations[p]) for p in group])
        shots.append(assign_pass(frame_ms, log.rows[log.rows["pass"] == pass_number], cams, rail))

    order = FACE_ORDER.get(len(crops), "".join(areas))
    ordered = sorted(range(len(crops)), key=lambda k: order.index(areas[k]) if areas[k] in order else k)
    unrolled, stretch = layout_board(
        [crops[k] for k in ordered], args.gap_px, not args.no_stretch_faces, args.margin_px
    )

    for crop, area in zip(crops, areas):
        _save(board_dir / f"face_{area}.jpg", crop.image)
        cv2.imwrite(str(board_dir / f"face_{area}_mask.png"), crop.mask.astype(np.uint8) * 255)
    unrolled_path = _save(board_dir / "unrolled.jpg", unrolled)
    scale = min(1.0, 4000 / unrolled.shape[1])
    cv2.imwrite(
        str(board_dir / "preview.jpg"),
        cv2.resize(unrolled, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    rejected = reject_reasons(faces, len(crops), log.passes, args.max_reject_seam_px)
    pieces = [] if rejected else write_pieces(unrolled, name, args)

    marker_pitch_units = float(np.median([g.pitch for g in geometries]))
    marker_side_mm = rail.pitch_mm / marker_pitch_units if rail is not None else None
    px_per_mm = px_per_unit / marker_side_mm if marker_side_mm else None
    report = {
        "board": name,
        "status": f"rejected: {'; '.join(rejected)}" if rejected else "ok",
        "board_id": log.board,
        "block": match.block,
        "session": log.session,
        "log": str(log.path),
        "match_reason": match.reason,
        "mtime_check": match.mtime_check,
        "face_order": [areas[k] for k in ordered],
        "face_stretch": dict(zip([areas[k] for k in ordered], stretch)),
        "px_per_marker_side": px_per_unit,
        "marker_side_mm": marker_side_mm,
        "px_per_mm": px_per_mm,
        "rail_pitch_mm": rail.pitch_mm if rail is not None else None,
        "unrolled": unrolled_path.name,
        "unrolled_shape": list(unrolled.shape[:2]),
        "pieces": pieces,
        "faces": [
            {**face, "area": area, "direction": str(log.rows.loc[log.rows["pass"] == p, "dir"].iloc[0])}
            for face, area, p in zip(faces, areas, pass_numbers)
        ],
        "shots": [_shots_table(group, s) for group, s in zip(groups, shots)],
    }
    (board_dir / "report.json").write_text(json.dumps(report, indent=1, default=_json_default))
    return _manifest_row(report, faces, shots, segment.n_frames, log.shots)


def reject_reasons(faces: List[Dict], n_faces: int, n_passes: int, max_seam_px: float) -> List[str]:
    reasons = []
    if n_faces != n_passes:
        reasons.append(f"{n_faces} of {n_passes} faces")
    for face in faces:
        if face["uncovered_columns"] > 0:
            reasons.append(f"face {face['face']}: {face['uncovered_columns']} uncovered columns")
        if face["seam_max_px"] is None or face["seam_max_px"] > max_seam_px:
            reasons.append(f"face {face['face']}: seam {face['seam_max_px']} px")
    return reasons


def write_pieces(unrolled: np.ndarray, name: str, args: argparse.Namespace) -> List[Dict]:
    project = args.out_root / "supervisely" / args.project_name / args.dataset_name
    (project / "img").mkdir(parents=True, exist_ok=True)
    (project / "ann").mkdir(parents=True, exist_ok=True)
    width = unrolled.shape[1]
    bounds = np.linspace(0, width, args.pieces + 1).round().astype(int)
    pieces = []
    for x0, x1 in zip(bounds[:-1], bounds[1:]):
        a, b = max(0, x0 - args.overlap_px // 2), min(width, x1 + args.overlap_px - args.overlap_px // 2)
        piece = unrolled[:, a:b]
        piece_name = f"{name}_x{a:05d}.jpg"
        _save(project / "img" / piece_name, piece)
        ann = {
            "description": "",
            "tags": [],
            "size": {"height": piece.shape[0], "width": piece.shape[1]},
            "objects": [],
        }
        (project / "ann" / f"{piece_name}.json").write_text(json.dumps(ann))
        pieces.append({"name": piece_name, "x0": int(a), "x1": int(b)})
    return pieces


def write_supervisely_meta(project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    meta = project / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"classes": [], "tags": [], "projectType": "images"}, indent=2))


def detect_all(boards: List[BoardMatch], cache_dir: Path, workers: int) -> Dict[Path, FrameObs]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    observations: Dict[Path, FrameObs] = {}
    for match in boards:
        cache = cache_dir / f"markers_s{match.log.session:03d}_{match.block}.pkl"
        if cache.exists():
            cached = pickle.loads(cache.read_bytes())
            if list(cached) == match.segment.frames:
                observations.update(cached)
                continue
        with ThreadPoolExecutor(workers) as pool:
            found = dict(
                zip(
                    match.segment.frames,
                    pool.map(lambda p: detect_frame(p.name, load_image(p), make_detector()), match.segment.frames),
                )
            )
        cache.write_bytes(pickle.dumps(found))
        observations.update(found)
        LOGGER.info("Markers detected for board %s (%d frames)", match.log.board, len(found))
    return observations


def calibrate_rails(
    boards: List[BoardMatch], observations: Dict[Path, FrameObs], labels: Dict[str, str]
) -> Dict[str, RailFit]:
    samples: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]] = {}
    for match in boards:
        log, segment = match.log, match.segment
        for p, idx in zip(sorted(log.rows["pass"].unique()), segment.passes):
            rows = log.rows[log.rows["pass"] == p]
            if len(idx) != len(rows):
                continue
            cams = np.array([camera_id(observations[segment.frames[i]]) for i in idx])
            samples.setdefault(match.block, []).append((cams, rows["pos_mm"].to_numpy(), str(rows["dir"].iloc[0])))
    rails = {}
    for block, block_samples in samples.items():
        LOGGER.info("Calibrating the rail for %s from %d complete passes", labels.get(block, block), len(block_samples))
        rail = fit_rail(block_samples)
        if rail is not None:
            rails[block] = rail
    return rails


def write_index(path: Path, matches: List[BoardMatch], labels: Dict[str, str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "board",
                "block",
                "session",
                "log",
                "first_frame",
                "last_frame",
                "start",
                "frames",
                "passes",
                "shots",
                "reason",
                "mtime_check",
            ]
        )
        for m in matches:
            seg, log = m.segment, m.log
            writer.writerow(
                [
                    board_name(m, labels) if log is not None else "",
                    m.block or "",
                    m.session if m.session is not None else "",
                    log.path.name if log is not None else "",
                    seg.frames[0].name if seg else "",
                    seg.frames[-1].name if seg else "",
                    seg.start.isoformat() if seg else "",
                    seg.n_frames if seg else 0,
                    "/".join(str(len(p)) for p in seg.passes) if seg else "",
                    log.shots if log is not None else "",
                    m.reason,
                    "" if m.mtime_check is None else m.mtime_check,
                ]
            )
    LOGGER.info("Board index written to %s", path)


def write_contact_sheet(path: Path, boards: List[BoardMatch], labels: Dict[str, str], thumb: int = 480) -> None:
    tiles = []
    for match in boards:
        first, last = match.segment.frames[0], match.segment.frames[match.segment.passes[0][-1]]
        row = []
        for frame in (first, last):
            image = load_image(frame)
            row.append(
                cv2.resize(image, (thumb, int(thumb * image.shape[0] / image.shape[1])), interpolation=cv2.INTER_AREA)
            )
        tile = np.hstack(row)
        cv2.putText(tile, board_name(match, labels), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    h, w = tiles[0].shape[:2]
    sheet = np.zeros((h * ((len(tiles) + cols - 1) // cols), w * cols, 3), np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * h : r * h + tile.shape[0], c * w : c * w + tile.shape[1]] = tile[:h, :w]
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    LOGGER.info("Board-ID contact sheet written to %s", path)


def board_name(match: BoardMatch, labels: Dict[str, str]) -> str:
    board = match.log.board
    return (
        f"{labels.get(match.block, match.block)}_{int(board):03d}"
        if board.isdigit()
        else f"{labels.get(match.block, match.block)}_{board}"
    )


def _shots_table(group: List[Path], shots: PassShots) -> List[Dict]:
    return [
        {
            "frame": path.name,
            "shot": int(shot),
            "delay_ms": round(float(delay), 1),
            "position_mm": round(float(pos), 1),
            "position_resid_mm": None if not np.isfinite(resid) else round(float(resid), 1),
        }
        for path, shot, delay, pos, resid in zip(
            group, shots.shot, shots.delay_ms, shots.position_mm, shots.position_resid_mm
        )
    ]


def _manifest_row(report: Dict, faces: List[Dict], shots: List[PassShots], n_frames: int, n_shots: int) -> Dict:
    seams = [f["seam_max_px"] for f in faces if f["seam_max_px"] is not None]
    return {
        "board": report["board"],
        "status": report["status"],
        "block": report["block"],
        "session": report["session"],
        "faces": "".join(report["face_order"]),
        "frames": n_frames,
        "shots": n_shots,
        "skipped_shots": sum(len(s.skipped_shots) for s in shots),
        "late_frames": sum(s.n_late for s in shots),
        "unplaced_frames": sum(f["n_frames"] - f["n_placed"] for f in faces),
        "seam_median_px": round(
            float(np.median([f["seam_median_px"] for f in faces if f["seam_median_px"] is not None])), 2
        ),
        "seam_max_px": round(max(seams), 2) if seams else "",
        "height_k": "/".join(f"{f['height_scale_k']:.4f}" for f in faces),
        "px_per_mm": round(report["px_per_mm"], 3) if report["px_per_mm"] else "",
        "unrolled_h": report["unrolled_shape"][0],
        "unrolled_w": report["unrolled_shape"][1],
        "pieces": len(report["pieces"]),
        "mtime_check": report["mtime_check"],
        "match_reason": report["match_reason"],
    }


def _read_manifest(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["board"]: row for row in csv.DictReader(f)}


def _write_manifest(path: Path, rows: Dict[str, Dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows.values() for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))


def _save(path: Path, image: np.ndarray) -> Path:
    if max(image.shape[:2]) > JPEG_MAX_SIDE:
        path = path.with_suffix(".tif")
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix == ".jpg" else []
    if not cv2.imwrite(str(path), image, params):
        msg = f"Cannot write {path}"
        raise ValueError(msg)
    return path


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


if __name__ == "__main__":
    main()
