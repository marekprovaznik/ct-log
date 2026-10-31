import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import datetime
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from camera_slider.compose import FaceCrop, crop_board, layout_board, render_face
from camera_slider.registration import (
    FrameObs,
    PassRegistration,
    detect_frame,
    find_gaps,
    make_detector,
    register_pass,
)
from camera_slider.wood import PairMatches, make_sift, match_features, wood_features

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
EXIF_IFD = 0x8769
EXIF_DATETIME = 306
EXIF_DATETIME_ORIGINAL = 36867
EXIF_SUBSEC_ORIGINAL = 37521
MATCH_SPAN = 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch the CameraSlider passes of one board into an unrolled surface."
    )
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--board_id", default=None)
    parser.add_argument("--n_faces", type=int, default=4)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--pass_gap_s", type=float, default=3.0)
    parser.add_argument("--min_markers", type=int, default=2)
    parser.add_argument("--outlier_px", type=float, default=8.0)
    parser.add_argument("--marker_mm", type=float, default=None)
    parser.add_argument("--px_per_mm", type=float, default=None)
    parser.add_argument("--no_height_correction", action="store_true")
    parser.add_argument("--max_seam_px", type=float, default=3.0)
    parser.add_argument("--ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--preview_width", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stitch_board(args)


def stitch_board(args: argparse.Namespace) -> Dict:
    if args.px_per_mm is not None and args.marker_mm is None:
        msg = "--px_per_mm needs --marker_mm"
        raise ValueError(msg)
    board_id = args.board_id or args.frames_dir.name
    paths = sorted(p for p in args.frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        msg = f"No images in {args.frames_dir}"
        raise ValueError(msg)
    times = dict(zip(paths, (capture_time(p) for p in paths)))
    passes = group_passes(paths, [times[p] for p in paths], args.pass_gap_s)
    LOGGER.info("%s: %d frames in %d passes (%s)", board_id, len(paths), len(passes), [len(p) for p in passes])
    if len(passes) != args.n_faces and not args.allow_partial:
        msg = f"Expected {args.n_faces} passes, found {len(passes)}; check --pass_gap_s or use --allow_partial"
        raise ValueError(msg)

    with ThreadPoolExecutor(args.workers) as pool:
        analysed = dict(zip(paths, pool.map(analyse_frame, paths)))
    out_dir = args.out_dir / board_id
    out_dir.mkdir(parents=True, exist_ok=True)
    geometries, crops, faces, px_per_unit = stitch_faces(
        passes,
        {p: a[0] for p, a in analysed.items()},
        {p: a[1] for p, a in analysed.items()},
        args,
        px_per_unit=args.px_per_mm * args.marker_mm if args.px_per_mm is not None else None,
    )
    for face, crop in enumerate(crops, start=1):
        cv2.imwrite(str(out_dir / f"face_{face}.{args.ext}"), crop.image, _write_params(args.ext))
        cv2.imwrite(str(out_dir / f"face_{face}_mask.png"), crop.mask.astype(np.uint8) * 255)

    board, _ = layout_board(crops)
    cv2.imwrite(str(out_dir / f"unrolled.{args.ext}"), board, _write_params(args.ext))
    scale = min(1.0, args.preview_width / board.shape[1])
    preview = cv2.resize(board, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "preview.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])

    report = {
        "board_id": board_id,
        "frames_dir": str(args.frames_dir.resolve()),
        "n_frames": len(paths),
        "n_passes": len(passes),
        "pass_grouping": f"capture-time gap > {args.pass_gap_s} s",
        "px_per_marker": px_per_unit,
        "marker_mm": args.marker_mm,
        "px_per_mm": px_per_unit / args.marker_mm if args.marker_mm else None,
        "unrolled_shape": list(board.shape[:2]),
        "faces": faces,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    for f in faces:
        LOGGER.info(
            "face %d: %d/%d frames placed (%d by features), height k %.4f, seam median %.2f px, "
            "~%d missing, %d flagged, %d uncovered columns",
            f["face"],
            f["n_placed"],
            f["n_frames"],
            f["n_placed_by_features"],
            f["height_scale_k"],
            f["seam_median_px"],
            f["n_missing_estimate"],
            len(f["flagged_frames"]),
            f["uncovered_columns"],
        )
    LOGGER.info("Wrote %s", out_dir)
    return report


def stitch_faces(
    groups: List[List[Path]],
    observations: Dict[Path, FrameObs],
    features: Dict[Path, Tuple[np.ndarray, Optional[np.ndarray]]],
    args: argparse.Namespace,
    px_per_unit: Optional[float] = None,
    render_paths: Optional[List[Path]] = None,
) -> Tuple[List[PassRegistration], List[FaceCrop], List[Dict], float]:
    geometries = [
        solve_face(
            [observations[p] for p in group],
            [features[p] for p in group],
            args.min_markers,
            args.outlier_px,
            not args.no_height_correction,
            args.workers,
        )
        for group in groups
    ]
    if px_per_unit is None:
        px_per_unit = float(np.median([g.marker_side_px for g in geometries]))
    crops, faces = [], []
    for face, (group, geometry) in enumerate(zip(groups, geometries), start=1):
        shapes = [observations[p].shape for p in group]
        frames = [(f.name, f.transform, shapes[i]) for i, f in enumerate(geometry.frames) if f.registered]
        by_name = {p.name: p for p in group}
        render = render_face(frames, lambda n: load_image(by_name[n]), px_per_unit)
        if render_paths is not None:
            cv2.imwrite(str(render_paths[face - 1]), render.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop = crop_board(render)
        del render
        crops.append(crop)
        faces.append(face_report(face, group, shapes, geometry, crop, args.max_seam_px))
    return geometries, crops, faces, px_per_unit


def analyse_frame(path: Path) -> Tuple[FrameObs, Tuple[np.ndarray, Optional[np.ndarray]]]:
    image = load_image(path)
    return detect_frame(path.name, image, make_detector()), wood_features(image, make_sift())


def solve_face(
    observations: List[FrameObs],
    features: List[Tuple[np.ndarray, Optional[np.ndarray]]],
    min_markers: int,
    outlier_px: float,
    height_correction: bool,
    workers: int,
) -> PassRegistration:
    candidates = [(a, b) for a in range(len(features)) for b in range(a + 1, min(a + 1 + MATCH_SPAN, len(features)))]
    with ThreadPoolExecutor(workers) as pool:
        matched = pool.map(lambda ab: match_features(features[ab[0]], features[ab[1]]), candidates)
        pairs = [PairMatches(a, b, *m) for (a, b), m in zip(candidates, matched) if m is not None]
    return register_pass(observations, pairs, min_markers, outlier_px=outlier_px, estimate_height=height_correction)


def face_report(
    face: int,
    group: List[Path],
    shapes: List[Tuple[int, int]],
    registration: PassRegistration,
    crop: FaceCrop,
    max_seam_px: float,
) -> Dict:
    names = [p.name for p in group]
    transforms = {i: f.transform for i, f in enumerate(registration.frames) if f.registered}
    gaps = find_gaps(names, transforms, shapes)
    frames = []
    for i, fit in enumerate(registration.frames):
        own = [e for (a, b), e in registration.seams.items() if i in (a, b)]
        seam = max(own) if own else None
        frames.append(
            {
                "name": fit.name,
                "placed_by": fit.placed_by,
                "n_markers": fit.n_markers,
                "outlier_markers": fit.outlier_markers,
                "marker_reproj_median_px": fit.reproj_median_px,
                "marker_reproj_max_px": fit.reproj_max_px,
                "seam_px": seam,
                "flagged": not fit.registered or seam is None or seam > max_seam_px,
            }
        )
    seen = sorted(int(m) for m in registration.marker_poses)
    seams = list(registration.seams.values())
    return {
        "face": face,
        "first_frame": names[0],
        "last_frame": names[-1],
        "n_frames": len(group),
        "n_placed": len(transforms),
        "n_placed_by_features": sum(f.placed_by == "features" for f in registration.frames),
        "height_scale_k": registration.height_k,
        "seam_median_px": float(np.median(seams)) if seams else None,
        "seam_max_px": max(seams) if seams else None,
        "n_tie_outliers": registration.n_tie_outliers,
        "marker_ids": [seen[0], seen[-1]],
        "marker_ids_not_used": sorted(set(range(seen[0], seen[-1] + 1)) - set(seen)),
        "pitch_marker_units": registration.pitch,
        "gaps": [dataclasses.asdict(g) for g in gaps],
        "n_missing_estimate": sum(g.n_missing for g in gaps),
        "file_number_gaps": _number_gaps(group),
        "flagged_frames": [f["name"] for f in frames if f["flagged"]],
        "crop_box": list(crop.box),
        "crop_shape": list(crop.image.shape[:2]),
        "uncovered_columns": crop.uncovered_columns,
        "frames": frames,
    }


def group_passes(paths: List[Path], times: List[Optional[datetime.datetime]], gap_s: float) -> List[List[Path]]:
    if any(t is None for t in times):
        msg = "Some frames have no EXIF capture time; cannot split passes"
        raise ValueError(msg)
    order = sorted(range(len(paths)), key=lambda i: (times[i], paths[i].name))
    passes = [[paths[order[0]]]]
    for prev, cur in zip(order, order[1:]):
        if (times[cur] - times[prev]).total_seconds() > gap_s:
            passes.append([])
        passes[-1].append(paths[cur])
    return passes


def capture_time(path: Path) -> Optional[datetime.datetime]:
    exif = Image.open(path).getexif()
    sub = exif.get_ifd(EXIF_IFD)
    stamp = sub.get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
    if stamp is None:
        return None
    subsec = str(sub.get(EXIF_SUBSEC_ORIGINAL, "0")).strip() or "0"
    return datetime.datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S") + datetime.timedelta(seconds=float(f"0.{subsec}"))


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        msg = f"Cannot read {path}"
        raise ValueError(msg)
    return image


def _number_gaps(group: List[Path]) -> List[List[str]]:
    numbered = [(int(m.group(1)), p.name) for p in group if (m := re.search(r"(\d+)\D*$", p.stem))]
    return [[a[1], b[1]] for a, b in zip(numbered, numbered[1:]) if abs(b[0] - a[0]) != 1]


def _write_params(ext: str) -> List[int]:
    return [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]


if __name__ == "__main__":
    main()
