import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from camera_slider.dataset import detect_all
from camera_slider.scan_index import index_frames, match_boards, read_logs, segment_frames
from camera_slider.stitch import load_image, stitch_faces
from camera_slider.wood import make_sift, wood_features

LOGGER = logging.getLogger(__name__)
LABEL_SPACING_MM = 100.0
LABEL_WIDTH_MM = 12.0
MERGE_MM = 30.0
NOMINAL_MM_PER_STEP = 47.6


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mm per marker step from the tape-measure calibration passes (red 10 cm labels of the tape)."
    )
    parser.add_argument("--scan_root", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    passes = [measure_pass(render, origin_x, px_per_step) for render, origin_x, px_per_step in stitch_calibration(args)]
    result = {
        "mm_per_step": float(np.mean([p["mm_per_step"] for p in passes])),
        "passes": passes,
        "method": "red 10 cm labels of the tape counted along the stitched calibration pass, linear fit mm ~ marker steps",
    }
    (args.out_dir / "tape_calibration.json").write_text(json.dumps(result, indent=1))
    LOGGER.info("mm per marker step %.4f from %d passes", result["mm_per_step"], len(passes))


def stitch_calibration(args: argparse.Namespace) -> List[Tuple[Path, float, float]]:
    frames = index_frames(args.scan_root, [], args.workers)
    matches = match_boards(segment_frames(frames, 15.0, 3.0), read_logs(args.scan_root), 3600.0, (20.0, 80.0))
    calibration = [m for m in matches if m.segment is not None and m.log is not None and m.log.is_calibration]
    observations = detect_all(calibration, args.out_dir / "cache", args.workers)
    build = argparse.Namespace(
        min_markers=2,
        outlier_px=8.0,
        no_height_correction=False,
        workers=args.workers,
        no_distortion=False,
        max_seam_px=3.0,
    )
    out = []
    for match in calibration:
        groups = [[match.segment.frames[i] for i in p] for p in match.segment.passes][: match.log.passes]
        paths = [p for g in groups for p in g]
        with ThreadPoolExecutor(args.workers) as pool:
            features = dict(zip(paths, pool.map(lambda p: wood_features(load_image(p), make_sift()), paths)))
        name = match.log.board.replace(" ", "_")
        renders = [args.out_dir / f"{name}_render_{k}.jpg" for k in range(len(groups))]
        _, crops, _, px_per_step = stitch_faces(groups, observations, features, build, render_paths=renders)
        out += [(render, crop.origin[0], px_per_step) for render, crop in zip(renders, crops)]
    return out


def measure_pass(render: Path, origin_x: float, px_per_step: float) -> Dict:
    image = cv2.imread(str(render))
    labels = label_positions(image, px_per_step / NOMINAL_MM_PER_STEP)
    px_per_mm = px_per_step / NOMINAL_MM_PER_STEP
    gaps = np.round(np.diff(labels) / (LABEL_SPACING_MM * px_per_mm)).astype(int)
    if (gaps != 1).any():
        msg = f"{render.name}: tape labels missing or doubled (gaps {sorted(set(gaps.tolist()))} x 10 cm)"
        raise ValueError(msg)
    mm = np.arange(len(labels)) * LABEL_SPACING_MM
    steps = origin_x + labels / px_per_step
    slope, intercept = np.polyfit(steps, mm, 1)
    resid = mm - (slope * steps + intercept)
    LOGGER.info(
        "%s: %d labels over %.0f mm, %.4f mm per step, residual sd %.2f mm, max %.2f mm",
        render.name,
        len(labels),
        mm[-1],
        slope,
        resid.std(),
        np.abs(resid).max(),
    )
    return {
        "render": render.name,
        "labels": len(labels),
        "span_mm": float(mm[-1]),
        "mm_per_step": float(slope),
        "resid_sd_mm": float(resid.std()),
        "resid_max_mm": float(np.abs(resid).max()),
    }


def label_positions(image: np.ndarray, px_per_mm: float) -> np.ndarray:
    y0, y1 = tape_band(image)
    hsv = cv2.cvtColor(image[y0:y1], cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(red)
    width = LABEL_WIDTH_MM * px_per_mm
    keep = [k for k in range(1, n) if 0.5 * width < stats[k, cv2.CC_STAT_WIDTH] < 2.0 * width]
    xs = np.sort(centroids[keep, 0])
    if len(xs) < 3:
        msg = "Fewer than 3 red tape labels found"
        raise ValueError(msg)
    groups = np.split(xs, np.flatnonzero(np.diff(xs) > MERGE_MM * px_per_mm) + 1)
    return np.array([g.mean() for g in groups])


def tape_band(image: np.ndarray, min_rows: int = 30) -> Tuple[int, int]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = (hsv[..., 0] >= 20) & (hsv[..., 0] <= 45) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    frac = yellow.sum(1) / np.maximum((image.max(2) > 0).sum(1), 1)
    edges = np.flatnonzero(np.diff(np.r_[0, (frac > 0.3).astype(np.int8), 0]))
    bands = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_rows]
    if len(bands) != 1:
        msg = f"Expected one tape band, found {len(bands)}"
        raise ValueError(msg)
    return bands[0]


if __name__ == "__main__":
    main()
