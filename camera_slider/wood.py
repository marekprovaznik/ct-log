import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

LOGGER = logging.getLogger(__name__)
SIFT_CONTRAST = 0.01
SIFT_FEATURES = 4000


@dataclasses.dataclass
class EdgeParams:
    window: int
    smooth: int
    min_run_frac: float
    margin: int
    min_gap: int
    reference_window: int
    tolerance: float


@dataclasses.dataclass
class PairMatches:
    a: int
    b: int
    pts_a: np.ndarray
    pts_b: np.ndarray


def make_sift() -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=SIFT_CONTRAST)


def wood_mask(image: np.ndarray, s_min: int = 60, v_min: int = 50, open_px: int = 5) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > s_min) & (hsv[..., 2] > v_min)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))


def segment_board(
    image: np.ndarray,
    scale: int = 1,
    open_px: int = 17,
    close_px: Tuple[int, int] = (12, 60),
    edge_window: int = 401,
    smooth_px: int = 101,
    min_run_frac: float = 0.5,
    edge_margin: int = 120,
    min_gap_px: int = 8,
    gap_v: int = 35,
    reference_window: int = 8001,
    edge_tolerance: float = 20.0,
) -> np.ndarray:
    small = cv2.resize(image, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA) if scale > 1 else image
    wood = wood_mask(small, open_px=max(1, open_px // scale) | 1)
    kernel = np.ones((max(1, close_px[0] // scale), max(1, close_px[1] // scale)), np.uint8)
    mask = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        msg = "No timber found"
        raise ValueError(msg)
    board = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    del labels
    lit = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2] >= gap_v
    edges = _column_edges(
        _fill_holes(board).astype(bool),
        lit,
        EdgeParams(
            window=edge_window // scale | 1,
            smooth=smooth_px // scale | 1,
            min_run_frac=min_run_frac,
            margin=edge_margin // scale,
            min_gap=max(1, min_gap_px // scale),
            reference_window=reference_window // scale | 1,
            tolerance=edge_tolerance / scale,
        ),
    )
    if edges is None:
        msg = "No timber found"
        raise ValueError(msg)
    return _draw_band(image.shape[:2], *edges, scale)


def wood_features(image: np.ndarray, sift: cv2.SIFT, scale: float = 0.5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.erode(wood_mask(small), np.ones((9, 9), np.uint8))
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), mask)
    return np.float64([kp.pt for kp in keypoints]).reshape(-1, 2) / scale, descriptors


def match_features(
    feats_a: Tuple[np.ndarray, Optional[np.ndarray]],
    feats_b: Tuple[np.ndarray, Optional[np.ndarray]],
    ratio: float = 0.75,
    ransac_px: float = 6.0,
    min_inliers: int = 30,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    (pts_a, des_a), (pts_b, des_b) = feats_a, feats_b
    if des_a is None or des_b is None or len(pts_a) < min_inliers or len(pts_b) < min_inliers:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < min_inliers:
        return None
    pa = pts_a[[m.queryIdx for m in good]]
    pb = pts_b[[m.trainIdx for m in good]]
    _, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=ransac_px)
    if inliers is None or inliers.sum() < min_inliers:
        return None
    keep = inliers.ravel().astype(bool)
    return pa[keep], pb[keep]


def _draw_band(
    shape: Tuple[int, int], columns: np.ndarray, top: np.ndarray, bottom: np.ndarray, scale: int
) -> np.ndarray:
    xs = (np.arange(shape[1]) + 0.5) / scale - 0.5
    inside = (xs >= columns[0] - 0.5) & (xs <= columns[-1] + 0.5)
    rows = np.arange(shape[0])[:, None] + 0.5
    band = np.zeros(shape, bool)
    band[:, inside] = (rows >= np.interp(xs[inside], columns, top) * scale) & (
        rows < np.interp(xs[inside], columns, bottom) * scale
    )
    return band


def _column_edges(
    mask: np.ndarray, lit: np.ndarray, params: EdgeParams
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    runs = np.array([_longest_run(column) for column in mask.T])
    lengths = runs[:, 1] - runs[:, 0]
    if not lengths.any():
        return None
    valid = np.flatnonzero(lengths >= params.min_run_frac * np.median(lengths[lengths > 0]))
    full = runs[valid]
    trimmed = np.array(
        [_trim_run(tuple(full[i]), lit[:, x], params.margin, params.min_gap) for i, x in enumerate(valid)]
    )
    reference = median_filter(trimmed, size=(params.reference_window, 1), mode="reflect")
    chosen = np.where(np.abs(trimmed - reference) <= np.abs(full - reference), trimmed, full)
    top, bottom = (
        uniform_filter1d(
            median_filter(
                _drop_outliers(valid, chosen[:, k], reference[:, k], params.tolerance), size=params.window, mode="reflect"
            ),
            params.smooth,
            mode="reflect",
        )
        for k in (0, 1)
    )
    return valid, top, bottom


def _drop_outliers(columns: np.ndarray, edge: np.ndarray, reference: np.ndarray, tolerance: float) -> np.ndarray:
    keep = np.abs(edge - reference) <= tolerance
    if not keep.any():
        return edge
    return np.interp(columns, columns[keep], edge[keep])


def _trim_run(run: Tuple[int, int], wood: np.ndarray, margin: int, min_gap: int) -> Tuple[int, int]:
    start, end = run
    width = min(margin, (end - start) // 4)
    if width < min_gap:
        return run
    top_gaps = _dark_runs(wood[start : start + width], min_gap)
    if top_gaps:
        start += top_gaps[-1][1]
    bottom_gaps = _dark_runs(wood[end - width : end], min_gap)
    if bottom_gaps:
        end = end - width + bottom_gaps[0][0]
    return start, end


def _dark_runs(wood: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~wood).astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_length]


def _longest_run(column: np.ndarray) -> Tuple[int, int]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], column.astype(np.int8), [0]])))
    if len(edges) == 0:
        return 0, 0
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    return (padded | (1 - flood))[1:-1, 1:-1]
