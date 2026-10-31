import dataclasses
import logging
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from camera_slider.wood import PairMatches

LOGGER = logging.getLogger(__name__)
ARUCO_DICT = cv2.aruco.DICT_7X7_1000
TEMPLATE = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
FRAME_PARAMS = 8
MARKER_PARAMS = 3
BLAS_THREADS = 4
K_BOUNDS = (0.5, 2.0)


@dataclasses.dataclass
class FrameObs:
    name: str
    ids: np.ndarray
    corners: np.ndarray
    shape: Tuple[int, int]


@dataclasses.dataclass
class FrameFit:
    name: str
    n_markers: int
    transform: Optional[np.ndarray] = None
    placed_by: Optional[str] = None
    reproj_median_px: Optional[float] = None
    reproj_max_px: Optional[float] = None
    outlier_markers: List[int] = dataclasses.field(default_factory=list)

    @property
    def registered(self) -> bool:
        return self.transform is not None


@dataclasses.dataclass
class PassRegistration:
    frames: List[FrameFit]
    marker_poses: Dict[int, np.ndarray]
    marker_side_px: float
    pitch: float
    height_k: float
    seams: Dict[Tuple[int, int], float]
    n_tie_outliers: int


@dataclasses.dataclass
class FrameGap:
    after: str
    before: str
    step_ratio: float
    n_missing: int


@dataclasses.dataclass
class _Problem:
    frames: List[int]
    markers: List[int]
    obs_frame: np.ndarray
    obs_marker: np.ndarray
    obs_xy: np.ndarray
    tie_a: np.ndarray
    tie_b: np.ndarray
    tie_ua: np.ndarray
    tie_ub: np.ndarray
    principal: np.ndarray
    side_px: float
    estimate_k: bool


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)


def detect_frame(name: str, image: np.ndarray, detector: cv2.aruco.ArucoDetector) -> FrameObs:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return FrameObs(name, np.zeros(0, dtype=int), np.zeros((0, 4, 2)), gray.shape[:2])
    ids = ids.ravel().astype(int)
    corners = np.stack([c[0] for c in corners]).astype(np.float64)
    unique, counts = np.unique(ids, return_counts=True)
    keep = np.isin(ids, unique[counts == 1])
    if not keep.all():
        LOGGER.warning("%s: dropping duplicated marker ids %s", name, unique[counts > 1].tolist())
    return FrameObs(name, ids[keep], corners[keep], gray.shape[:2])


def register_pass(
    observations: List[FrameObs],
    pairs: List[PairMatches],
    min_markers: int = 2,
    min_ties: int = 15,
    robust_scale_px: float = 2.0,
    outlier_px: float = 8.0,
    max_rounds: int = 4,
    estimate_height: bool = True,
    max_ties_per_pair: int = 50,
    seed: int = 0,
) -> PassRegistration:
    sides = [_marker_side(o.corners) for o in observations if len(o.ids)]
    if not sides:
        msg = "No markers detected in any frame of the pass"
        raise ValueError(msg)
    side_px = float(np.median(sides))
    h, w = observations[0].shape
    principal = np.array([w / 2, h / 2])
    pairs = _subsample(pairs, max_ties_per_pair, seed)
    poses, transforms = _initialise(observations, pairs, min_markers, side_px)

    excluded_markers: Set[Tuple[int, int]] = set()
    excluded_ties = [np.zeros(len(p.pts_a), bool) for p in pairs]
    k = 1.0
    for _ in range(max_rounds):
        problem = _build_problem(
            observations, pairs, poses, transforms, excluded_markers, excluded_ties, principal, side_px, estimate_height
        )
        poses, transforms, k = _solve(problem, poses, transforms, k, robust_scale_px)
        marker_err = _marker_errors(observations, poses, transforms, side_px)
        tie_err = _tie_errors(pairs, transforms, k, principal, side_px)
        new_markers = {key for key, err in marker_err.items() if err > outlier_px} - excluded_markers
        new_ties = sum(int(((e > outlier_px) & ~x).sum()) for e, x in zip(tie_err, excluded_ties))
        excluded_markers |= new_markers
        excluded_ties = [x | (e > outlier_px) for e, x in zip(tie_err, excluded_ties)]
        transforms = _drop_unconstrained(
            observations, pairs, poses, transforms, excluded_markers, excluded_ties, min_markers, min_ties
        )
        if not new_markers and not new_ties:
            break
    poses, transforms, pitch = _align_to_strip(poses, transforms)

    marker_err = _marker_errors(observations, poses, transforms, side_px, per_corner=True)
    frames = []
    for i, obs in enumerate(observations):
        fit = FrameFit(obs.name, len(obs.ids))
        if i in transforms:
            fit.transform = _wood_transform(transforms[i], k, principal)
            inliers = [
                marker_err[(i, int(m))]
                for m in obs.ids
                if (i, int(m)) in marker_err and (i, int(m)) not in excluded_markers
            ]
            fit.placed_by = "markers" if inliers else "features"
            if inliers:
                fit.reproj_median_px = float(np.median(inliers))
                fit.reproj_max_px = float(np.max(inliers))
            fit.outlier_markers = [int(m) for m in obs.ids if (i, int(m)) in excluded_markers or m not in poses]
        frames.append(fit)
    tie_err = _tie_errors(pairs, transforms, k, principal, side_px)
    seams = {
        (p.a, p.b): float(np.median(e[~x]))
        for p, e, x in zip(pairs, tie_err, excluded_ties)
        if p.a in transforms and p.b in transforms and (~x).any()
    }
    n_tie_outliers = int(sum(x.sum() for x in excluded_ties))
    LOGGER.info(
        "Registered %d/%d frames, height k=%.4f, %d marker and %d tie outliers (> %.1f px)",
        len(transforms),
        len(observations),
        k,
        len(excluded_markers),
        n_tie_outliers,
        outlier_px,
    )
    return PassRegistration(frames, poses, side_px, pitch, k, seams, n_tie_outliers)


def find_gaps(
    names: List[str], transforms: Dict[int, np.ndarray], shapes: List[Tuple[int, int]], ratio: float = 1.5
) -> List[FrameGap]:
    centres = []
    for i in sorted(transforms):
        h, w = shapes[i]
        c = transforms[i] @ np.array([w / 2, h / 2, 1.0])
        centres.append((names[i], c[0] / c[2]))
    if len(centres) < 3:
        return []
    steps = np.abs(np.diff([c[1] for c in centres]))
    median_step = np.median(steps)
    gaps = []
    for (a, _), (b, _), step in zip(centres, centres[1:], steps):
        if step > ratio * median_step:
            gaps.append(FrameGap(a, b, float(step / median_step), int(round(step / median_step)) - 1))
    return gaps


def marker_corners(poses: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    rot = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)
    return np.einsum("mij,cj->mci", rot, TEMPLATE) + poses[:, None, :2]


def _initialise(
    observations: List[FrameObs], pairs: List[PairMatches], min_markers: int, side_px: float
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    seed = next((i for i, o in enumerate(observations) if len(o.ids) >= min_markers), None)
    if seed is None:
        msg = f"No frame has at least {min_markers} markers"
        raise ValueError(msg)
    poses: Dict[int, np.ndarray] = {}
    transforms = {seed: np.diag([1 / side_px, 1 / side_px, 1.0])}
    _add_markers(observations[seed], transforms[seed], poses)
    neighbours: Dict[int, List[PairMatches]] = {}
    for pair in pairs:
        neighbours.setdefault(pair.a, []).append(pair)
        neighbours.setdefault(pair.b, []).append(pair)
    progress = True
    while progress:
        progress = False
        for i, obs in enumerate(observations):
            if i in transforms:
                continue
            transform = _place_by_markers(obs, poses, min_markers)
            if transform is None:
                transform = _place_by_ties(i, neighbours.get(i, []), transforms)
            if transform is None:
                continue
            transforms[i] = transform
            _add_markers(obs, transform, poses)
            progress = True
    return poses, transforms


def _place_by_markers(obs: FrameObs, poses: Dict[int, np.ndarray], min_markers: int) -> Optional[np.ndarray]:
    known = [k for k, m in enumerate(obs.ids) if m in poses]
    if len(known) < min_markers:
        return None
    src = obs.corners[known].reshape(-1, 2)
    dst = marker_corners(np.stack([poses[obs.ids[k]] for k in known])).reshape(-1, 2)
    affine, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.02)
    return None if affine is None else np.vstack([affine, [0, 0, 1]])


def _place_by_ties(index: int, pairs: List[PairMatches], transforms: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    for pair in sorted(pairs, key=lambda p: -len(p.pts_a)):
        other, own, theirs = (pair.b, pair.pts_a, pair.pts_b) if pair.a == index else (pair.a, pair.pts_b, pair.pts_a)
        if other in transforms:
            affine, _ = cv2.estimateAffine2D(own, _apply(transforms[other], theirs), method=cv2.LMEDS)
            if affine is not None:
                return np.vstack([affine, [0, 0, 1]])
    return None


def _add_markers(obs: FrameObs, transform: np.ndarray, poses: Dict[int, np.ndarray]) -> None:
    if len(obs.ids) == 0:
        return
    canon = cv2.perspectiveTransform(obs.corners.reshape(1, -1, 2), transform)[0].reshape(-1, 4, 2)
    for marker_id, corners in zip(obs.ids, canon):
        if marker_id not in poses:
            edge = (corners[1] - corners[0]) + (corners[2] - corners[3])
            poses[marker_id] = np.array([*corners.mean(0), np.arctan2(edge[1], edge[0])])


def _build_problem(
    observations: List[FrameObs],
    pairs: List[PairMatches],
    poses: Dict[int, np.ndarray],
    transforms: Dict[int, np.ndarray],
    excluded_markers: Set[Tuple[int, int]],
    excluded_ties: List[np.ndarray],
    principal: np.ndarray,
    side_px: float,
    estimate_k: bool,
) -> _Problem:
    frames = sorted(transforms)
    frame_index = {f: j for j, f in enumerate(frames)}
    obs = [
        (frame_index[f], int(m), c)
        for f in frames
        for m, c in zip(observations[f].ids, observations[f].corners)
        if m in poses and (f, int(m)) not in excluded_markers
    ]
    markers = sorted({m for _, m, _ in obs})
    if not markers:
        msg = "No marker observation left to fix the board scale; check marker detection"
        raise ValueError(msg)
    marker_index = {m: k for k, m in enumerate(markers)}
    ties = [
        (frame_index[p.a], frame_index[p.b], p.pts_a[~x], p.pts_b[~x])
        for p, x in zip(pairs, excluded_ties)
        if p.a in frame_index and p.b in frame_index and (~x).any()
    ]
    return _Problem(
        frames=frames,
        markers=markers,
        obs_frame=np.array([j for j, _, _ in obs], dtype=int),
        obs_marker=np.array([marker_index[m] for _, m, _ in obs], dtype=int),
        obs_xy=np.stack([c for _, _, c in obs]) if obs else np.zeros((0, 4, 2)),
        tie_a=np.concatenate([np.full(len(t[2]), t[0]) for t in ties]).astype(int) if ties else np.zeros(0, int),
        tie_b=np.concatenate([np.full(len(t[2]), t[1]) for t in ties]).astype(int) if ties else np.zeros(0, int),
        tie_ua=np.concatenate([t[2] for t in ties]) if ties else np.zeros((0, 2)),
        tie_ub=np.concatenate([t[3] for t in ties]) if ties else np.zeros((0, 2)),
        principal=principal,
        side_px=side_px,
        estimate_k=estimate_k,
    )


def _solve(
    problem: _Problem,
    poses: Dict[int, np.ndarray],
    transforms: Dict[int, np.ndarray],
    k: float,
    robust_scale_px: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], float]:
    gauge = problem.markers[0]
    n_frame_cols = FRAME_PARAMS * len(problem.frames)
    n_marker_cols = MARKER_PARAMS * (len(problem.markers) - 1)

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        h = np.concatenate([x[:n_frame_cols].reshape(-1, FRAME_PARAMS), np.ones((len(problem.frames), 1))], 1)
        p = np.concatenate([poses[gauge][None], x[n_frame_cols : n_frame_cols + n_marker_cols].reshape(-1, 3)])
        return h.reshape(-1, 3, 3), p, x[-1] if problem.estimate_k else k

    def residuals(x: np.ndarray) -> np.ndarray:
        return _residuals(problem, *unpack(x))

    def jacobian(x: np.ndarray) -> np.ndarray:
        return _jacobian(problem, *unpack(x), n_frame_cols, len(x))

    x0 = np.concatenate(
        [(transforms[f] / transforms[f][2, 2]).ravel()[:FRAME_PARAMS] for f in problem.frames]
        + [poses[m] for m in problem.markers[1:]]
        + ([np.array([k])] if problem.estimate_k else [])
    )
    with threadpool_limits(BLAS_THREADS):
        result = least_squares(
            residuals,
            x0,
            jac=jacobian,
            loss="soft_l1",
            f_scale=robust_scale_px,
            x_scale="jac",
            method="trf",
            tr_solver="exact",
            bounds=_bounds(len(x0), problem.estimate_k),
        )
    LOGGER.info(
        "Bundle adjustment: %d frames, %d markers, %d marker obs, %d ties, rms %.2f -> %.2f px, %d evaluations",
        len(problem.frames),
        len(problem.markers),
        len(problem.obs_frame),
        len(problem.tie_a),
        np.sqrt(np.mean(residuals(x0) ** 2)),
        np.sqrt(np.mean(result.fun**2)),
        result.nfev,
    )
    h, p, k = unpack(result.x)
    new_poses = dict(poses)
    new_poses.update(zip(problem.markers, p))
    return new_poses, {f: h[j] for j, f in enumerate(problem.frames)}, float(k)


def _residuals(problem: _Problem, h: np.ndarray, p: np.ndarray, k: float) -> np.ndarray:
    corners = marker_corners(p)[problem.obs_marker].reshape(-1, 2)
    obs_frame = np.repeat(problem.obs_frame, 4)
    obs_canon, _, _ = _project(h[obs_frame], problem.obs_xy.reshape(-1, 2))
    ta, _, _ = _project(h[problem.tie_a], _shrink(problem.tie_ua, k, problem.principal))
    tb, _, _ = _project(h[problem.tie_b], _shrink(problem.tie_ub, k, problem.principal))
    return problem.side_px * np.concatenate([(obs_canon - corners).ravel(), (ta - tb).ravel()])


def _jacobian(problem: _Problem, h: np.ndarray, p: np.ndarray, k: float, n_frame_cols: int, n_cols: int) -> np.ndarray:
    n_pts, n_ties = 4 * len(problem.obs_frame), len(problem.tie_a)
    jac = np.zeros((2 * n_pts + 2 * n_ties, n_cols))
    params = np.arange(FRAME_PARAMS)
    obs_frame = np.repeat(problem.obs_frame, 4)
    _, d_h, _ = _project(h[obs_frame], problem.obs_xy.reshape(-1, 2))
    rows = np.arange(2 * n_pts).reshape(n_pts, 2)
    for axis in range(2):
        jac[rows[:, axis, None], FRAME_PARAMS * obs_frame[:, None] + params] = d_h[:, axis]

    obs_marker = np.repeat(problem.obs_marker, 4)
    free = obs_marker > 0
    theta = p[obs_marker, 2]
    tx, ty = np.tile(TEMPLATE[:, 0], len(problem.obs_frame)), np.tile(TEMPLATE[:, 1], len(problem.obs_frame))
    d_theta = np.stack([-np.sin(theta) * tx - np.cos(theta) * ty, np.cos(theta) * tx - np.sin(theta) * ty], -1)
    marker_base = n_frame_cols + MARKER_PARAMS * (obs_marker[free] - 1)
    for axis in range(2):
        jac[rows[free, axis], marker_base + axis] = -1.0
        jac[rows[free, axis], marker_base + 2] = -d_theta[free, axis]

    tie_rows = 2 * n_pts + np.arange(2 * n_ties).reshape(n_ties, 2)
    d_k = np.zeros((n_ties, 2))
    for frame, pts, sign in ((problem.tie_a, problem.tie_ua, 1.0), (problem.tie_b, problem.tie_ub, -1.0)):
        _, d_h, d_x = _project(h[frame], _shrink(pts, k, problem.principal))
        for axis in range(2):
            jac[tie_rows[:, axis, None], FRAME_PARAMS * frame[:, None] + params] = sign * d_h[:, axis]
        d_k += sign * np.einsum("nij,nj->ni", d_x, -(pts - problem.principal) / k**2)
    if problem.estimate_k:
        jac[tie_rows, n_cols - 1] = d_k
    return problem.side_px * jac


def _project(h: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = pts[:, 0], pts[:, 1]
    num = np.einsum("nij,nj->ni", h[:, :2, :2], pts) + h[:, :2, 2]
    w = h[:, 2, 0] * x + h[:, 2, 1] * y + h[:, 2, 2]
    r = num / w[:, None]
    zeros = np.zeros_like(x)
    d_h = (
        np.stack(
            [
                np.stack([x, y, np.ones_like(x), zeros, zeros, zeros, -r[:, 0] * x, -r[:, 0] * y], -1),
                np.stack([zeros, zeros, zeros, x, y, np.ones_like(x), -r[:, 1] * x, -r[:, 1] * y], -1),
            ],
            1,
        )
        / w[:, None, None]
    )
    d_x = (h[:, :2, :2] - r[:, :, None] * h[:, 2, None, :2]) / w[:, None, None]
    return r, d_h, d_x


def _bounds(n: int, estimate_k: bool) -> Tuple[np.ndarray, np.ndarray]:
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    if estimate_k:
        lower[-1], upper[-1] = K_BOUNDS
    return lower, upper


def _shrink(pts: np.ndarray, k: float, principal: np.ndarray) -> np.ndarray:
    return principal + (pts - principal) / k


def _marker_errors(
    observations: List[FrameObs],
    poses: Dict[int, np.ndarray],
    transforms: Dict[int, np.ndarray],
    side_px: float,
    per_corner: bool = False,
) -> Dict[Tuple[int, int], float]:
    errors = {}
    for i, t in transforms.items():
        obs = observations[i]
        for m, c in zip(obs.ids, obs.corners):
            if m in poses:
                canon = _apply(t, c)
                err = side_px * np.linalg.norm(canon - marker_corners(poses[m][None])[0], axis=1)
                errors[(i, int(m))] = float(np.median(err) if per_corner else err.max())
    return errors


def _tie_errors(
    pairs: List[PairMatches], transforms: Dict[int, np.ndarray], k: float, principal: np.ndarray, side_px: float
) -> List[np.ndarray]:
    errors = []
    for p in pairs:
        if p.a not in transforms or p.b not in transforms:
            errors.append(np.zeros(len(p.pts_a)))
            continue
        ta = _apply(_wood_transform(transforms[p.a], k, principal), p.pts_a)
        tb = _apply(_wood_transform(transforms[p.b], k, principal), p.pts_b)
        errors.append(side_px * np.linalg.norm(ta - tb, axis=1))
    return errors


def _drop_unconstrained(
    observations: List[FrameObs],
    pairs: List[PairMatches],
    poses: Dict[int, np.ndarray],
    transforms: Dict[int, np.ndarray],
    excluded_markers: Set[Tuple[int, int]],
    excluded_ties: List[np.ndarray],
    min_markers: int,
    min_ties: int,
) -> Dict[int, np.ndarray]:
    kept = dict(transforms)
    changed = True
    while changed:
        changed = False
        for i in list(kept):
            n_markers = sum(m in poses and (i, int(m)) not in excluded_markers for m in observations[i].ids)
            n_ties = max(
                [
                    int((~x).sum())
                    for p, x in zip(pairs, excluded_ties)
                    if i in (p.a, p.b) and {p.a, p.b} <= kept.keys()
                ],
                default=0,
            )
            if n_markers < min_markers and n_ties < min_ties:
                del kept[i]
                changed = True
    return kept


def _align_to_strip(
    poses: Dict[int, np.ndarray], transforms: Dict[int, np.ndarray]
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], float]:
    ids = np.array(sorted(poses))
    if len(ids) < 2:
        msg = "At least two markers are needed to find the strip axis"
        raise ValueError(msg)
    centres = np.stack([poses[m][:2] for m in ids])
    _, _, vt = np.linalg.svd(centres - centres.mean(0))
    axis = vt[0] if np.corrcoef(ids, (centres - centres.mean(0)) @ vt[0])[0, 1] >= 0 else -vt[0]
    angle = -np.arctan2(axis[1], axis[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = centres @ rot.T
    pitch, intercept = np.polyfit(ids, rotated[:, 0], 1)
    shift = -np.array([intercept, rotated[:, 1].mean()])
    affine = np.eye(3)
    affine[:2, :2] = rot
    affine[:2, 2] = shift
    new_poses = {m: np.array([*(rot @ pose[:2] + shift), pose[2] + angle]) for m, pose in poses.items()}
    return new_poses, {f: affine @ t for f, t in transforms.items()}, float(pitch)


def _wood_transform(transform: np.ndarray, k: float, principal: np.ndarray) -> np.ndarray:
    shrink = np.array([[1 / k, 0, principal[0] * (1 - 1 / k)], [0, 1 / k, principal[1] * (1 - 1 / k)], [0, 0, 1]])
    return transform @ shrink


def _apply(transform: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(1, -1, 2).astype(np.float64), transform)[0]


def _subsample(pairs: List[PairMatches], max_ties: int, seed: int) -> List[PairMatches]:
    rng = np.random.default_rng(seed)
    out = []
    for p in pairs:
        if len(p.pts_a) > max_ties:
            keep = np.sort(rng.choice(len(p.pts_a), max_ties, replace=False))
            p = PairMatches(p.a, p.b, p.pts_a[keep], p.pts_b[keep])
        out.append(p)
    return out


def _marker_side(corners: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2).mean())
