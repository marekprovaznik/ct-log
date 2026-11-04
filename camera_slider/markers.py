import dataclasses
from typing import List, Optional, Tuple

import cv2
import numpy as np

STRIP_DICT = cv2.aruco.DICT_7X7_100
CELLS = 9
CELL_PX = 16
MIN_SIDE_PX = 60
MAX_BORDER_WHITE = 4
MAX_BITS_FREE = 4
MAX_BITS_ORDERED = 12
MIN_BIT_MARGIN = 6
PITCH_RANGE = (1.0, 1.3)
PITCH_PRIOR = 1.12
SEED_MAX_BITS = 2
SEED_MIN_MARGIN = 10
ORDER_TOLERANCE = 0.2
ACROSS_TOLERANCE = 0.25
RECOVER_MIN_VISIBLE = 0.5
RECOVER_MIN_ECC = 0.8
RECOVER_MAX_BIT_ERROR_RATE = 0.12
RECOVER_MAX_SHIFT = 0.25
INNER_MASK_CELLS = 0.7
SEPARATOR_SCALE = 4
SEPARATOR_MIN_H = 25
SEPARATOR_BRIGHT = 90
SEPARATOR_PITCH_TOLERANCE = 0.15
PARTIAL_MIN_SEEN = 0.5
PARTIAL_MAX_ERROR_RATE = 0.1
PARTIAL_MIN_MARGIN_RATE = 0.2
SOURCE_DECODED = 0
SOURCE_ORDERED = 1
SOURCE_RECOVERED = 2
TEMPLATE = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


@dataclasses.dataclass
class Candidate:
    corners: np.ndarray
    marker_id: int
    bit_errors: int
    margin: int
    seeded: bool = False


@dataclasses.dataclass
class StripModel:
    axis: np.ndarray
    offset: float
    pitch: float
    across: float
    side: float
    quarter_turns: int

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.axis[1], self.axis[0]])

    def predicted_id(self, centres: np.ndarray) -> np.ndarray:
        return (centres @ self.axis - self.offset) / self.pitch


@dataclasses.dataclass
class Markers:
    ids: np.ndarray
    corners: np.ndarray
    bit_errors: np.ndarray
    source: np.ndarray
    n_aruco: int


def make_aruco() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.perspectiveRemovePixelPerCell = 10
    params.errorCorrectionRate = 1.0
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(STRIP_DICT), params)


def detect_markers(gray: np.ndarray, detector: cv2.aruco.ArucoDetector, recover: bool = True) -> Markers:
    corners, ids, rejected = detector.detectMarkers(gray)
    n_aruco = 0 if ids is None else len(ids)
    codes = dictionary_bits()
    quads = [c[0] for c in corners] + [r[0] for r in rejected]
    candidates = _dedupe([c for c in (decode_candidate(gray, q.astype(np.float64), codes) for q in quads) if c])
    model = fit_strip(candidates)
    if model is None and recover:
        candidates = _dedupe(candidates + seed_neighbours(gray, candidates, codes))
        model = fit_strip(candidates)
    if model is None and recover:
        candidates = _dedupe(candidates + seed_from_separators(gray, codes))
        model = fit_strip(candidates)
    if model is None:
        strong = _unique_ids([c for c in candidates if _strong(c)])
        return _markers(strong, [SOURCE_DECODED] * len(strong), n_aruco)
    accepted, sources = accept_ordered(gray, candidates, model, codes)
    accepted = [refine_candidate(gray, c, codes) for c in accepted]
    if recover:
        found = recover_markers(gray, accepted, model, codes)
        accepted += found
        sources += [SOURCE_RECOVERED] * len(found)
    order = np.argsort([c.marker_id for c in accepted])
    return _markers([accepted[k] for k in order], [sources[k] for k in order], n_aruco)


def dictionary_bits() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(STRIP_DICT)
    n = dictionary.bytesList.shape[0]
    return np.stack([dictionary.getBitsFromByteList(dictionary.bytesList[i : i + 1], 7) for i in range(n)])


def marker_template(bits: np.ndarray, cell_px: int = CELL_PX) -> np.ndarray:
    grid = np.zeros((CELLS, CELLS), np.uint8)
    grid[1:-1, 1:-1] = bits * 255
    return cv2.resize(grid, (CELLS * cell_px, CELLS * cell_px), interpolation=cv2.INTER_NEAREST)


def read_cells(gray: np.ndarray, corners: np.ndarray, cell_px: int = CELL_PX) -> Tuple[np.ndarray, np.ndarray]:
    n = CELLS * cell_px
    hom = cv2.getPerspectiveTransform(corners.astype(np.float32), _square(n))
    patch = cv2.warpPerspective(gray, hom, (n, n), flags=cv2.INTER_LINEAR)
    inside = cv2.warpPerspective(np.full(gray.shape, 255, np.uint8), hom, (n, n), flags=cv2.INTER_NEAREST) > 0
    q = cell_px // 4
    cells = np.zeros((CELLS, CELLS))
    seen = np.zeros((CELLS, CELLS), bool)
    for i in range(CELLS):
        for j in range(CELLS):
            box = (slice(i * cell_px + q, (i + 1) * cell_px - q), slice(j * cell_px + q, (j + 1) * cell_px - q))
            seen[i, j] = inside[box].all()
            cells[i, j] = np.median(patch[box])
    values = patch[inside]
    if len(values) == 0:
        return np.zeros((CELLS, CELLS), np.uint8), seen
    threshold = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    return (cells > threshold).astype(np.uint8), seen


def decode_candidate(gray: np.ndarray, corners: np.ndarray, codes: np.ndarray) -> Optional[Candidate]:
    if _side(corners) < MIN_SIDE_PX:
        return None
    bits, seen = read_cells(gray, corners)
    if not seen.all():
        return None
    border = np.r_[bits[0], bits[-1], bits[1:-1, 0], bits[1:-1, -1]]
    if border.sum() > MAX_BORDER_WHITE:
        return None
    inner = bits[1:-1, 1:-1]
    errors = np.stack([(codes != np.rot90(inner, -r)[None]).sum((1, 2)) for r in range(4)], 1)
    best_id, rotation = divmod(int(np.argmin(errors)), 4)
    best = int(errors[best_id, rotation])
    second = int(np.min(np.delete(errors, best_id, 0)))
    return Candidate(np.roll(corners, -rotation, 0), best_id, best, second - best)


def fit_strip(candidates: List[Candidate]) -> Optional[StripModel]:
    strong = _unique_ids([c for c in candidates if _strong(c)])
    if len(strong) < 2:
        return None
    centres = np.stack([c.corners.mean(0) for c in strong])
    ids = np.array([c.marker_id for c in strong], float)
    sides = np.array([_side(c.corners) for c in strong])
    side = float(np.median(sides))
    best: Optional[np.ndarray] = None
    for a in range(len(strong)):
        for b in range(a + 1, len(strong)):
            step = (centres[b] - centres[a]) / (ids[b] - ids[a])
            pitch = float(np.linalg.norm(step))
            if not PITCH_RANGE[0] <= pitch / side <= PITCH_RANGE[1]:
                continue
            axis = step / pitch
            normal = np.array([-axis[1], axis[0]])
            along = centres @ axis - centres[a] @ axis - pitch * (ids - ids[a])
            across = (centres - centres[a]) @ normal
            inliers = (np.abs(along) < ORDER_TOLERANCE * pitch) & (np.abs(across) < ACROSS_TOLERANCE * side)
            if best is None or inliers.sum() > best[0].sum():
                best = (inliers, axis)
    if best is None or best[0].sum() < 2:
        return None
    inliers, axis = best
    pitch, offset = np.polyfit(ids[inliers], centres[inliers] @ axis, 1)
    if pitch < 0:
        msg = "Strip pitch flipped sign during the refit"
        raise ValueError(msg)
    normal = np.array([-axis[1], axis[0]])
    turns = [_quarter_turns(c.corners, axis) for k, c in enumerate(strong) if inliers[k]]
    return StripModel(
        axis,
        float(offset),
        float(pitch),
        float(np.median(centres[inliers] @ normal)),
        float(np.median(sides[inliers])),
        int(np.bincount(turns, minlength=4).argmax()),
    )


def seed_neighbours(gray: np.ndarray, candidates: List[Candidate], codes: np.ndarray) -> List[Candidate]:
    seeds = [c for c in candidates if c.bit_errors <= SEED_MAX_BITS and c.margin >= SEED_MIN_MARGIN]
    if len(seeds) != 1:
        return []
    seed = seeds[0]
    corners = seed.corners
    edges = [corners[1] - corners[0], corners[3] - corners[0]]
    best: List[Candidate] = []
    for direction in [edges[0], -edges[0], edges[1], -edges[1]]:
        step = direction / np.linalg.norm(direction) * PITCH_PRIOR * _side(corners)
        found = []
        for k in (1, -1, 2, -2):
            marker_id = seed.marker_id + k
            if not 0 <= marker_id < len(codes):
                continue
            refined = refine_marker(gray, corners + step * k, codes[marker_id])
            if refined is not None:
                found.append(Candidate(refined[0], marker_id, refined[1], SEED_MIN_MARGIN, True))
        if len(found) > len(best):
            best = found
    return best


def seed_from_separators(gray: np.ndarray, codes: np.ndarray) -> List[Candidate]:
    bars = separator_bars(gray)
    if len(bars) < 2:
        return []
    gaps = np.diff([b[0] for b in bars])
    pitch = float(np.median(gaps))
    bottom = float(np.median([b[1] + b[3] for b in bars]))
    side = pitch / PITCH_PRIOR
    found: List[Candidate] = []
    for a, b, gap in zip(bars, bars[1:], gaps):
        if abs(gap / pitch - 1) > SEPARATOR_PITCH_TOLERANCE:
            continue
        centre_x = (a[0] + b[0]) / 2
        quad = np.array(
            [
                [centre_x - side / 2, bottom - side],
                [centre_x + side / 2, bottom - side],
                [centre_x + side / 2, bottom],
                [centre_x - side / 2, bottom],
            ]
        )
        candidate = decode_partial(gray, quad, codes)
        if candidate is not None:
            found.append(candidate)
    return _unique_ids(_dedupe(found))


def separator_bars(gray: np.ndarray) -> List[Tuple[float, float, float, float]]:
    small = cv2.resize(gray, None, fx=1 / SEPARATOR_SCALE, fy=1 / SEPARATOR_SCALE, interpolation=cv2.INTER_AREA)
    n, _, stats, _ = cv2.connectedComponentsWithStats((small > SEPARATOR_BRIGHT).astype(np.uint8))
    bars = [
        tuple(float(v) * SEPARATOR_SCALE for v in stats[k, :4])
        for k in range(1, n)
        if stats[k, 3] >= SEPARATOR_MIN_H and stats[k, 2] < stats[k, 3] / 3
    ]
    centred = sorted(((x + w / 2, y, w, h) for x, y, w, h in bars), key=lambda b: b[0])
    if not centred:
        return []
    height = float(np.median([b[3] for b in centred]))
    return [b for b in centred if b[3] >= 0.5 * height]


def decode_partial(gray: np.ndarray, quad: np.ndarray, codes: np.ndarray) -> Optional[Candidate]:
    scored = []
    for rotation in range(4):
        bits, seen = read_cells(gray, np.roll(quad, -rotation, 0))
        inner, inner_seen = bits[1:-1, 1:-1], seen[1:-1, 1:-1]
        if inner_seen.mean() < PARTIAL_MIN_SEEN:
            return None
        rates = (codes != inner[None])[:, inner_seen].mean(1)
        scored += [(float(rate), k, rotation) for k, rate in enumerate(rates)]
    scored.sort()
    rate, marker_id, rotation = scored[0]
    second = next(r for r, k, _ in scored if k != marker_id)
    if rate > PARTIAL_MAX_ERROR_RATE or second - rate < PARTIAL_MIN_MARGIN_RATE:
        return None
    refined = refine_marker(gray, np.roll(quad, -rotation, 0), codes[marker_id])
    if refined is None:
        return None
    return Candidate(refined[0], marker_id, refined[1], SEED_MIN_MARGIN, True)


def accept_ordered(
    gray: np.ndarray, candidates: List[Candidate], model: StripModel, codes: np.ndarray
) -> Tuple[List[Candidate], List[int]]:
    by_id = {}
    for c in candidates:
        centre = c.corners.mean(0)
        predicted = float(model.predicted_id(centre[None])[0])
        marker_id = int(round(predicted))
        if not 0 <= marker_id < len(codes) or abs(predicted - marker_id) > ORDER_TOLERANCE:
            continue
        if abs(centre @ model.normal - model.across) > ACROSS_TOLERANCE * model.side:
            continue
        if c.marker_id == marker_id and c.bit_errors <= MAX_BITS_FREE:
            source, accepted = (SOURCE_RECOVERED if c.seeded else SOURCE_DECODED), c
        else:
            rotated = _orient(c.corners, model)
            bits, _ = read_cells(gray, rotated)
            errors = int((bits[1:-1, 1:-1] != codes[marker_id]).sum())
            if errors > MAX_BITS_ORDERED:
                continue
            source, accepted = SOURCE_ORDERED, Candidate(rotated, marker_id, errors, 0)
        if marker_id not in by_id or accepted.bit_errors < by_id[marker_id][0].bit_errors:
            by_id[marker_id] = (accepted, source)
    keys = sorted(by_id)
    return [by_id[k][0] for k in keys], [by_id[k][1] for k in keys]


def recover_markers(
    gray: np.ndarray, accepted: List[Candidate], model: StripModel, codes: np.ndarray
) -> List[Candidate]:
    if len(accepted) < 2:
        return []
    src = np.concatenate([strip_corners(c.marker_id, model) for c in accepted])
    dst = np.concatenate([c.corners for c in accepted])
    hom, _ = cv2.findHomography(src, dst, 0)
    if hom is None:
        return []
    h, w = gray.shape
    margin = 0.5 * model.side
    known = {c.marker_id for c in accepted}
    found = []
    for marker_id in range(len(codes)):
        if marker_id in known:
            continue
        corners = cv2.perspectiveTransform(strip_corners(marker_id, model)[None], hom)[0]
        x, y = corners.mean(0)
        if not (-margin <= x <= w + margin and -margin <= y <= h + margin):
            continue
        refined = refine_marker(gray, corners, codes[marker_id])
        if refined is not None:
            found.append(Candidate(refined[0], marker_id, refined[1], 0))
    return found


def strip_corners(marker_id: int, model: StripModel) -> np.ndarray:
    angle = model.quarter_turns * np.pi / 2
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return TEMPLATE @ rot.T + np.array([model.pitch / model.side * marker_id, 0.0])


def refine_candidate(gray: np.ndarray, candidate: Candidate, codes: np.ndarray) -> Candidate:
    refined = refine_marker(gray, candidate.corners, codes[candidate.marker_id])
    if refined is None:
        return candidate
    return dataclasses.replace(candidate, corners=refined[0])


def refine_marker(gray: np.ndarray, corners: np.ndarray, code: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
    n = CELLS * CELL_PX
    square = _square(n)
    to_patch = cv2.getPerspectiveTransform(corners.astype(np.float32), square)
    patch = cv2.warpPerspective(gray, to_patch, (n, n), flags=cv2.INTER_LINEAR)
    visible = cv2.warpPerspective(np.full(gray.shape, 255, np.uint8), to_patch, (n, n), flags=cv2.INTER_NEAREST)
    if (visible > 0).mean() < RECOVER_MIN_VISIBLE:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        ecc, warp = cv2.findTransformECC(
            marker_template(code).astype(np.float32),
            patch.astype(np.float32),
            np.eye(3, dtype=np.float32),
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            cv2.erode(visible, np.ones((5, 5), np.uint8)) & _inner_mask(n),
            3,
        )
    except cv2.error:
        return None
    if ecc < RECOVER_MIN_ECC:
        return None
    in_patch = cv2.perspectiveTransform(square[None].astype(np.float64), warp.astype(np.float64))[0]
    if np.abs(in_patch - square).max() > RECOVER_MAX_SHIFT * n:
        return None
    refined = cv2.perspectiveTransform(in_patch[None], np.linalg.inv(to_patch).astype(np.float64))[0]
    bits, seen = read_cells(gray, refined)
    inner, inner_seen = bits[1:-1, 1:-1], seen[1:-1, 1:-1]
    errors = int((inner != code)[inner_seen].sum())
    if errors > RECOVER_MAX_BIT_ERROR_RATE * max(1, inner_seen.sum()):
        return None
    return refined, errors


def _strong(c: Candidate) -> bool:
    return c.bit_errors <= MAX_BITS_FREE and c.margin >= MIN_BIT_MARGIN


def _quarter_turns(corners: np.ndarray, axis: np.ndarray) -> int:
    x_axis = corners[1] - corners[0] + corners[2] - corners[3]
    normal = np.array([-axis[1], axis[0]])
    angle = np.arctan2(x_axis @ normal, x_axis @ axis)
    return int(np.round(angle / (np.pi / 2))) % 4


def _orient(corners: np.ndarray, model: StripModel) -> np.ndarray:
    for rotation in range(4):
        rotated = np.roll(corners, -rotation, 0)
        if _quarter_turns(rotated, model.axis) == model.quarter_turns:
            return rotated
    return corners


def _inner_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), np.uint8)
    inset = int(INNER_MASK_CELLS * CELL_PX)
    mask[inset : n - inset, inset : n - inset] = 255
    return mask


def _square(n: int) -> np.ndarray:
    return np.float32([[0, 0], [n, 0], [n, n], [0, n]])


def _side(corners: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(corners, -1, 0) - corners, axis=1).mean())


def _dedupe(candidates: List[Candidate]) -> List[Candidate]:
    kept: List[Candidate] = []
    for c in sorted(candidates, key=lambda c: c.bit_errors):
        centre = c.corners.mean(0)
        if all(np.linalg.norm(k.corners.mean(0) - centre) > 0.5 * _side(c.corners) for k in kept):
            kept.append(c)
    return kept


def _unique_ids(candidates: List[Candidate]) -> List[Candidate]:
    ids = [c.marker_id for c in candidates]
    return [c for c in candidates if ids.count(c.marker_id) == 1]


def _markers(candidates: List[Candidate], sources: List[int], n_aruco: int) -> Markers:
    if not candidates:
        return Markers(np.zeros(0, int), np.zeros((0, 4, 2)), np.zeros(0, int), np.zeros(0, int), n_aruco)
    return Markers(
        np.array([c.marker_id for c in candidates]),
        np.stack([c.corners for c in candidates]),
        np.array([c.bit_errors for c in candidates]),
        np.array(sources),
        n_aruco,
    )
