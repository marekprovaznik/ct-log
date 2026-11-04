import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_slider.registration import Distortion, undistort_maps, undistort_points
from camera_slider.wood import segment_board

NO_OWNER = 255


@dataclasses.dataclass
class FaceRender:
    image: np.ndarray
    coverage: np.ndarray
    origin: Tuple[float, float]
    px_per_unit: float
    owner: np.ndarray


@dataclasses.dataclass
class FaceCrop:
    image: np.ndarray
    mask: np.ndarray
    box: Tuple[int, int, int, int]
    x_range: Tuple[float, float]
    uncovered_columns: int
    owner: np.ndarray
    origin: Tuple[float, float]


@dataclasses.dataclass
class FacePlacement:
    x: int
    y: int
    width: int
    height: int
    scale_x: float


def render_face(
    frames: List[Tuple[str, np.ndarray, Tuple[int, int]]],
    load: Callable[[str], np.ndarray],
    px_per_unit: float,
    tile_width: int = 8192,
    distortion: Optional[Distortion] = None,
) -> FaceRender:
    if len(frames) >= NO_OWNER:
        msg = f"At most {NO_OWNER - 1} frames per face fit the owner map, got {len(frames)}"
        raise ValueError(msg)
    footprints = np.concatenate([_footprint(t, s, distortion) for _, t, s in frames])
    origin = footprints.min(0)
    width, height = np.ceil((footprints.max(0) - origin) * px_per_unit).astype(int) + 1
    to_canvas = np.array(
        [[px_per_unit, 0, -origin[0] * px_per_unit], [0, px_per_unit, -origin[1] * px_per_unit], [0, 0, 1]]
    )
    placed = []
    for name, transform, shape in frames:
        mapping = to_canvas @ transform
        corners = _footprint(mapping, shape, distortion)
        lo = np.clip(np.floor(corners.min(0)).astype(int), 0, [width, height])
        hi = np.clip(np.ceil(corners.max(0)).astype(int) + 1, 0, [width, height])
        placed.append((name, mapping, shape, lo, hi))
    image = np.zeros((height, width, 3), np.uint8)
    coverage = np.zeros((height, width), bool)
    owner = np.full((height, width), NO_OWNER, np.uint8)
    cache: Dict[str, np.ndarray] = {}
    feathers: Dict[Tuple[int, int], np.ndarray] = {}
    maps: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for tx0 in range(0, width, tile_width):
        tx1 = min(width, tx0 + tile_width)
        acc = np.zeros((height, tx1 - tx0, 3), np.float32)
        weight = np.zeros((height, tx1 - tx0), np.float32)
        best = np.zeros((height, tx1 - tx0), np.float32)
        own = owner[:, tx0:tx1]
        active = [(index, *p) for index, p in enumerate(placed) if p[3][0] < tx1 and p[4][0] > tx0]
        cache = {name: cache[name] for _, name, *_ in active if name in cache}
        for index, name, mapping, shape, lo, hi in active:
            if distortion is not None and shape not in maps:
                maps[shape] = undistort_maps(distortion, shape)
            if name not in cache:
                cache[name] = _undistort_image(load(name).astype(np.float32), maps.get(shape))
            if shape not in feathers:
                feathers[shape] = _undistort_image(_feather(shape), maps.get(shape))
            x0, x1 = max(lo[0], tx0), min(hi[0], tx1)
            y0, y1 = lo[1], hi[1]
            roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ mapping
            size = (int(x1 - x0), int(y1 - y0))
            w = cv2.warpPerspective(feathers[shape], roi, size, flags=cv2.INTER_LINEAR)
            warped = cv2.warpPerspective(cache[name], roi, size, flags=cv2.INTER_LINEAR)
            acc[y0:y1, x0 - tx0 : x1 - tx0] += warped * w[..., None]
            weight[y0:y1, x0 - tx0 : x1 - tx0] += w
            wins = w > best[y0:y1, x0 - tx0 : x1 - tx0]
            best[y0:y1, x0 - tx0 : x1 - tx0][wins] = w[wins]
            own[y0:y1, x0 - tx0 : x1 - tx0][wins] = index
        image[:, tx0:tx1] = (acc / np.maximum(weight, 1e-6)[..., None]).clip(0, 255).astype(np.uint8)
        coverage[:, tx0:tx1] = weight > 0
    return FaceRender(image, coverage, (float(origin[0]), float(origin[1])), px_per_unit, owner)


def crop_board(render: FaceRender) -> FaceCrop:
    board = segment_board(render.image) & render.coverage
    ys, xs = np.nonzero(board)
    if len(xs) == 0:
        msg = "No timber found in the rendered face"
        raise ValueError(msg)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    band = render.coverage[y0 + (y1 - y0) // 4 : y1 - (y1 - y0) // 4, x0:x1]
    uncovered = int((~band.all(0)).sum())
    mask = board[y0:y1, x0:x1]
    image = render.image[y0:y1, x0:x1] * mask[..., None]
    x_range = (render.origin[0] + x0 / render.px_per_unit, render.origin[0] + x1 / render.px_per_unit)
    return FaceCrop(image, mask, (x0, y0, x1, y1), x_range, uncovered, render.owner[y0:y1, x0:x1], render.origin)


def layout_board(
    crops: List[FaceCrop], gap: int = 40, stretch: bool = True, margin: int = 40
) -> Tuple[np.ndarray, List[float]]:
    placements = layout_placements([c.image.shape[:2] for c in crops], gap, stretch, margin)
    width = max(p.width for p in placements) + 2 * margin
    height = sum(p.height for p in placements) + gap * (len(placements) - 1) + 2 * margin
    board = np.zeros((height, width, 3), np.uint8)
    for crop, p in zip(crops, placements):
        image = (
            cv2.resize(crop.image, (p.width, p.height), interpolation=cv2.INTER_LINEAR)
            if p.width != crop.image.shape[1]
            else crop.image
        )
        board[p.y : p.y + p.height, p.x : p.x + p.width] = image
    return board, [p.scale_x for p in placements]


def layout_placements(
    shapes: List[Tuple[int, int]], gap: int = 40, stretch: bool = True, margin: int = 40
) -> List[FacePlacement]:
    widths = [w for _, w in shapes]
    length = int(np.median(widths))
    out_widths = [length if stretch else w for w in widths]
    width = max(out_widths) + 2 * margin
    placements = []
    y = margin
    for (h, w), out_w in zip(shapes, out_widths):
        placements.append(FacePlacement((width - out_w) // 2, y, out_w, h, out_w / w))
        y += h + gap
    return placements


def _footprint(hom: np.ndarray, shape: Tuple[int, int], distortion: Optional[Distortion] = None) -> np.ndarray:
    h, w = shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, 0], [w, h / 2], [w / 2, h], [0, h / 2]], np.float64)
    if distortion is not None:
        corners = undistort_points(corners, distortion)
    return cv2.perspectiveTransform(corners[None], hom)[0]


def _undistort_image(image: np.ndarray, maps: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if maps is None:
        return image
    return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _feather(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32) + 1
    return (dist / dist.max()) ** 2
