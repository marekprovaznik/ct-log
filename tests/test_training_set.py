import base64
from pathlib import Path
import zlib

from camera_slider.training_set import (
    Piece,
    downsample_labels,
    group_strips,
    rasterise_object,
    tile_starts,
    wood_components,
)
import cv2
import numpy as np


def _bitmap_obj(mask: np.ndarray, origin: list) -> dict:
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask, 3] = 255
    ok, png = cv2.imencode(".png", rgba)
    data = base64.b64encode(zlib.compress(png.tobytes())).decode()
    return {"classTitle": "knot", "geometryType": "bitmap", "bitmap": {"data": data, "origin": origin}}


def test_polygon_bbox_and_mask():
    """A polygon rasterises to its bbox with the interior filled."""
    obj = {
        "classTitle": "knot",
        "geometryType": "polygon",
        "points": {"exterior": [[10, 20], [30, 20], [30, 40], [10, 40]], "interior": []},
    }
    mask, bbox = rasterise_object(obj, 100, 100)
    assert bbox == (10, 20, 31, 41)
    assert mask.shape == (21, 21)
    assert mask.all()


def test_bitmap_origin():
    """A bitmap keeps its origin offset."""
    local = np.zeros((5, 7), bool)
    local[1:4, 2:6] = True
    mask, bbox = rasterise_object(_bitmap_obj(local, [50, 60]), 100, 100)
    assert bbox == (52, 61, 56, 64)
    assert mask.all()


def test_downsample_labels_ties_go_to_higher_priority():
    """Area-averaged argmax keeps a half-covered defect and breaks ties towards later classes."""
    native = np.zeros((4, 4), np.uint8)
    native[:, :2] = 1
    native[:2, :2] = 2
    native[2:, 2:] = 1
    out = downsample_labels(native, (2, 2), [0, 1, 2])
    assert out.tolist() == [[2, 0], [1, 1]]


def test_tile_starts_cover_end():
    """Tiles cover the strip end and keep the stride."""
    assert tile_starts(2000, 800, 600) == [0, 600, 1200]
    assert tile_starts(700, 800, 600) == [0]


def test_group_strips_splits_on_gap():
    """Non-consecutive pieces of one board form separate strips."""
    pieces = [Piece(f"b_{x}", "spruce_001", "spruce", x, Path("."), None, 100, 10) for x in (0, 100, 300)]
    strips = group_strips(pieces)
    assert [[p.x for p in s] for s in strips] == [[0, 100], [300]]


def test_wood_components_keeps_n_largest():
    """Only the N largest bright components survive, with holes filled."""
    image = np.zeros((50, 50, 3), np.uint8)
    image[2:20, 2:48] = 100
    image[25:45, 2:48] = 100
    image[30:35, 10:15] = 0
    image[47:49, 47:49] = 100
    mask, areas = wood_components(image, 2, 5)
    assert len(areas) == 2
    assert mask[32, 12]
    assert not mask[48, 48]
