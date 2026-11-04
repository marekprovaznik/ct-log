from pathlib import Path
from typing import Dict

from camera_slider.compose import NO_OWNER, FaceCrop, FacePlacement, layout_board, layout_placements, render_face
from camera_slider.mapping import (
    BoardMap,
    FaceMap,
    crop_to_frame,
    distort_points,
    face_to_unrolled,
    frame_to_crop,
    load_mapping,
    owners,
    unrolled_to_face,
    write_mapping,
)
from camera_slider.registration import Distortion, FrameFit, PassRegistration, undistort_points
from camera_slider.transfer_annotations import RowShift, builder_to_edited, edited_to_builder, fit_piece, in_boxes
import cv2
import numpy as np
import pytest

FRAME_SHAPE = (120, 160)
PX_PER_UNIT = 10.0


@pytest.fixture(scope="module")
def stitched(tmp_path_factory: pytest.TempPathFactory) -> Dict:
    """Two faces of two translated frames each, rendered, laid out and written as a mapping."""
    out_dir = tmp_path_factory.mktemp("board")
    distortion = Distortion(-0.01, 0.002, (FRAME_SHAPE[1] / 2, FRAME_SHAPE[0] / 2), float(np.hypot(*FRAME_SHAPE) / 2))
    rng = np.random.default_rng(0)
    images = {
        f"f{i}": np.clip(
            (cv2.GaussianBlur(rng.uniform(0, 255, (*FRAME_SHAPE, 3)), (0, 0), 6) - 127) * 8 + 127, 0, 255
        ).astype(np.uint8)
        for i in range(4)
    }
    groups, geometries, crops = [], [], []
    for face in range(2):
        names = [f"f{2 * face}", f"f{2 * face + 1}"]
        fits = [
            FrameFit(n, 4, np.array([[0.1, 0, 10.0 * i], [0, 0.1, 0.5 * face], [0, 0, 1]])) for i, n in enumerate(names)
        ]
        render = render_face(
            [(f.name, f.transform, FRAME_SHAPE) for f in fits], images.__getitem__, PX_PER_UNIT, distortion=distortion
        )
        h, w = render.owner.shape
        crops.append(
            FaceCrop(render.image, np.ones((h, w), bool), (0, 0, w, h), (0.0, 0.0), 0, render.owner, render.origin)
        )
        groups.append([out_dir / n for n in names])
        geometries.append(PassRegistration(fits, {}, 20.0, 1.0, 1.0, {}, 0, distortion))
    unrolled, _ = layout_board([crops[1], crops[0]], 10, True, 5)
    pieces = [{"name": "b_x00000.jpg", "x0": 0, "x1": unrolled.shape[1]}]
    write_mapping(
        out_dir,
        "b",
        groups,
        geometries,
        crops,
        [FRAME_SHAPE] * 2,
        ["A", "B"],
        [1, 0],
        PX_PER_UNIT,
        unrolled.shape[:2],
        pieces,
        10,
        True,
        5,
    )
    return {"dir": out_dir, "crops": crops, "unrolled": unrolled, "images": images}


def test_owner_map_covers_both_frames(stitched: Dict) -> None:
    """The owner map assigns every covered pixel to one of the face's frames."""
    owner = stitched["crops"][0].owner
    assert set(np.unique(owner)) <= {0, 1, NO_OWNER}
    middle = owner.shape[0] // 2
    assert (owner[middle, : owner.shape[1] // 4] == 0).all()
    assert (owner[middle, -owner.shape[1] // 4 : -2] == 1).all()


def test_mapping_round_trip_is_exact(stitched: Dict) -> None:
    """Unrolled -> frame -> unrolled returns the same position."""
    board = load_mapping(stitched["dir"])
    h, w = stitched["unrolled"].shape[:2]
    ys, xs = np.mgrid[0:h:3, 0:w:3]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    rows, crop = unrolled_to_face(board, pts)
    checked = 0
    for k, face in enumerate(board.faces):
        sel = np.nonzero(rows == k)[0]
        own = owners(face, crop[sel])
        for index in np.unique(own[own != NO_OWNER]):
            hit = sel[own == index]
            raw = crop_to_frame(face, crop[hit], face.frames[index])
            back = face_to_unrolled(face, frame_to_crop(face, raw, face.frames[index]))
            assert np.abs(back - pts[hit]).max() < 1e-6
            checked += len(hit)
    assert checked > len(pts) // 2


def test_mapped_pixels_show_the_frame_content(stitched: Dict) -> None:
    """A pixel deep inside one frame shows that frame's raw pixel at the mapped position."""
    board = load_mapping(stitched["dir"])
    face = board.faces[0]
    crop = np.array([[30.0, 4.0], [40.0, 6.0]])
    own = owners(face, crop)
    raw = crop_to_frame(face, crop, face.frames[own[0]])
    image = stitched["images"][face.frames[own[0]].name].astype(np.float32)
    got = cv2.remap(image, raw[:, :1].astype(np.float32), raw[:, 1:].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
    unrolled = face_to_unrolled(face, crop)
    ref = stitched["unrolled"][np.round(unrolled[:, 1]).astype(int), np.round(unrolled[:, 0]).astype(int)]
    assert np.abs(got - ref).mean() < 12


def test_layout_rows_follow_the_face_order(stitched: Dict) -> None:
    """Face B, ordered first, sits in row 0 of the unrolled board."""
    board = load_mapping(stitched["dir"])
    assert [f.area for f in board.faces] == ["B", "A"]
    assert board.faces[0].placement.y == 5


def test_distort_points_inverts_undistort() -> None:
    """distort_points is the inverse of the radial undistortion."""
    distortion = Distortion(-0.02, 0.004, (80.0, 60.0), 100.0)
    pts = np.random.default_rng(1).uniform(0, 160, (200, 2))
    assert np.abs(undistort_points(distort_points(pts, distortion), distortion) - pts).max() < 1e-9


def test_layout_placements_stretch_to_median() -> None:
    """Faces are stretched to the median length and stacked with the gap."""
    placements = layout_placements([(10, 100), (20, 110), (5, 90)], gap=3, stretch=True, margin=2)
    assert [p.width for p in placements] == [100, 100, 100]
    assert [p.y for p in placements] == [2, 15, 38]
    assert placements[1].scale_x == pytest.approx(100 / 110)


def test_edited_shift_round_trip() -> None:
    """Row shifts of a hand-edited piece undo and redo exactly; gap pixels go to the nearest row."""
    shifts = [RowShift(0, (0, 10), 0, 0, 0.1, True), RowShift(1, (14, 20), 0, 30, 0.1, True)]
    pts = np.array([[5.0, 3.0], [5.0, 46.0], [5.0, 41.0], [5.0, 25.0]])
    rows, builder = edited_to_builder(shifts, pts)
    assert rows.tolist() == [0, 1, 1, 0]
    assert np.allclose(builder_to_edited(shifts, rows, builder), pts)


def test_missing_mapping_raises(tmp_path: Path) -> None:
    """Loading a board without mapping.json fails loudly."""
    with pytest.raises(FileNotFoundError):
        load_mapping(tmp_path)


def test_fit_piece_recovers_row_shift_and_leftover() -> None:
    """A row moved down by hand is fitted exactly and a stray patch is reported as unexplained."""
    rng = np.random.default_rng(2)
    builder = np.zeros((200, 300), np.uint8)
    builder[10:60] = _texture(rng, 50)
    builder[80:110] = _texture(rng, 30)
    edited = np.zeros_like(builder)
    edited[10:60] = builder[10:60]
    edited[150:180] = builder[80:110]
    edited[100:120, 20:40] = 200
    board = BoardMap("b", (200, 300), [_face(0, 0, 70), _face(1, 70, 60)], [])
    fitted = fit_piece(edited, builder, board, max_mad=2.0)
    assert [(r["row"], r["dx"], r["dy"]) for r in fitted["rows"]] == [(0, 0, 0), (1, 0, 70)]
    assert fitted["unexplained_boxes"] == [[20, 100, 40, 120]]
    assert in_boxes(np.array([[25.0, 105.0], [100.0, 20.0]]), fitted["unexplained_boxes"]) == 1


def _texture(rng: np.random.Generator, height: int) -> np.ndarray:
    noise = cv2.GaussianBlur(rng.uniform(0, 255, (height, 300)), (0, 0), 4)
    return np.clip((noise - 127) * 10 + 140, 40, 255).astype(np.uint8)


def _face(row: int, y: int, height: int) -> FaceMap:
    distortion = Distortion(0.0, 0.0, (0.0, 0.0), 1.0)
    return FaceMap(
        str(row), row, (0.0, 0.0), (0, 0, 300, height), FacePlacement(0, y, 300, height, 1.0), 1.0, 1.0, distortion,
        [], np.zeros((height, 300), np.uint8),
    )
