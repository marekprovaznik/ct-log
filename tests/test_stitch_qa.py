import json
from pathlib import Path
from typing import List, Tuple

from camera_slider.stitch_qa import (
    SCALE,
    Thresholds,
    add_stray_origin,
    classify,
    evaluate,
    measure_piece,
    report_signals,
)
import cv2
import numpy as np
import pandas as pd
import pytest

PPMM = 8.0
LENGTH_PX = 8000
FACE_ROWS = ((40, 700), (780, 1100), (1180, 1840), (1920, 2240))
KNOTS_PER_EDGE = 12


@pytest.fixture(scope="module")
def unrolled_faces() -> List[np.ndarray]:
    """Four wood faces with knots that cross each edge of the cycle 0 -> 1 -> 2 -> 3 -> 0.

    Each physical edge gets its own knot positions, painted on the bottom strip of one row and the top
    strip of the next, so the edge-continuity signal can tell the true row cycle from a swapped one.
    """
    rng = np.random.default_rng(0)
    faces = []
    for y0, y1 in FACE_ROWS:
        face = np.full((y1 - y0, LENGTH_PX, 3), (90, 150, 200), np.uint8)
        face = (face + rng.normal(0, 6, face.shape)).clip(0, 255).astype(np.uint8)
        faces.append(face)
    for row in range(4):
        nxt = (row + 1) % 4
        for x in rng.uniform(600, LENGTH_PX - 600, KNOTS_PER_EDGE).astype(int):
            r = int(rng.uniform(25, 60))
            cv2.circle(faces[row], (x, faces[row].shape[0] - 1), r, (30, 50, 70), -1)
            cv2.circle(faces[nxt], (x, 0), r, (30, 50, 70), -1)
    return faces


def compose(faces: List[np.ndarray], rows: Tuple[int, ...]) -> np.ndarray:
    """Stack the faces in the given row order onto a black piece."""
    piece = np.zeros((FACE_ROWS[-1][1] + 40, LENGTH_PX, 3), np.uint8)
    for slot, face_index in enumerate(rows):
        y0, y1 = FACE_ROWS[slot]
        face = faces[face_index]
        piece[y0:y1] = cv2.resize(face, (LENGTH_PX, y1 - y0)) if face.shape[0] != y1 - y0 else face
    return piece


def measure(tmp_path: Path, name: str, piece: np.ndarray) -> dict:
    """Write a piece and measure it."""
    path = tmp_path / f"{name}.jpg"
    cv2.imwrite(str(path), piece)
    return measure_piece(path, None, 4, PPMM, [0, 1, 2, 3])


def test_bands_and_clean_piece(tmp_path: Path, unrolled_faces: List[np.ndarray]) -> None:
    """A correctly stacked piece has four bands, no strays and no edge steps."""
    row = measure(tmp_path, "spruce_001_x00000", compose(unrolled_faces, (0, 1, 2, 3)))
    assert row["n_bands"] == 4
    assert row["stray_max_mm2"] == 0.0
    assert row["edge_step_mm"] < 1.0


def test_stray_sliver_is_found(tmp_path: Path, unrolled_faces: List[np.ndarray]) -> None:
    """A thin wood sliver in the gap between two rows is reported as a stray segment."""
    piece = compose(unrolled_faces, (0, 1, 2, 3))
    piece[740:752, 2000:6000] = (90, 150, 200)
    row = measure(tmp_path, "spruce_001_x00000", piece)
    assert row["stray_max_mm2"] > 40.0
    assert row["n_bands"] == 4


def test_edge_step_is_measured(tmp_path: Path, unrolled_faces: List[np.ndarray]) -> None:
    """Wood added above a face over part of its length shows up as an edge step of that height."""
    piece = compose(unrolled_faces, (0, 1, 2, 3))
    piece[1140:1180, 4000:7000] = (90, 150, 200)
    row = measure(tmp_path, "spruce_001_x00000", piece)
    assert row["edge_step_mm"] == pytest.approx(40 / PPMM, abs=SCALE / PPMM)


def test_face_order_prefers_true_cycle(tmp_path: Path, unrolled_faces: List[np.ndarray]) -> None:
    """Swapping rows 1 and 3 makes the swapped cycle score above the expected one, and not otherwise."""
    good = measure(tmp_path, "spruce_001_x00000", compose(unrolled_faces, (0, 1, 2, 3)))
    swapped = measure(tmp_path, "spruce_002_x00000", compose(unrolled_faces, (0, 3, 2, 1)))
    pieces = classify(pd.DataFrame([good, swapped]), Thresholds(), "0123").set_index("piece")
    assert pieces.loc["spruce_001_x00000", "order_margin"] < 0
    assert pieces.loc["spruce_002_x00000", "order_margin"] > 0.08
    assert pieces.loc["spruce_002_x00000", "order_cycle"] == "0321"
    assert "face_order" in pieces.loc["spruce_002_x00000", "types"]
    assert "face_order" not in pieces.loc["spruce_001_x00000", "types"]


def test_face_shift_measures_offset(tmp_path: Path, unrolled_faces: List[np.ndarray]) -> None:
    """Shifting one face along the board by 15 mm is measured on its edges."""
    faces = list(unrolled_faces)
    faces[2] = np.roll(faces[2], int(15 * PPMM), axis=1)
    row = measure(tmp_path, "spruce_001_x00000", compose(faces, (0, 1, 2, 3)))
    assert abs(row["shift_med_mm"]) == pytest.approx(15, abs=1.0)


def test_evaluate_counts_per_type() -> None:
    """Precision and recall are computed per type and for any defect."""
    pieces = pd.DataFrame({"piece": ["a", "b", "c"], "types": ["edge_step", "", "edge_step;face_shift"]})
    labels = pd.DataFrame({"piece": ["a", "b", "c"], "types": ["edge_step", "face_shift", ""]})
    metrics = evaluate(pieces, labels)
    assert metrics["edge_step"]["tp"] == 1
    assert metrics["edge_step"]["fp"] == 1
    assert metrics["face_shift"]["fn"] == 1
    assert metrics["any"]["recall"] == 0.5


def test_stray_origin_from_reference() -> None:
    """A stray absent from the builder piece is a hand edit; one present there is a builder stray."""
    pieces = pd.DataFrame(
        {
            "piece": ["a", "b", "c", "d"],
            "types": ["stray_segment", "stray_segment", "stray_segment", ""],
            "evidence": ["x", "y", "z", ""],
        }
    )
    reference = pd.DataFrame({"piece": ["a", "b", "d"], "stray_max_mm2": [0.0, 500.0, 0.0]})
    out = add_stray_origin(pieces, reference, Thresholds()).set_index("piece")
    assert out.loc["a", "stray_origin"] == "hand_edit"
    assert out.loc["b", "stray_origin"] == "builder"
    assert out.loc["c", "stray_origin"] == "no_reference"
    assert out.loc["d", "stray_origin"] == ""
    assert out.loc["a", "evidence"] == "x (origin: hand_edit)"


def test_report_signals_without_mapping_use_board_seam(tmp_path: Path) -> None:
    """Without mapping.json every piece gets the board's worst seam, marked as board level."""
    board = tmp_path / "spruce_001"
    board.mkdir()
    report = {
        "face_stretch": {"A": 1.0, "B": 1.0},
        "faces": [
            {"seam_max_px": 1.2, "uncovered_columns": 0, "flagged_frames": []},
            {"seam_max_px": 4.1, "uncovered_columns": 0, "flagged_frames": ["IMG_1.JPG"]},
        ],
        "pieces": [{"name": "spruce_001_x00000.jpg"}, {"name": "spruce_001_x08000.jpg"}],
    }
    (board / "report.json").write_text(json.dumps(report))
    rows = report_signals(tmp_path, ["spruce_001_x00000", "spruce_001_x08000"]).set_index("piece")
    assert rows.seam_max_px.tolist() == [4.1, 4.1]
    assert rows.seam_scope.tolist() == ["board", "board"]
    assert rows.flagged_frames.tolist() == [1, 1]
