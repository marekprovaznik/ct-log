from typing import Dict, List, Tuple

from camera_slider.markers import (
    SOURCE_RECOVERED,
    STRIP_DICT,
    detect_markers,
    make_aruco,
)
import cv2
import numpy as np
import pytest

SIDE = 180
PITCH = 200
IDS = list(range(20, 28))


@pytest.fixture(scope="module")
def strip_frame() -> Dict:
    """Markers on a black strip with grey separators, rotated a quarter turn, IDs decreasing along +x."""
    dictionary = cv2.aruco.getPredefinedDictionary(STRIP_DICT)
    image = np.full((700, 1800), 150, np.uint8)
    image[0:260] = 0
    truth: Dict[int, np.ndarray] = {}
    for k, marker_id in enumerate(IDS):
        x0 = 1700 - (k + 1) * PITCH
        marker = cv2.rotate(cv2.aruco.generateImageMarker(dictionary, marker_id, SIDE), cv2.ROTATE_90_CLOCKWISE)
        y0 = 40
        if 0 <= x0 and x0 + SIDE <= image.shape[1]:
            image[y0 : y0 + SIDE, x0 : x0 + SIDE] = marker
            image[0:240, x0 + SIDE + 4 : x0 + PITCH - 4] = 150
            corners = np.array([[x0 + SIDE, y0], [x0 + SIDE, y0 + SIDE], [x0, y0 + SIDE], [x0, y0]], float)
            truth[marker_id] = corners - 0.5
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return {"image": blurred, "truth": truth}


def _corner_error(found_ids: List[int], found: np.ndarray, truth: Dict[int, np.ndarray]) -> Tuple[int, float]:
    errors = [np.linalg.norm(c - truth[i], axis=1).max() for i, c in zip(found_ids, found) if i in truth]
    return len(errors), float(max(errors))


def test_all_strip_markers_found_with_accurate_corners(strip_frame: Dict) -> None:
    """Every fully visible marker is found and its corners lie within 1.5 px of the drawn square."""
    markers = detect_markers(strip_frame["image"], make_aruco())
    assert set(strip_frame["truth"]) <= set(markers.ids.tolist())
    n, worst = _corner_error(markers.ids.tolist(), markers.corners, strip_frame["truth"])
    assert n == len(strip_frame["truth"])
    assert worst < 1.5


def test_clipped_marker_recovered_from_strip_position(strip_frame: Dict) -> None:
    """A marker cut by the frame edge is recovered at its strip position, not decoded by ArUco."""
    clipped = strip_frame["image"][70:].copy()
    markers = detect_markers(clipped, make_aruco())
    recovered = markers.ids[markers.source == SOURCE_RECOVERED].tolist()
    assert len(markers.ids) >= len(strip_frame["truth"]) - 1
    assert recovered
    shifted = {i: c - np.array([0.0, 70.0]) for i, c in strip_frame["truth"].items()}
    _, worst = _corner_error(markers.ids.tolist(), markers.corners, shifted)
    assert worst < 3.0
