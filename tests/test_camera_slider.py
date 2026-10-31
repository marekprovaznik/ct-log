import datetime
from pathlib import Path
from typing import Dict, List

from camera_slider.registration import (
    ARUCO_DICT,
    PassRegistration,
    detect_frame,
    find_gaps,
    make_detector,
    register_pass,
)
from camera_slider.stitch import group_passes
from camera_slider.wood import PairMatches, make_sift, match_features, segment_board, wood_features
import cv2
import numpy as np
import pytest

TEXTURE_PX_PER_MM = 8.0
MARKER_MM = 20.0
PITCH_MM = 22.4
FOCAL_PX = 4500.0
STRIP_DEPTH_MM = 900.0
WOOD_DEPTH_MM = 870.0
FRAME_SIZE = (1200, 800)
ROLL_DEG = 3.0
STEP_MM = 60.0
DROPPED_FRAME = 3


@pytest.fixture(scope="module")
def synthetic_pass() -> Dict:
    """Frames of a translating pinhole camera over a wood plane and a marker strip 30 mm below it.

    The strip lies beside the board, so markers and wood are imaged at different magnifications
    (k = 900 / 870); one frame is dropped to create a gap.
    """
    rng = np.random.default_rng(0)
    length_mm = 700.0
    wood = _wood_texture(rng, int(length_mm * TEXTURE_PX_PER_MM), int(50 * TEXTURE_PX_PER_MM))
    strip, strip_mask = _strip_texture(int(length_mm * TEXTURE_PX_PER_MM))
    centres = [130.0 + STEP_MM * i for i in range(8)]
    frames = []
    for i, cx in enumerate(centres):
        if i == DROPPED_FRAME:
            continue
        image = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
        wood_map = _plane_to_image(cx, WOOD_DEPTH_MM, -25.0)
        image = _paste(image, wood, np.full(wood.shape[:2], 255, np.uint8), wood_map)
        strip_map = _plane_to_image(cx, STRIP_DEPTH_MM, -55.0)
        image = _paste(image, strip, strip_mask, strip_map)
        frames.append(image)
    return {"frames": frames, "names": [f"IMG_{i:04d}.JPG" for i in range(len(frames))]}


@pytest.fixture(scope="module")
def registration(synthetic_pass: Dict) -> PassRegistration:
    """Joint marker + wood registration of the synthetic pass."""
    detector = make_detector()
    observations = [detect_frame(n, f, detector) for n, f in zip(synthetic_pass["names"], synthetic_pass["frames"])]
    features = [wood_features(f, make_sift()) for f in synthetic_pass["frames"]]
    pairs = []
    for a in range(len(features)):
        for b in range(a + 1, min(a + 3, len(features))):
            matched = match_features(features[a], features[b])
            if matched is not None:
                pairs.append(PairMatches(a, b, *matched))
    return register_pass(observations, pairs)


def test_all_frames_are_registered(registration: PassRegistration) -> None:
    """Every synthetic frame has markers and wood, so all of them are placed."""
    assert all(f.registered for f in registration.frames)


def test_height_scale_matches_plane_depths(registration: PassRegistration) -> None:
    """k recovers the magnification ratio between the strip plane and the wood plane."""
    assert registration.height_k == pytest.approx(STRIP_DEPTH_MM / WOOD_DEPTH_MM, rel=2e-3)


def test_wood_seams_are_subpixel(registration: PassRegistration) -> None:
    """Wood tie points from overlapping frames land on the same board position."""
    assert max(registration.seams.values()) < 1.0


def test_marker_pitch_is_recovered(registration: PassRegistration) -> None:
    """The strip pitch in marker units equals the printed pitch over the marker side, up to corner-localisation bias."""
    assert registration.pitch == pytest.approx(PITCH_MM / MARKER_MM, rel=5e-3)


def test_dropped_frame_is_reported(registration: PassRegistration, synthetic_pass: Dict) -> None:
    """The step over the dropped frame is about twice the median step."""
    transforms = {i: f.transform for i, f in enumerate(registration.frames)}
    shapes = [f.shape[:2] for f in synthetic_pass["frames"]]
    gaps = find_gaps(synthetic_pass["names"], transforms, shapes)
    assert [(g.after, g.n_missing) for g in gaps] == [(synthetic_pass["names"][DROPPED_FRAME - 1], 1)]


def test_group_passes_splits_on_time_gaps() -> None:
    """A pause longer than the gap threshold starts a new pass."""
    start = datetime.datetime(2026, 10, 23, 22, 15)
    offsets = [0.0, 0.5, 1.0, 7.0, 7.5, 14.0]
    paths = [Path(f"IMG_{i}.JPG") for i in range(len(offsets))]
    times = [start + datetime.timedelta(seconds=s) for s in offsets]
    passes = group_passes(paths, times, gap_s=3.0)
    assert [[p.name for p in group] for group in passes] == [
        ["IMG_0.JPG", "IMG_1.JPG", "IMG_2.JPG"],
        ["IMG_3.JPG", "IMG_4.JPG"],
        ["IMG_5.JPG"],
    ]


def test_group_passes_needs_capture_times() -> None:
    """Frames without EXIF time cannot be split into passes."""
    with pytest.raises(ValueError):
        group_passes([Path("a.jpg")], [None], gap_s=3.0)


def test_segment_board_fills_dark_defects() -> None:
    """A dark knot inside the board stays part of the board mask; the strip and background do not."""
    image = np.zeros((200, 400, 3), np.uint8)
    image[50:150, 20:380] = (60, 130, 200)
    cv2.circle(image, (200, 100), 15, (20, 20, 20), -1)
    image[10:40, 20:380] = 255
    mask = segment_board(image, scale=2)
    assert mask[100, 200]
    assert not mask[25, 200]
    assert not mask[180, 200]
    ys, xs = np.nonzero(mask)
    assert abs(ys.min() - 50) <= 2 and abs(ys.max() - 149) <= 2


def _plane_to_image(centre_mm: float, depth_mm: float, y0_mm: float) -> np.ndarray:
    scale = FOCAL_PX / depth_mm / TEXTURE_PX_PER_MM
    w, h = FRAME_SIZE
    to_image = np.array(
        [
            [scale, 0, w / 2 - FOCAL_PX * centre_mm / depth_mm],
            [0, scale, h / 2 + FOCAL_PX * y0_mm / depth_mm],
            [0, 0, 1],
        ]
    )
    roll = cv2.getRotationMatrix2D((w / 2, h / 2), ROLL_DEG, 1.0)
    return np.vstack([roll, [0, 0, 1]]) @ to_image


def _paste(image: np.ndarray, texture: np.ndarray, mask: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(texture, mapping, size, flags=cv2.INTER_AREA)
    alpha = cv2.warpPerspective(mask, mapping, size, flags=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    return (image * (1 - alpha) + warped * alpha).astype(np.uint8)


def _wood_texture(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    noise = rng.normal(size=(height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(noise, (0, 0), sigmaX=12, sigmaY=1.5)
    detail = cv2.GaussianBlur(rng.normal(size=(height, width)).astype(np.float32), (0, 0), 2)
    shade = 1 + 0.25 * grain / grain.std() + 0.15 * detail / detail.std()
    base = np.array([60, 125, 195], np.float32)
    return np.clip(base * shade[..., None], 0, 255).astype(np.uint8)


def _strip_texture(width: int) -> List[np.ndarray]:
    side = int(MARKER_MM * TEXTURE_PX_PER_MM)
    pitch = PITCH_MM * TEXTURE_PX_PER_MM
    height = side + 40
    strip = np.full((height, width, 3), 245, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_id = 0
    while 20 + marker_id * pitch + side < width:
        x = int(round(20 + marker_id * pitch))
        strip[20 : 20 + side, x : x + side] = cv2.cvtColor(
            cv2.aruco.generateImageMarker(dictionary, marker_id, side), cv2.COLOR_GRAY2BGR
        )
        marker_id += 1
    return [strip, np.full((height, width), 255, np.uint8)]


def test_segment_board_drops_side_sliver_and_edge_notch() -> None:
    """A thin side-face sliver below the board is dropped and a dark notch at the edge is filled."""
    image = np.zeros((200, 800, 3), np.uint8)
    image[40:140, 20:780] = (60, 130, 200)
    image[150:156, 40:760] = (60, 130, 200)
    image[120:140, 300:330] = (20, 20, 20)
    mask = segment_board(image, scale=2, edge_window=201)
    assert not mask[153, 400]
    assert mask[130, 315]
    assert mask[90, 400]
    assert not mask[30, 400]
