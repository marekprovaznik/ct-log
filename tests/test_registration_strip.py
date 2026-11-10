from camera_slider.registration import (
    FRAME_PARAMS,
    STRIP_PARAMS,
    FrameObs,
    _anchored_frames,
    _disagreeing_markers,
    _drop_false_ties,
    _jacobian,
    _Problem,
    _residuals,
)
from camera_slider.wood import PairMatches
import numpy as np

SIDE = 100.0


def _frame(name: str, ids: list, shift_x: float = 0.0, off: dict = None) -> FrameObs:
    """Markers on a straight strip, one step per ID, with optional per-ID displacement in px."""
    square = SIDE * np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    centres = [np.array([1.1 * SIDE * m + shift_x + (off or {}).get(m, 0.0), 500.0]) for m in ids]
    return FrameObs(name, np.array(ids), np.stack([c + square for c in centres]), (1000, 1500))


def test_disagreeing_marker_found_from_its_own_frame() -> None:
    """A marker off the frame's ID line is excluded; the frame stays anchored by the others."""
    frames = [_frame("a", [0, 1, 2, 3, 4, 5], off={3: 60.0}), _frame("b", [10, 11, 12, 13, 14])]
    disagreeing = _disagreeing_markers(frames, SIDE)
    assert disagreeing == {(0, 3)}
    assert _anchored_frames(frames, disagreeing) == {0, 1}


def test_frame_below_marker_threshold_is_not_anchored() -> None:
    """Two markers are too few to anchor a frame and are not judged against each other."""
    frames = [_frame("a", [0, 1])]
    assert _disagreeing_markers(frames, SIDE) == set()
    assert _anchored_frames(frames, set()) == set()


def test_false_tie_pair_is_dropped_against_marker_motion() -> None:
    """A pair that contradicts the shared markers goes; single bad points go from a good pair."""
    a, b = _frame("a", [0, 1, 2, 3, 4]), _frame("b", [0, 1, 2, 3, 4], shift_x=-220.0)
    points = np.column_stack([np.linspace(0, 800, 10), np.full(10, 100.0)])
    good = points + np.array([-220.0, 0.0])
    false_pair = PairMatches(0, 1, points, good + 2000.0)
    mixed_b = good.copy()
    mixed_b[0] += 2000.0
    kept = _drop_false_ties([false_pair, PairMatches(0, 1, points, mixed_b)], [a, b], set(), SIDE)
    assert len(kept) == 1
    assert len(kept[0].pts_a) == 9


def _problem(rng: np.random.Generator) -> _Problem:
    """Two frames, three markers on a strip, ties between the frames and a position prior on both."""
    return _Problem(
        frames=[0, 1],
        markers=[3, 4, 5],
        obs_frame=np.array([0, 0, 1, 1]),
        obs_marker=np.array([0, 1, 1, 2]),
        obs_xy=rng.uniform(0, 100, (4, 4, 2)),
        tie_a=np.zeros(6, int),
        tie_b=np.ones(6, int),
        tie_ua=rng.uniform(0, 100, (6, 2)),
        tie_ub=rng.uniform(0, 100, (6, 2)),
        principal=np.array([50.0, 50.0]),
        radius=70.0,
        side_px=40.0,
        estimate_k=True,
        estimate_dist=True,
        marker_ids=np.array([3.0, 6.0, 13.0]),
        strip_turn=np.pi / 2,
        prior_frame=np.array([0, 1]),
        prior_steps=np.array([3.4, 4.6]),
        prior_sigma=np.array([0.1, 0.3]),
        obs_scale=np.array([2.0, 2.0, 1.0, 1.0]).repeat(8),
    )


def test_jacobian_matches_finite_differences() -> None:
    """The analytic Jacobian, strip and prior rows included, matches central differences."""
    rng = np.random.default_rng(3)
    problem = _problem(rng)
    poses = np.array([[3.3, 0.1, 1.6], [4.4, 0.0, 1.5], [5.6, -0.1, 1.6]])
    n_frame_cols = FRAME_PARAMS * 2
    strip_col = n_frame_cols + 3 * 2
    x = np.concatenate(
        [
            (np.eye(3) * 0.02 + rng.normal(0, 1e-4, (2, 3, 3))).reshape(2, 9)[:, :FRAME_PARAMS].ravel(),
            poses[1:].ravel(),
            np.array([0.1, 0.05, 0.02, 1.1, -0.05]),
            np.array([1.01]),
            np.array([0.01, -0.002]),
        ]
    )

    def unpack(v: np.ndarray):
        h = np.concatenate([v[:n_frame_cols].reshape(-1, FRAME_PARAMS), np.ones((2, 1))], 1).reshape(-1, 3, 3)
        p = np.concatenate([poses[:1], v[n_frame_cols:strip_col].reshape(-1, 3)])
        return h, p, v[strip_col : strip_col + STRIP_PARAMS], v[strip_col + STRIP_PARAMS], v[-2:]

    analytic = _jacobian(problem, *unpack(x), n_frame_cols, strip_col, len(x))
    numeric = np.zeros_like(analytic)
    for j in range(len(x)):
        step = 1e-6 * max(1.0, abs(x[j]))
        hi, lo = x.copy(), x.copy()
        hi[j] += step
        lo[j] -= step
        numeric[:, j] = (_residuals(problem, *unpack(hi)) - _residuals(problem, *unpack(lo))) / (2 * step)
    assert np.abs(analytic - numeric).max() < 1e-4 * max(1.0, np.abs(numeric).max())
