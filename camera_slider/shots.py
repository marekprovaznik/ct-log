import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from camera_slider.registration import FrameObs

LOGGER = logging.getLogger(__name__)
TIME_TOLERANCE_MS = 120.0
TIME_CAP_MS = 700.0
SKIP_COST_MS = 300.0
SHIFT_CANDIDATES = (-2, -1, 0, 1, 2)
RAIL_INLIER_MM = 30.0
LATE_MS = 150.0


@dataclasses.dataclass
class RailFit:
    pitch_mm: float
    offset_mm: Dict[str, float]
    n_frames: int
    resid_mm: float

    def position(self, camera_id: np.ndarray, direction: str) -> np.ndarray:
        return self.pitch_mm * camera_id + self.offset_mm.get(direction, np.mean(list(self.offset_mm.values())))


@dataclasses.dataclass
class PassShots:
    shot: np.ndarray
    delay_ms: np.ndarray
    position_mm: np.ndarray
    position_resid_mm: np.ndarray
    skipped_shots: List[int]
    offset_ms: float
    shift: int

    @property
    def n_late(self) -> int:
        return int((self.delay_ms > LATE_MS).sum())


def camera_id(obs: FrameObs) -> float:
    if len(obs.ids) < 2:
        return float("nan")
    xs = obs.corners.mean(1)[:, 0]
    slope, intercept = np.polyfit(xs, obs.ids.astype(float), 1)
    return float(slope * obs.shape[1] / 2 + intercept)


def fit_rail(samples: List[Tuple[np.ndarray, np.ndarray, str]]) -> Optional[RailFit]:
    cams = np.concatenate([s[0] for s in samples]) if samples else np.zeros(0)
    if len(cams) < 10:
        return None
    pos = np.concatenate([s[1] for s in samples])
    directions = sorted({s[2] for s in samples})
    dirs = np.concatenate([np.full(len(s[0]), s[2]) for s in samples])
    keep = np.isfinite(cams)
    for _ in range(3):
        design = np.column_stack([cams] + [(dirs == d).astype(float) for d in directions])
        coef, *_ = np.linalg.lstsq(design[keep], pos[keep], rcond=None)
        resid = pos - design @ coef
        keep = np.isfinite(cams) & (np.abs(resid) < RAIL_INLIER_MM)
    fit = RailFit(
        float(coef[0]), dict(zip(directions, map(float, coef[1:]))), int(keep.sum()), float(np.std(resid[keep]))
    )
    LOGGER.info(
        "Rail fit: %.3f mm per marker id, offsets %s, %d frames, residual std %.1f mm",
        fit.pitch_mm,
        {k: round(v, 1) for k, v in fit.offset_mm.items()},
        fit.n_frames,
        fit.resid_mm,
    )
    return fit


def assign_pass(frame_ms: np.ndarray, rows: pd.DataFrame, cams: np.ndarray, rail: Optional[RailFit]) -> PassShots:
    t_ms = rows["t_ms"].to_numpy()
    pos = rows["pos_mm"].to_numpy()
    direction = str(rows["dir"].iloc[0])
    interval = float(np.median(np.diff(t_ms))) if len(t_ms) > 1 else 0.0
    base = _time_offset(frame_ms, t_ms)
    predicted = rail.position(cams, direction) if rail is not None else np.full(len(cams), np.nan)
    best = None
    for shift in SHIFT_CANDIDATES:
        offset = base + shift * interval
        idx = _monotone_assign(frame_ms + offset, t_ms)
        resid = pos[idx] - predicted
        finite = np.isfinite(resid)
        if finite.sum() >= 3:
            score = float(np.median(np.abs(resid[finite])))
        else:
            score = float(np.median(np.abs(frame_ms + offset - t_ms[idx]))) + abs(shift) * 1e6
        if best is None or score < best[0]:
            best = (score, shift, offset, idx)
    _, shift, offset, idx = best
    delay = frame_ms + offset - t_ms[idx]
    from_time = np.interp(frame_ms + offset, t_ms, pos)
    position = np.where(np.isfinite(predicted), predicted, from_time)
    shots = rows["shot"].to_numpy()
    return PassShots(
        shot=shots[idx],
        delay_ms=delay,
        position_mm=position,
        position_resid_mm=pos[idx] - predicted,
        skipped_shots=sorted(set(shots.tolist()) - set(shots[idx].tolist())),
        offset_ms=float(offset),
        shift=int(shift),
    )


def _time_offset(frame_ms: np.ndarray, t_ms: np.ndarray) -> float:
    candidates = (t_ms[None, :] - frame_ms[:, None]).ravel()
    hits = [(np.abs(t_ms[None, :] - (frame_ms[:, None] + c)).min(1) < TIME_TOLERANCE_MS).sum() for c in candidates]
    coarse = candidates[int(np.argmax(hits))]
    nearest = np.abs(t_ms[None, :] - (frame_ms[:, None] + coarse)).argmin(1)
    close = np.abs(t_ms[nearest] - frame_ms - coarse) < TIME_TOLERANCE_MS
    return float(np.median((t_ms[nearest] - frame_ms)[close]))


def _monotone_assign(frame_ms: np.ndarray, t_ms: np.ndarray) -> np.ndarray:
    n, m = len(frame_ms), len(t_ms)
    cost = np.full((n + 1, m + 1), np.inf)
    take = np.zeros((n + 1, m + 1), bool)
    cost[0, :] = np.arange(m + 1) * SKIP_COST_MS
    for i in range(1, n + 1):
        match = cost[i - 1, :-1] + np.minimum(np.abs(frame_ms[i - 1] - t_ms), TIME_CAP_MS)
        for j in range(1, m + 1):
            skip = cost[i, j - 1] + SKIP_COST_MS
            if match[j - 1] <= skip:
                cost[i, j], take[i, j] = match[j - 1], True
            else:
                cost[i, j] = skip
    j = int(np.argmin(cost[n] + (m - np.arange(m + 1)) * SKIP_COST_MS))
    idx = np.zeros(n, dtype=int)
    i = n
    while i > 0:
        if take[i, j]:
            idx[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return idx
