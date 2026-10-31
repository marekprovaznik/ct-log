from concurrent.futures import ThreadPoolExecutor
import dataclasses
import datetime
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from camera_slider.stitch import capture_time

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
SHOTS_PER_PASS = 46
FULL_FRACTION = 0.95
PARTIAL_FRACTION = 0.9
COST_MISMATCH = 50.0
COST_INCOMPLETE = 3.0
COST_FULL_ON_ABORTED = 5.0
COST_SKIP_LOG = 2.0
COST_SKIP_ABORTED = 0.5
COST_SKIP_SEGMENT = 20.0
COST_MTIME = 1.0


@dataclasses.dataclass
class SessionLog:
    path: Path
    block: str
    board: str
    session: int
    passes: int
    shots: int
    status: str
    rows: pd.DataFrame
    mtime: datetime.datetime

    @property
    def is_calibration(self) -> bool:
        return not self.board.isdigit()


@dataclasses.dataclass
class Segment:
    index: int
    frames: List[Path]
    times: List[datetime.datetime]
    passes: List[List[int]]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> datetime.datetime:
        return self.times[0]

    @property
    def end(self) -> datetime.datetime:
        return self.times[-1]


@dataclasses.dataclass
class BoardMatch:
    segment: Optional[Segment]
    log: Optional[SessionLog]
    block: Optional[str]
    session: Optional[int]
    reason: str
    mtime_check: Optional[bool] = None


def index_frames(roots: Sequence[Path], exclude: Sequence[str], workers: int) -> List[Tuple[Path, datetime.datetime]]:
    paths = [
        p
        for root in roots
        for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and not any(part in exclude for part in p.parts)
    ]
    with ThreadPoolExecutor(workers) as pool:
        times = list(pool.map(capture_time, paths))
    missing = [p for p, t in zip(paths, times) if t is None]
    if missing:
        LOGGER.warning("%d frames without EXIF time are skipped, e.g. %s", len(missing), missing[0])
    frames = sorted(((p, t) for p, t in zip(paths, times) if t is not None), key=lambda pt: (pt[1], pt[0].name))
    LOGGER.info("Indexed %d frames under %s", len(frames), [str(r) for r in roots])
    return frames


def read_logs(roots: Sequence[Path]) -> List[SessionLog]:
    logs = [log for root in roots for path in sorted(root.rglob("*.csv")) if (log := _read_log(path)) is not None]
    LOGGER.info("Read %d ESP session logs", len(logs))
    return logs


def segment_frames(
    frames: List[Tuple[Path, datetime.datetime]], board_gap_s: float, pass_gap_s: float
) -> List[Segment]:
    segments: List[Segment] = []
    for path, t in frames:
        if not segments or (t - segments[-1].end).total_seconds() > board_gap_s:
            segments.append(Segment(len(segments), [], [], []))
        segment = segments[-1]
        if not segment.passes or (t - segment.end).total_seconds() > pass_gap_s:
            segment.passes.append([])
        segment.passes[-1].append(len(segment.frames))
        segment.frames.append(path)
        segment.times.append(t)
    LOGGER.info("Split into %d boards (gap > %.0f s)", len(segments), board_gap_s)
    return segments


def match_boards(
    segments: List[Segment],
    logs: List[SessionLog],
    clock_offset_s: float,
    csv_delay_s: Tuple[float, float],
) -> List[BoardMatch]:
    blocks: Dict[str, List[SessionLog]] = {}
    for log in logs:
        blocks.setdefault(log.block, []).append(log)
    best: Optional[Tuple[float, List[BoardMatch]]] = None
    for order in itertools.permutations(sorted(blocks)):
        slots = [
            (block, session, next((log for log in blocks[block] if log.session == session), None))
            for block in order
            for session in range(
                min(log.session for log in blocks[block]), max(log.session for log in blocks[block]) + 1
            )
        ]
        cost, matches = _align(segments, slots, clock_offset_s, csv_delay_s)
        LOGGER.info("Block order %s: alignment cost %.1f", list(order), cost)
        if best is None or cost < best[0]:
            best = (cost, matches)
    matches = best[1]
    for m in matches:
        if m.segment is not None and m.log is not None:
            m.mtime_check = _mtime_agrees(m.segment, m.log, clock_offset_s, csv_delay_s)
    disagree = [m for m in matches if m.mtime_check is False]
    LOGGER.info(
        "Matched %d boards to logs; %d disagree with the CSV file times (cross-check only)",
        sum(m.segment is not None and m.log is not None for m in matches),
        len(disagree),
    )
    return matches


def _align(
    segments: List[Segment],
    slots: List[Tuple[str, int, Optional[SessionLog]]],
    clock_offset_s: float,
    csv_delay_s: Tuple[float, float],
) -> Tuple[float, List[BoardMatch]]:
    n, m = len(segments), len(slots)
    cost = np.full((n + 1, m + 1), np.inf)
    step = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            options = []
            if i > 0 and j > 0:
                log = slots[j - 1][2]
                tie_break = 0.0
                if log is not None and not _mtime_agrees(segments[i - 1], log, clock_offset_s, csv_delay_s):
                    tie_break = COST_MTIME
                options.append((cost[i - 1, j - 1] + _pair_cost(segments[i - 1], log) + tie_break, 1))
            if j > 0:
                options.append((cost[i, j - 1] + (COST_SKIP_LOG if slots[j - 1][2] else COST_SKIP_ABORTED), 2))
            if i > 0:
                options.append((cost[i - 1, j] + COST_SKIP_SEGMENT, 3))
            cost[i, j], step[i, j] = min(options)
    matches: List[BoardMatch] = []
    i, j = n, m
    while i > 0 or j > 0:
        kind = step[i, j]
        if kind == 1:
            block, session, log = slots[j - 1]
            segment = segments[i - 1]
            reason = _pair_reason(segment, log)
            matches.append(BoardMatch(segment, log, block, session, reason))
            i, j = i - 1, j - 1
        elif kind == 2:
            block, session, log = slots[j - 1]
            if log is not None:
                matches.append(BoardMatch(None, log, block, session, "log without frames"))
            j -= 1
        else:
            matches.append(BoardMatch(segments[i - 1], None, None, None, "frames without log"))
            i -= 1
    return float(cost[n, m]), matches[::-1]


def _mtime_agrees(segment: Segment, log: SessionLog, clock_offset_s: float, csv_delay_s: Tuple[float, float]) -> bool:
    delay = (log.mtime - (segment.end - datetime.timedelta(seconds=clock_offset_s))).total_seconds()
    return bool(csv_delay_s[0] <= delay <= csv_delay_s[1])


def _pair_cost(segment: Segment, log: Optional[SessionLog]) -> float:
    n_passes = len(segment.passes)
    if log is None:
        partial = segment.n_frames < PARTIAL_FRACTION * SHOTS_PER_PASS * max(n_passes, 1) or n_passes < 2
        return 0.0 if partial else COST_FULL_ON_ABORTED
    if n_passes == log.passes and FULL_FRACTION * log.shots <= segment.n_frames <= log.shots + 2:
        return 0.0
    if n_passes <= log.passes and segment.n_frames < log.shots:
        return COST_INCOMPLETE
    return COST_MISMATCH


def _pair_reason(segment: Segment, log: Optional[SessionLog]) -> str:
    if log is None:
        return "aborted session without log"
    cost = _pair_cost(segment, log)
    if cost == 0.0:
        return "ok"
    if cost == COST_INCOMPLETE:
        return "incomplete frames"
    return "pass or shot count mismatch"


def _read_log(path: Path) -> Optional[SessionLog]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2 or not lines[0].startswith("#"):
        return None
    head = lines[0].lstrip("#").strip().split(";")
    meta = dict(zip(head[0::2], head[1::2]))
    columns = lines[1].split(";")
    body = []
    for line in lines[2:]:
        if line.startswith("#"):
            break
        values = line.split(";")
        if values and values[0].isdigit():
            body.append(values[: len(columns)])
    rows = pd.DataFrame(body, columns=columns)
    for column in ("pass", "shot", "ok"):
        rows[column] = rows[column].astype(int)
    for column in ("pos_mm", "t_ms"):
        rows[column] = rows[column].astype(float)
    return SessionLog(
        path=path,
        block=path.parent.name,
        board=meta.get("board", path.stem).strip(),
        session=int(meta["session"]),
        passes=int(meta.get("passes", rows["pass"].max())),
        shots=int(meta.get("shots", len(rows))),
        status=meta.get("status", ""),
        rows=rows,
        mtime=datetime.datetime.fromtimestamp(path.stat().st_mtime),
    )
