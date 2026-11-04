import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ann_pipeline.timber.data import decode_bitmap
import cv2
import numpy as np
from src.utils.mask import mask_to_base64
import torch

from camera_slider.compose import NO_OWNER
from camera_slider.mapping import (
    MAPPING_FILE,
    BoardMap,
    crop_to_frame,
    face_to_unrolled,
    frame_key,
    frame_to_crop,
    load_mapping,
    owners,
    unrolled_to_face,
)

LOGGER = logging.getLogger(__name__)
LAYOUT_VERSION = 1
SUBSAMPLES = np.array([[0.0, 0.0], [-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]])
FIT_MARGIN_PX = 60
FIT_REFINE_PX = 8
FIT_SCALE = 4
WOOD_GRAY = 20
WOOD_ROW_FRACTION = 0.1
DROP_FIELDS = ("id",)
SHARE_STEP_PX = 8
CHUNK_PX = 250_000
SHIFT_SAMPLE_STEP = 97
MAX_FOREIGN_SHARE = 0.02


@dataclasses.dataclass
class RowShift:
    row: int
    band: Tuple[int, int]
    dx: int
    dy: int
    mad: float
    ok: bool


@dataclasses.dataclass
class FramePoints:
    paths: List[str]
    gid: np.ndarray
    raw: np.ndarray
    lost_src: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move Supervisely annotations between two stitches of a board through the source frames."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-layout", help="fit per-piece face-row shifts of a hand-edited project")
    fit.add_argument("--edited_dir", type=Path, required=True, help="dataset dir of the edited project (img/)")
    fit.add_argument("--builder_dir", type=Path, required=True, help="dataset dir with the builder pieces (img/)")
    fit.add_argument("--mapping_root", type=Path, required=True, help="boards dir with <board>/mapping.json")
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--boards", nargs="*", default=None)
    fit.add_argument("--max_mad", type=float, default=2.0, help="mean abs grey difference of an ok row fit")
    fit.add_argument("--workers", type=int, default=8)
    move = sub.add_parser("transfer", help="map annotations of stitch A to stitch B")
    move.add_argument("--src_dir", type=Path, required=True, help="dataset dir of stitch A (ann/)")
    move.add_argument("--src_mapping_root", type=Path, required=True)
    move.add_argument("--src_layout", type=Path, default=None, help="fit-layout output when A was edited by hand")
    move.add_argument("--dst_mapping_root", type=Path, required=True)
    move.add_argument("--dst_layout", type=Path, default=None, help="fit-layout output to reproduce on B")
    move.add_argument("--out_dir", type=Path, required=True, help="dataset dir to write B's ann/ into")
    move.add_argument("--frames_out", type=Path, default=None, help="also save frame coordinates per board")
    move.add_argument("--boards", nargs="*", default=None)
    move.add_argument("--max_jump_px", type=float, default=1.5, help="seam jump above which an object is flagged")
    move.add_argument("--workers", type=int, default=8)
    move.add_argument("--chunk_px", type=int, default=CHUNK_PX, help="mask pixels mapped at once; sets memory per worker")
    move.add_argument(
        "--stitch_tags",
        nargs="*",
        default=["wrong_stitch", "parovani_ok"],
        help="image tags that judge the stitch; moved to was_<tag> when B is a different stitch",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "fit-layout":
        fit_layout(args)
    else:
        transfer(args)


def fit_layout(args: argparse.Namespace) -> None:
    pieces = sorted(p.name for p in (args.edited_dir / "img").iterdir())
    jobs = []
    for name in pieces:
        board = board_of(name)
        if args.boards is not None and board not in args.boards:
            continue
        if not (args.mapping_root / board / MAPPING_FILE).exists():
            LOGGER.warning("%s: no mapping for board %s, skipped", name, board)
            continue
        jobs.append((name, args.edited_dir / "img" / name, args.builder_dir / "img" / name, args.mapping_root / board))
    with ProcessPoolExecutor(args.workers) as pool:
        fitted = list(pool.map(_fit_piece, jobs, [args.max_mad] * len(jobs)))
    layout = {"version": LAYOUT_VERSION, "max_mad": args.max_mad, "pieces": dict(fitted)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(layout, indent=1))
    moved = sum(any(r["dx"] or r["dy"] for r in p["rows"]) for p in layout["pieces"].values())
    bad = [n for n, p in layout["pieces"].items() if not all(r["ok"] for r in p["rows"]) or p["unexplained_px"]]
    LOGGER.info("%d pieces fitted, %d with moved rows, %d need a look: %s", len(fitted), moved, len(bad), bad)


def fit_piece(edited: np.ndarray, builder: np.ndarray, board: BoardMap, max_mad: float) -> Dict:
    if edited.shape != builder.shape:
        msg = f"Edited piece {edited.shape} and builder piece {builder.shape} differ in size"
        raise ValueError(msg)
    h, w = builder.shape
    rows = []
    covered = np.zeros(h, np.int32)
    explained = np.zeros_like(edited, bool)
    for row, face in enumerate(board.faces):
        p = face.placement
        wood = np.nonzero(((builder[p.y : p.y + p.height] > WOOD_GRAY).mean(1)) > WOOD_ROW_FRACTION)[0]
        if len(wood) == 0:
            continue
        y0, y1 = p.y + int(wood[0]), p.y + int(wood[-1]) + 1
        dx, dy, mad = _match_band(edited, builder[y0:y1, FIT_MARGIN_PX : w - FIT_MARGIN_PX], y0)
        rows.append(dataclasses.asdict(RowShift(row, (y0, y1), dx, dy, mad, mad <= max_mad)))
        covered[max(0, y0 + dy) : min(h, y1 + dy)] += 1
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        band = np.zeros_like(builder)
        band[p.y : p.y + p.height] = builder[p.y : p.y + p.height]
        explained |= cv2.warpAffine(band, shift, (w, h), flags=cv2.INTER_NEAREST) > WOOD_GRAY
    explained = cv2.dilate(explained.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    unexplained = (edited > WOOD_GRAY) & ~explained
    unexplained = cv2.morphologyEx(unexplained.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(unexplained, connectivity=8)
    boxes = [[int(x), int(y), int(x + bw), int(y + bh)] for x, y, bw, bh, _ in stats[1:n]]
    return {
        "rows": rows,
        "overlap_rows": int((covered > 1).sum()),
        "unexplained_px": int(unexplained.sum()),
        "unexplained_boxes": boxes,
    }


def transfer(args: argparse.Namespace) -> None:
    src_layout = read_layout(args.src_layout)
    dst_layout = read_layout(args.dst_layout)
    src_boxes = read_unexplained(args.src_layout)
    anns: Dict[str, Dict[str, Dict]] = {}
    for path in sorted((args.src_dir / "ann").glob("*.json")):
        piece = path.name[: -len(".json")]
        if args.boards is None or board_of(piece) in args.boards:
            anns.setdefault(board_of(piece), {})[piece] = json.loads(path.read_text())
    (args.out_dir / "ann").mkdir(parents=True, exist_ok=True)
    meta = args.src_dir.parent / "meta.json"
    if meta.exists():
        (args.out_dir.parent / "meta.json").write_text(json.dumps(with_history_tags(meta, args.stitch_tags), indent=1))
    if args.frames_out is not None:
        args.frames_out.mkdir(parents=True, exist_ok=True)
    jobs = [(board, pieces, src_layout, dst_layout, src_boxes, args) for board, pieces in sorted(anns.items())]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_transfer_board, jobs))
    records = [r for board_records, _, _ in results for r in board_records]
    piece_rows = [r for _, board_pieces, _ in results for r in board_pieces]
    lost = [o for _, _, board_lost in results for o in board_lost]
    write_report(args.out_dir / "transfer_report.csv", records)
    write_report(args.out_dir / "piece_report.csv", piece_rows)
    (args.out_dir / "lost_objects.json").write_text(json.dumps(lost, indent=1))
    flagged = [r for r in records if r["flagged"]]
    LOGGER.info(
        "%d boards, %d objects moved, %d flagged (seam jump > %.1f px, lost pixels or unreliable layout), "
        "%d lost entirely, %d of %d pieces with mixed sources flagged",
        len(results),
        len(records),
        len(flagged),
        args.max_jump_px,
        len(lost),
        sum(r["flagged"] for r in piece_rows),
        len(piece_rows),
    )


def transfer_board(
    board: str,
    pieces: Dict[str, Dict],
    src_layout: Dict[str, List[RowShift]],
    dst_layout: Dict[str, List[RowShift]],
    src_boxes: Dict[str, List[List[int]]],
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    src = load_mapping(args.src_mapping_root / board)
    dst = load_mapping(args.dst_mapping_root / board)
    out = {p["name"]: [] for p in dst.pieces}
    sources = piece_sources(src, dst)
    same = same_stitch(src, dst)
    records = []
    lost = []
    saved: Dict[str, List] = {"obj": [], "gid": [], "raw": []}
    objects_meta = []
    for piece_name, ann in sorted(pieces.items()):
        shifts = src_layout.get(piece_name, [])
        x0 = src.piece(piece_name)["x0"]
        for obj in ann["objects"]:
            mask, origin = object_mask(obj, (ann["size"]["height"], ann["size"]["width"]))
            moved = move_object(
                mask, origin, src, shifts, x0, dst, dst_layout, src_boxes.get(piece_name, []), args.chunk_px,
                args.frames_out is not None,
            )
            record = {
                "board": board,
                "src_piece": piece_name,
                "src_object_id": obj.get("id"),
                "class": obj["classTitle"],
                "geometry": obj["geometryType"],
                "src_px": moved.src_px,
                "lost_src_px": moved.lost_src_px,
                "layout_ok": all(s.ok for s in shifts),
                "unexplained_px": moved.unexplained_px,
                **moved.seams,
            }
            smart = map_smart_input(obj, src, shifts, x0, dst, dst_layout, moved.shifts)
            tags = carry_tags(obj.get("tags", []), args.stitch_tags, same)
            placed = place_objects(obj, moved.patches, out, smart, tags)
            placed["lost_dst_px"] = moved.lost_dst_px
            record.update(placed)
            if not placed["dst_pieces"]:
                lost.append({"board": board, "src_piece": piece_name, "reason": "no pixel maps into B", "object": obj})
            record["src_components"] = n_components(mask)
            record["flagged"] = bool(
                record["lost_src_px"]
                or record["lost_dst_px"]
                or not record["layout_ok"]
                or record["unexplained_px"]
                or record["seam_gap_px"] > args.max_jump_px
                or record["seam_overlap_px"] > args.max_jump_px
                or record["dst_components"] > record["src_components"] + max(len(record["dst_pieces"].split()) - 1, 0)
            )
            records.append(record)
            if args.frames_out is not None:
                saved["obj"].append(np.full(len(moved.centre_gid), len(objects_meta), np.int32))
                saved["gid"].append(moved.centre_gid.astype(np.int16))
                saved["raw"].append(moved.centre_raw.astype(np.float32))
                objects_meta.append({"src_piece": piece_name, "object": obj})
    piece_rows = []
    for p in dst.pieces:
        main, row = piece_provenance(board, p["name"], sources[p["name"]], pieces, records)
        row["same_stitch"] = same
        piece_rows.append(row)
        ann = {
            "description": pieces[main].get("description", "") if main in pieces else "",
            "tags": carry_tags(pieces[main]["tags"], args.stitch_tags, same) if main in pieces else [],
            "size": {"height": dst.unrolled_shape[0], "width": p["x1"] - p["x0"]},
            "objects": out[p["name"]],
        }
        (args.out_dir / "ann" / f"{p['name']}.json").write_text(json.dumps(ann))
    if args.frames_out is not None:
        np.savez_compressed(
            args.frames_out / f"{board}.npz",
            paths=np.array([f.path for face in src.faces for f in face.frames]),
            **{k: np.concatenate(v) if v else np.zeros(0) for k, v in saved.items()},
        )
        (args.frames_out / f"{board}.objects.json").write_text(json.dumps(objects_meta))
    LOGGER.info("%s: %d objects", board, len(records))
    return records, piece_rows, lost


def same_stitch(src: BoardMap, dst: BoardMap) -> bool:
    if src.unrolled_shape != dst.unrolled_shape or len(src.faces) != len(dst.faces):
        return False
    for a, b in zip(src.faces, dst.faces):
        if (a.area, a.crop_box, a.origin, a.placement, a.px_per_unit) != (
            b.area,
            b.crop_box,
            b.origin,
            b.placement,
            b.px_per_unit,
        ):
            return False
        if [frame_key(f.path) for f in a.frames] != [frame_key(f.path) for f in b.frames]:
            return False
        if any(not np.allclose(fa.homography, fb.homography, rtol=0, atol=1e-12) for fa, fb in zip(a.frames, b.frames)):
            return False
    return True


def carry_tags(tags: List[Dict], stitch_tags: List[str], same: bool) -> List[Dict]:
    if same:
        return tags
    out = []
    for tag in tags:
        if tag["name"] in stitch_tags:
            tag = {k: v for k, v in tag.items() if k != "tagId"}
            tag["name"] = f"was_{tag['name']}"
        out.append(tag)
    return out


def with_history_tags(meta_path: Path, stitch_tags: List[str]) -> Dict:
    meta = json.loads(meta_path.read_text())
    names = {t["name"] for t in meta.get("tags", [])}
    for tag in list(meta.get("tags", [])):
        if tag["name"] in stitch_tags and f"was_{tag['name']}" not in names:
            history = {k: v for k, v in tag.items() if k not in ("id", "hotkey")}
            history["name"] = f"was_{tag['name']}"
            meta["tags"].append(history)
    return meta


def piece_sources(src: BoardMap, dst: BoardMap) -> Dict[str, Dict[str, float]]:
    shares = {}
    for piece in dst.pieces:
        xs = np.arange(0, piece["x1"] - piece["x0"], SHARE_STEP_PX)
        grids = []
        for face in dst.faces:
            ys = np.arange(face.placement.y, face.placement.y + face.placement.height, SHARE_STEP_PX)
            grids.append(np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2))
        names, _, _ = from_frames(src, {}, to_frames(dst, [], piece["x0"], np.concatenate(grids).astype(np.float64)))
        names = names[names != ""]
        shares[piece["name"]] = {n: float(np.mean(names == n)) for n in sorted(set(names))}
    return shares


def piece_provenance(
    board: str, name: str, shares: Dict[str, float], pieces: Dict[str, Dict], records: List[Dict]
) -> Tuple[Optional[str], Dict]:
    main = max(shares, key=shares.get) if shares else None
    tag_names = {n: sorted(t["name"] for t in pieces[n]["tags"]) if n in pieces else None for n in shares}
    foreign = [n for n in shares if n != main and tag_names[n] != tag_names.get(main)]
    foreign_records = [r for r in records if r["src_piece"] in foreign and name in r["dst_pieces"].split()]
    foreign_objects = len(foreign_records)
    row = {
        "board": board,
        "dst_piece": name,
        "main_src_piece": main or "",
        "main_share": round(shares.get(main, 0.0), 4),
        "tags": " ".join(tag_names.get(main) or []),
        "sources": " ".join(f"{n}:{shares[n]:.4f}:{'+'.join(tag_names[n] or ['-'])}" for n in shares),
        "objects_from_differently_tagged": foreign_objects,
        "their_classes": " ".join(sorted({r["class"] for r in foreign_records})),
        "flagged": bool(foreign_objects or sum(shares[n] for n in foreign) > MAX_FOREIGN_SHARE),
    }
    return main, row


def map_smart_input(
    obj: Dict,
    src: BoardMap,
    shifts: List[RowShift],
    x0: int,
    dst: BoardMap,
    dst_layout: Dict[str, List[RowShift]],
    part_shifts: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    smart = obj.get("smartToolInput")
    keys = [k for k in ("crop", "positive", "negative") if smart and smart.get(k)]
    if not keys:
        return {}
    pts = np.array([q for k in keys for q in smart[k]], np.float64).reshape(-1, 2)
    point_names, mapped, _ = from_frames(dst, dst_layout, to_frames(src, shifts, x0, pts))
    out = {}
    for name, shift in part_shifts.items():
        moved = np.where((point_names == name)[:, None], mapped, pts + shift)
        new, i = dict(smart), 0
        for k in keys:
            new[k] = moved[i : i + len(smart[k])].round(3).tolist()
            i += len(smart[k])
        out[name] = new
    return out


def to_frames(board: BoardMap, shifts: List[RowShift], x0: int, pts: np.ndarray) -> FramePoints:
    rows, builder = edited_to_builder(shifts, pts)
    unrolled = builder + np.array([x0, 0.0])
    rows, crop = unrolled_to_face(board, unrolled, rows)
    paths = [f.path for face in board.faces for f in face.frames]
    offsets = np.cumsum([0] + [len(face.frames) for face in board.faces])
    gid = np.full(len(pts), -1, np.int64)
    raw = np.full((len(pts), 2), np.nan)
    for k, face in enumerate(board.faces):
        sel = np.nonzero(rows == k)[0]
        own = owners(face, crop[sel])
        for index, part in _groups(own):
            if index == NO_OWNER:
                continue
            hit = sel[part]
            raw[hit] = crop_to_frame(face, crop[hit], face.frames[index])
            gid[hit] = offsets[k] + index
    return FramePoints(paths, gid, raw, gid < 0)


def from_frames(
    board: BoardMap, layout: Dict[str, List[RowShift]], frames: FramePoints
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    which, pts, unrolled = from_frames_index(board, layout, frames)
    names = np.array([""] + [p["name"] for p in board.pieces], dtype=object)[which + 1]
    return names, pts, unrolled


def from_frames_index(
    board: BoardMap, layout: Dict[str, List[RowShift]], frames: FramePoints
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = board.frame_index()
    unrolled = np.full((len(frames.gid), 2), np.nan)
    rows = np.full(len(frames.gid), -1)
    for gid, hit in _groups(frames.gid):
        path = frame_key(frames.paths[gid]) if gid >= 0 else None
        if path not in lookup:
            continue
        k, index = lookup[path]
        face = board.faces[k]
        unrolled[hit] = face_to_unrolled(face, frame_to_crop(face, frames.raw[hit], face.frames[index]))
        rows[hit] = k
    starts = np.array([p["x0"] for p in board.pieces])
    ends = np.array([p["x1"] for p in board.pieces])
    valid = np.isfinite(unrolled[:, 0])
    rx = np.where(valid, np.round(np.nan_to_num(unrolled[:, 0])), -1)
    which = np.searchsorted(starts, rx, side="right") - 1
    valid &= (which >= 0) & (rx < ends[np.clip(which, 0, None)])
    which[~valid] = -1
    pts = np.full_like(unrolled, np.nan)
    for w, hit in _groups(which):
        if w < 0:
            continue
        name = board.pieces[w]["name"]
        pts[hit] = builder_to_edited(layout.get(name, []), rows[hit], unrolled[hit] - np.array([starts[w], 0.0]))
    return which, pts, unrolled


def _groups(labels: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    order = np.argsort(labels, kind="stable")
    values, starts = np.unique(labels[order], return_index=True)
    return list(zip(values.tolist(), np.split(order, starts[1:])))


def edited_to_builder(shifts: List[RowShift], pts: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if not shifts:
        return None, pts
    starts = np.array([s.band[0] + s.dy for s in shifts], np.float64)
    ends = np.array([s.band[1] + s.dy for s in shifts], np.float64)
    y = pts[:, 1:2]
    distance = np.maximum(starts[None] - 0.5 - y, 0) + np.maximum(y - (ends[None] - 0.5), 0)
    nearest = distance.argmin(1)
    move = np.array([[s.dx, s.dy] for s in shifts], np.float64)
    return np.array([s.row for s in shifts])[nearest], pts - move[nearest]


def builder_to_edited(shifts: List[RowShift], rows: np.ndarray, pts: np.ndarray) -> np.ndarray:
    move = np.zeros((len(pts), 2))
    for s in shifts:
        move[rows == s.row] = (s.dx, s.dy)
    return pts + move


@dataclasses.dataclass
class MovedObject:
    src_px: int
    lost_src_px: int
    lost_dst_px: int
    unexplained_px: int
    seams: Dict
    patches: Dict[str, Tuple[np.ndarray, np.ndarray]]
    shifts: Dict[str, np.ndarray]
    centre_gid: np.ndarray
    centre_raw: np.ndarray


def move_object(
    mask: np.ndarray,
    origin: Tuple[int, int],
    src: BoardMap,
    shifts: List[RowShift],
    x0: int,
    dst: BoardMap,
    dst_layout: Dict[str, List[RowShift]],
    boxes: List[List[int]],
    chunk_px: int = CHUNK_PX,
    keep_frames: bool = False,
) -> MovedObject:
    h, w = mask.shape
    n = len(SUBSAMPLES)
    rows = max(1, chunk_px // max(w, 1))
    canvases: Dict[str, List] = {}
    moves: Dict[str, List[np.ndarray]] = {}
    seams = SeamStats()
    counts = {"src": 0, "lost_src": 0, "lost_dst": 0, "unexplained": 0}
    centre_gid, centre_raw = [], []
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        ys, xs = np.nonzero(mask[r0 : min(h, r1 + 1)])
        if len(xs) == 0:
            continue
        pix = np.stack([xs + origin[0], ys + r0 + origin[1]], 1).astype(np.float64)
        core = ys + r0 < r1
        frames = to_frames(src, shifts, x0, (pix[:, None, :] + SUBSAMPLES[None]).reshape(-1, 2))
        which, pts, unrolled = from_frames_index(dst, dst_layout, frames)
        gid = frames.gid.reshape(-1, n)[:, 0]
        seams.add(ys, xs, r1 - r0, gid, unrolled[::n], mask.shape[1])
        keep = np.repeat(core, n)
        which, pts, samples = which[keep], pts[keep], (pix[:, None, :] + SUBSAMPLES[None]).reshape(-1, 2)[keep]
        counts["src"] += int(core.sum())
        counts["lost_src"] += int(frames.lost_src.reshape(-1, n)[core, 0].sum())
        counts["lost_dst"] += int((which.reshape(-1, n)[:, 0] < 0).sum())
        counts["unexplained"] += in_boxes(pix[core], boxes)
        if keep_frames:
            centre_gid.append(gid[core])
            centre_raw.append(frames.raw.reshape(-1, n, 2)[core, 0].astype(np.float32))
        for index, hit in _groups(which):
            if index < 0:
                continue
            piece = dst.pieces[index]
            name = piece["name"]
            if name not in canvases:
                canvases[name] = [np.zeros((dst.unrolled_shape[0], piece["x1"] - piece["x0"]), bool), None, None]
            canvas = canvases[name]
            q = np.round(pts[hit]).astype(np.int64)
            inside = (q[:, 0] >= 0) & (q[:, 0] < canvas[0].shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < canvas[0].shape[0])
            q = q[inside]
            if len(q):
                canvas[0][q[:, 1], q[:, 0]] = True
                lo, hi = q.min(0), q.max(0)
                canvas[1] = lo if canvas[1] is None else np.minimum(canvas[1], lo)
                canvas[2] = hi if canvas[2] is None else np.maximum(canvas[2], hi)
            moves.setdefault(name, []).append((pts[hit] - samples[hit])[::SHIFT_SAMPLE_STEP])
    return MovedObject(
        counts["src"],
        counts["lost_src"],
        counts["lost_dst"],
        counts["unexplained"],
        seams.result(),
        {
            name: (img[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1].copy(), lo)
            for name, (img, lo, hi) in canvases.items()
            if lo is not None
        },
        {name: np.median(np.concatenate(m), 0) for name, m in moves.items()},
        np.concatenate(centre_gid) if centre_gid else np.zeros(0, np.int64),
        np.concatenate(centre_raw) if centre_raw else np.zeros((0, 2)),
    )


class SeamStats:
    def __init__(self) -> None:
        self.pairs = 0
        self.gap = 0.0
        self.overlap = 0.0
        self.jump = 0.0

    def add(self, ys: np.ndarray, xs: np.ndarray, core_rows: int, gid: np.ndarray, dst: np.ndarray, w: int) -> None:
        h = max(int(ys.max()) + 1, core_rows)
        gid_img = np.full((h + 1, w), -2, np.int64)
        pos = np.full((h + 1, w, 2), np.nan)
        gid_img[ys, xs] = gid
        pos[ys, xs] = dst
        pairs = (
            (gid_img[:core_rows, :-1], gid_img[:core_rows, 1:], pos[:core_rows, :-1], pos[:core_rows, 1:]),
            (gid_img[:core_rows], gid_img[1 : core_rows + 1], pos[:core_rows], pos[1 : core_rows + 1]),
        )
        for axis, (ga, gb, pa, pb) in enumerate(pairs):
            if ga.size == 0:
                continue
            valid = (ga > -2) & (gb > -2) & np.isfinite(pa[..., 0]) & np.isfinite(pb[..., 0])
            delta = (pb - pa)[valid]
            same = (ga == gb)[valid]
            expected = np.median(delta[same], 0) if same.any() else np.eye(2)[axis]
            cross = ~same
            self.pairs += int(cross.sum())
            if cross.any():
                off = delta[cross] - expected
                self.gap = max(self.gap, float(off[:, axis].max()))
                self.overlap = max(self.overlap, float(-off[:, axis].min()))
                self.jump = max(self.jump, float(np.linalg.norm(off, axis=1).max()))

    def result(self) -> Dict:
        return {
            "seam_pairs": self.pairs,
            "seam_jump_px": round(self.jump, 3),
            "seam_gap_px": round(self.gap, 3),
            "seam_overlap_px": round(self.overlap, 3),
        }


def place_objects(
    obj: Dict,
    patches: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out: Dict[str, List],
    smart: Dict[str, Dict],
    tags: List[Dict],
) -> Dict:
    pieces = []
    dst_px = 0
    components = 0
    for name in sorted(patches):
        patch, lo = patches[name]
        new = _clean(obj)
        new["tags"] = tags
        if name in smart:
            new["smartToolInput"] = smart[name]
        if obj["geometryType"] == "rectangle":
            new["points"] = {"exterior": [lo.tolist(), (lo + patch.shape[::-1] - 1).tolist()], "interior": []}
        else:
            bitmap = bitmap_with_origin(patch)
            bitmap["origin"] = [int(lo[0]) + bitmap["origin"][0], int(lo[1]) + bitmap["origin"][1]]
            new["bitmap"] = bitmap
        out[name].append(new)
        pieces.append(name)
        dst_px += int(patch.sum())
        components += n_components(patch)
    return {"dst_pieces": " ".join(pieces), "dst_px": dst_px, "dst_components": components}


def object_mask(obj: Dict, size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    if obj["geometryType"] == "bitmap":
        return decode_bitmap(obj["bitmap"]["data"]), tuple(obj["bitmap"]["origin"])
    if obj["geometryType"] == "rectangle":
        (x0, y0), (x1, y1) = obj["points"]["exterior"]
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        return np.ones((y1 - y0 + 1, x1 - x0 + 1), bool), (x0, y0)
    msg = f"Unsupported geometry {obj['geometryType']}"
    raise ValueError(msg)


def bitmap_with_origin(mask: np.ndarray) -> Dict:
    ys, xs = np.nonzero(mask)
    y0, x0 = int(ys.min()), int(xs.min())
    patch = mask[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1]
    return {"origin": [x0, y0], "data": mask_to_base64(torch.from_numpy(patch))}


def n_components(mask: np.ndarray) -> int:
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)


def read_layout(path: Optional[Path]) -> Dict[str, List[RowShift]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if data["version"] != LAYOUT_VERSION:
        msg = f"{path}: layout version {data['version']}, expected {LAYOUT_VERSION}"
        raise ValueError(msg)
    return {
        name: [RowShift(r["row"], tuple(r["band"]), r["dx"], r["dy"], r["mad"], r["ok"]) for r in piece["rows"]]
        for name, piece in data["pieces"].items()
    }


def read_unexplained(path: Optional[Path]) -> Dict[str, List[List[int]]]:
    if path is None:
        return {}
    pieces = json.loads(path.read_text())["pieces"]
    return {name: piece.get("unexplained_boxes", []) for name, piece in pieces.items()}


def in_boxes(pix: np.ndarray, boxes: List[List[int]]) -> int:
    inside = np.zeros(len(pix), bool)
    for x0, y0, x1, y1 in boxes:
        inside |= (pix[:, 0] >= x0) & (pix[:, 0] < x1) & (pix[:, 1] >= y0) & (pix[:, 1] < y1)
    return int(inside.sum())


def write_report(path: Path, records: List[Dict]) -> None:
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    LOGGER.info("Report written to %s", path)


def board_of(piece: str) -> str:
    return piece.rsplit("_x", 1)[0]


def _match_band(edited: np.ndarray, template: np.ndarray, y0: int) -> Tuple[int, int, float]:
    small = cv2.resize(edited, None, fx=1 / FIT_SCALE, fy=1 / FIT_SCALE, interpolation=cv2.INTER_AREA)
    small_t = cv2.resize(template, None, fx=1 / FIT_SCALE, fy=1 / FIT_SCALE, interpolation=cv2.INTER_AREA)
    _, _, _, loc = cv2.minMaxLoc(
        -cv2.matchTemplate(small.astype(np.float32), small_t.astype(np.float32), cv2.TM_SQDIFF)
    )
    h, w = template.shape
    wx0 = max(0, loc[0] * FIT_SCALE - FIT_REFINE_PX)
    wy0 = max(0, loc[1] * FIT_SCALE - FIT_REFINE_PX)
    window = edited[wy0 : wy0 + h + 2 * FIT_REFINE_PX, wx0 : wx0 + w + 2 * FIT_REFINE_PX]
    _, _, _, best = cv2.minMaxLoc(
        -cv2.matchTemplate(window.astype(np.float32), template.astype(np.float32), cv2.TM_SQDIFF)
    )
    x, y = wx0 + best[0], wy0 + best[1]
    mad = float(np.abs(edited[y : y + h, x : x + w].astype(np.float32) - template).mean())
    return int(x - FIT_MARGIN_PX), int(y - y0), round(mad, 3)


def _fit_piece(job: Tuple[str, Path, Path, Path], max_mad: float) -> Tuple[str, Dict]:
    name, edited_path, builder_path, board_dir = job
    edited = cv2.imread(str(edited_path), cv2.IMREAD_GRAYSCALE)
    builder = cv2.imread(str(builder_path), cv2.IMREAD_GRAYSCALE)
    return name, fit_piece(edited, builder, load_mapping(board_dir), max_mad)


def _transfer_board(job: Tuple) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    return transfer_board(*job)


def _clean(obj: Dict) -> Dict:
    return {k: v for k, v in obj.items() if k not in DROP_FIELDS and k not in ("bitmap", "points", "smartToolInput")}


if __name__ == "__main__":
    main()
