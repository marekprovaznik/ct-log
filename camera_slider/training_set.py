import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import datetime
import json
import logging
import math
from pathlib import Path
import random
import re
import subprocess
from typing import Dict, List, Optional, Tuple
import zlib

import cv2
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
PIECE_RE = re.compile(r"^(?P<board>(?P<species>[a-z]+)_\d+)_x(?P<x>\d+)\.(jpg|png|tif)$")
WOOD_N = {"spruce": 4, "larch": 2}
IGNORE_INDEX = 255
IGNORE = "ignore"
BACKGROUND = "background"
WOOD = "wood"
WOOD_COLOUR = [205, 170, 125]


@dataclasses.dataclass
class Piece:
    name: str
    board: str
    species: str
    x: int
    img_path: Path
    ann_path: Optional[Path]
    width: int = 0
    height: int = 0


@dataclasses.dataclass
class StripResult:
    strip: Dict
    tiles: List[Dict]
    objects: List[Dict]
    wood_components: List[Dict]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the precomputed timber training set (strips, tiles, masks).")
    parser.add_argument(
        "--project_dir", type=Path, required=True, help="Supervisely project dir (meta.json, <ds>/img, <ds>/ann)"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="dataset_v1 manifest.csv with px_per_mm per board")
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--target_px_per_mm", type=float, default=4.0)
    parser.add_argument("--tile", type=int, default=800)
    parser.add_argument("--overlap_px", type=int, default=None, help="tile overlap at target res")
    parser.add_argument("--knot_p99_mm", type=float, default=None, help="sets overlap_px = ceil(p99 * target px/mm)")
    parser.add_argument("--annotated", choices=["all", "objects", "tag", "list"], default="objects")
    parser.add_argument("--annotated_tag", default="bumaska")
    parser.add_argument("--piece_list", type=Path, default=None, help="text file of piece names for --annotated list")
    parser.add_argument("--exclude_tag", nargs="*", default=[], help="pieces with any of these tags are dropped")
    parser.add_argument("--classes", nargs="*", default=None, help="defect classes in priority order, low to high")
    parser.add_argument("--ignored_classes", nargs="*", default=[])
    parser.add_argument(
        "--rename", nargs="*", default=[], help="SOURCE=TARGET class renames, e.g. Bark_inclusion=bark_inclusion"
    )
    parser.add_argument("--require_objects", action="store_true", help="drop pieces without objects")
    parser.add_argument("--black_threshold", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=13)
    parser.add_argument("--split_ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument(
        "--split_file", type=Path, default=None, help="JSON {train|val|test: [boards]} to reuse a split"
    )
    parser.add_argument(
        "--snapshot_info", type=Path, default=None, help="JSON merged into DATASET.json source_snapshot"
    )
    parser.add_argument("--annotated_piece_rule", default=None, help="text for DATASET.json")
    parser.add_argument("--boards", nargs="*", default=None, help="restrict to these boards (testing)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args)


def build(args: argparse.Namespace) -> None:
    meta = _read_meta(args.project_dir)
    rename = dict(item.split("=", 1) for item in args.rename)
    classes = build_classes(meta, args.classes, args.ignored_classes, rename)
    priority = [c["name"] for c in classes]
    px_per_mm = _read_px_per_mm(args.manifest)
    pieces = collect_pieces(args.project_dir, args)
    strips = group_strips(pieces)
    if args.boards:
        strips = [s for s in strips if s[0].board in set(args.boards)]
    if not strips:
        msg = "No annotated strips found"
        raise ValueError(msg)
    overlap = (
        args.overlap_px
        if args.overlap_px is not None
        else math.ceil((args.knot_p99_mm or 50.0) * args.target_px_per_mm)
    )
    if overlap >= args.tile:
        msg = f"overlap {overlap} must be smaller than tile {args.tile}"
        raise ValueError(msg)
    boards = sorted({s[0].board for s in strips})
    split = load_or_make_split(boards, args)
    out = args.out_root
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    jobs = []
    counters: Dict[str, int] = {}
    for strip_pieces in strips:
        board = strip_pieces[0].board
        k = counters.get(board, 0)
        counters[board] = k + 1
        jobs.append(
            {
                "strip_id": f"{board}_s{k}",
                "pieces": strip_pieces,
                "split": split[board],
                "px_per_mm": px_per_mm[board],
                "classes": classes,
                "rename": rename,
                "target_px_per_mm": args.target_px_per_mm,
                "tile": args.tile,
                "overlap": overlap,
                "black_threshold": args.black_threshold,
                "out": out,
            }
        )
    results: List[StripResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(build_strip, jobs):
            LOGGER.info("%s: %d tiles", result.strip["strip"], len(result.tiles))
            results.append(result)
    write_outputs(out, results, classes, priority, split, overlap, px_per_mm, args)


def build_classes(meta: Dict, order: Optional[List[str]], ignored: List[str], rename: Dict[str, str]) -> List[Dict]:
    meta_classes = {rename.get(c["title"], c["title"]): c for c in meta.get("classes", [])}
    names = order if order is not None else [n for n in meta_classes if n.lower() != WOOD]
    missing = [n for n in names if meta_classes and n not in meta_classes]
    if missing:
        msg = f"classes not in meta.json: {missing}"
        raise ValueError(msg)
    unmapped = sorted(n for n in meta_classes if n not in names and n not in (IGNORE, WOOD))
    if unmapped:
        LOGGER.warning("classes left out (their pixels stay wood/background): %s", unmapped)
    classes = [
        {"index": 0, "name": BACKGROUND, "colour": [0, 0, 0], "ignored": False},
        {"index": 1, "name": WOOD, "colour": WOOD_COLOUR, "ignored": False},
        {"index": IGNORE_INDEX, "name": IGNORE, "colour": [255, 255, 255], "ignored": True},
    ]
    for i, name in enumerate(names, start=2):
        source = next((c for t, c in meta_classes.items() if t == name), None)
        colour = _hex_to_rgb(source["color"]) if source else _default_colour(i)
        classes.append({"index": i, "name": name, "colour": colour, "ignored": name in ignored})
    return classes


def collect_pieces(project_dir: Path, args: argparse.Namespace) -> List[Piece]:
    listed = set(args.piece_list.read_text().split()) if args.piece_list else None
    pieces = []
    for img_path in sorted(project_dir.glob("*/img/*")):
        match = PIECE_RE.match(img_path.name)
        if not match:
            continue
        ann_path = img_path.parent.parent / "ann" / f"{img_path.name}.json"
        ann = json.loads(ann_path.read_text()) if ann_path.exists() else {"objects": [], "tags": []}
        tags = {t["name"] for t in ann.get("tags", [])}
        if tags & set(args.exclude_tag) or (args.require_objects and not ann.get("objects")):
            continue
        keep = {
            "all": True,
            "objects": bool(ann.get("objects")),
            "tag": args.annotated_tag in tags,
            "list": listed is not None and (img_path.name in listed or img_path.stem in listed),
        }[args.annotated]
        if not keep:
            continue
        size = ann.get("size", {})
        pieces.append(
            Piece(
                name=img_path.name,
                board=match["board"],
                species=match["species"],
                x=int(match["x"]),
                img_path=img_path,
                ann_path=ann_path if ann_path.exists() else None,
                width=int(size.get("width", 0)),
                height=int(size.get("height", 0)),
            )
        )
    for piece in pieces:
        if not piece.width:
            image = cv2.imread(str(piece.img_path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            piece.height, piece.width = image.shape[0] * 2, image.shape[1] * 2
    return pieces


def group_strips(pieces: List[Piece]) -> List[List[Piece]]:
    by_board: Dict[str, List[Piece]] = {}
    for piece in pieces:
        by_board.setdefault(piece.board, []).append(piece)
    strips = []
    for board in sorted(by_board):
        run: List[Piece] = []
        for piece in sorted(by_board[board], key=lambda p: p.x):
            if run and run[-1].x + run[-1].width != piece.x:
                strips.append(run)
                run = []
            run.append(piece)
        strips.append(run)
    return strips


def load_or_make_split(boards: List[str], args: argparse.Namespace) -> Dict[str, str]:
    if args.split_file:
        fixed = json.loads(args.split_file.read_text())
        split = {b: name for name, members in fixed.items() for b in members}
        missing = [b for b in boards if b not in split]
        if missing:
            msg = f"boards missing from split file: {missing}"
            raise ValueError(msg)
        return {b: split[b] for b in boards}
    split = {}
    rng = random.Random(args.split_seed)
    for species in sorted({b.split("_")[0] for b in boards}):
        members = sorted(b for b in boards if b.startswith(species + "_"))
        rng.shuffle(members)
        n_test = max(1, round(len(members) * args.split_ratios[2])) if len(members) >= 3 else 0
        n_val = max(1, round(len(members) * args.split_ratios[1])) if len(members) >= 3 else 0
        for i, board in enumerate(members):
            split[board] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split


def build_strip(job: Dict) -> StripResult:
    pieces: List[Piece] = job["pieces"]
    classes: List[Dict] = job["classes"]
    species = pieces[0].species
    images, wood_parts, wood_info = [], [], []
    for piece in pieces:
        image = cv2.imread(str(piece.img_path), cv2.IMREAD_COLOR)
        if image.shape[:2] != (piece.height, piece.width):
            msg = f"{piece.name}: image {image.shape[:2]} != ann size {(piece.height, piece.width)}"
            raise ValueError(msg)
        wood, areas = wood_components(image, WOOD_N[species], job["black_threshold"])
        images.append(image)
        wood_parts.append(wood)
        wood_info.append({"piece": piece.name, "species": species, "n_kept": len(areas), "areas": areas})
    heights = {im.shape[0] for im in images}
    if len(heights) != 1:
        msg = f"{job['strip_id']}: pieces differ in height {heights}"
        raise ValueError(msg)
    image = np.concatenate(images, axis=1)
    del images
    native = np.concatenate(wood_parts, axis=1).astype(np.uint8)
    del wood_parts
    name_to_index = {c["name"]: c["index"] for c in classes}
    objects = []
    for piece in pieces:
        ann = json.loads(piece.ann_path.read_text()) if piece.ann_path else {"objects": []}
        offset = piece.x - pieces[0].x
        for obj in ann.get("objects", []):
            title = job["rename"].get(obj["classTitle"], obj["classTitle"])
            if title not in name_to_index or title in (BACKGROUND, WOOD):
                continue
            mask, bbox = rasterise_object(obj, piece.height, piece.width)
            if bbox is None:
                continue
            objects.append(
                {
                    "source_class": obj["classTitle"],
                    "class": title,
                    "index": name_to_index[title],
                    "piece": piece.name,
                    "offset": offset,
                    "bbox": bbox,
                    "mask": mask,
                }
            )
    paint_order = [c for c in classes if c["index"] not in (0, 1)]
    defect_order = [c for c in paint_order if c["index"] != IGNORE_INDEX]
    for cls in paint_order:
        for obj in objects:
            if obj["index"] != cls["index"]:
                continue
            x0, y0, x1, y1 = obj["bbox"]
            ox = obj["offset"]
            region = native[y0:y1, ox + x0 : ox + x1]
            region[obj["mask"]] = cls["index"]
    scale = job["target_px_per_mm"] / job["px_per_mm"]
    out_h, out_w = round(image.shape[0] * scale), round(image.shape[1] * scale)
    small_image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    del image
    small_mask = downsample_labels(native, (out_w, out_h), [c["index"] for c in classes])
    del native
    tile, overlap = job["tile"], job["overlap"]
    canvas_h, canvas_w = max(out_h, tile), max(out_w, tile)
    pad_y, pad_x = canvas_h - out_h, canvas_w - out_w
    if pad_y or pad_x:
        small_image = cv2.copyMakeBorder(small_image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
        small_mask = cv2.copyMakeBorder(small_mask, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    xs = tile_starts(canvas_w, tile, tile - overlap)
    ys = tile_starts(canvas_h, tile, tile - overlap)
    obj_boxes = []
    for obj in objects:
        x0, y0, x1, y1 = obj["bbox"]
        ox = obj["offset"]
        box = (
            math.floor((ox + x0) * scale),
            math.floor(y0 * scale),
            min(out_w, math.ceil((ox + x1) * scale)),
            min(out_h, math.ceil(y1 * scale)),
        )
        obj_boxes.append(box)
    defect_names = [c["name"] for c in defect_order]
    tiles = []
    covered = np.zeros(len(objects), bool)
    for y in ys:
        for x in xs:
            tile_id = f"{job['strip_id']}_x{x:05d}_y{y:05d}"
            tile_img = small_image[y : y + tile, x : x + tile]
            tile_mask = small_mask[y : y + tile, x : x + tile]
            cv2.imwrite(str(job["out"] / "images" / f"{tile_id}.png"), tile_img)
            cv2.imwrite(str(job["out"] / "masks" / f"{tile_id}.png"), tile_mask)
            valid = tile_mask != IGNORE_INDEX
            row = {
                "tile_id": tile_id,
                "board": pieces[0].board,
                "species": species,
                "split": job["split"],
                "strip": job["strip_id"],
                "x": x,
                "y": y,
                "w": tile,
                "h": tile,
                "pad": int((~valid).any(axis=0).sum() + (~valid).any(axis=1).sum()),
                "wood_frac": round(float(((tile_mask != 0) & valid).mean()), 4),
            }
            for name in defect_names:
                row[f"n_{name}"] = 0
            for i, (bx0, by0, bx1, by1) in enumerate(obj_boxes):
                if bx0 >= x and by0 >= y and bx1 <= x + tile and by1 <= y + tile:
                    if objects[i]["class"] in defect_names:
                        row[f"n_{objects[i]['class']}"] += 1
                    covered[i] = True
            tiles.append(row)
    object_rows = [
        {
            "strip": job["strip_id"],
            "piece": obj["piece"],
            "class": obj["class"],
            "source_class": obj["source_class"],
            "w_mm": round((obj["bbox"][2] - obj["bbox"][0]) / job["px_per_mm"], 1),
            "h_mm": round((obj["bbox"][3] - obj["bbox"][1]) / job["px_per_mm"], 1),
            "whole_in_tile": bool(covered[i]),
        }
        for i, obj in enumerate(objects)
    ]
    strip = {
        "strip": job["strip_id"],
        "board": pieces[0].board,
        "species": species,
        "split": job["split"],
        "pieces": ";".join(p.name for p in pieces),
        "x0_native": pieces[0].x,
        "px_per_mm_native": job["px_per_mm"],
        "scale": round(scale, 6),
        "width": out_w,
        "height": out_h,
        "n_tiles": len(tiles),
    }
    return StripResult(strip=strip, tiles=tiles, objects=object_rows, wood_components=wood_info)


def wood_components(image: np.ndarray, n: int, threshold: int) -> Tuple[np.ndarray, List[int]]:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((grey > threshold).astype(np.uint8), connectivity=8)
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:n] + 1
    keep = np.zeros(count, np.uint8)
    keep[order] = 1
    mask = keep[labels]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool), [int(stats[i, cv2.CC_STAT_AREA]) for i in order]


def rasterise_object(obj: Dict, height: int, width: int) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    geometry = obj.get("geometryType")
    if geometry == "bitmap":
        raw = zlib.decompress(base64.b64decode(obj["bitmap"]["data"]))
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        local = (
            (decoded[..., 3] > 0)
            if decoded.ndim == 3 and decoded.shape[2] == 4
            else (decoded.reshape(decoded.shape[0], decoded.shape[1], -1).max(axis=2) > 0)
        )
        ox, oy = obj["bitmap"]["origin"]
        full = np.zeros((height, width), bool)
        h, w = local.shape
        x1, y1 = min(width, ox + w), min(height, oy + h)
        full[oy:y1, ox:x1] = local[: y1 - oy, : x1 - ox]
    elif geometry in ("polygon", "rectangle"):
        exterior = np.asarray(obj["points"]["exterior"], np.float64)
        if geometry == "rectangle":
            (ax, ay), (bx, by) = exterior
            exterior = np.array([[ax, ay], [bx, ay], [bx, by], [ax, by]])
        x0 = max(0, int(np.floor(exterior[:, 0].min())) - 1)
        y0 = max(0, int(np.floor(exterior[:, 1].min())) - 1)
        x1 = min(width, int(np.ceil(exterior[:, 0].max())) + 2)
        y1 = min(height, int(np.ceil(exterior[:, 1].max())) + 2)
        local = np.zeros((max(0, y1 - y0), max(0, x1 - x0)), np.uint8)
        cv2.fillPoly(local, [np.round(exterior - [x0, y0]).astype(np.int32)], 1)
        for hole in obj["points"].get("interior", []):
            cv2.fillPoly(local, [np.round(np.asarray(hole) - [x0, y0]).astype(np.int32)], 0)
        full = None
    else:
        LOGGER.warning("skipping geometry %s", geometry)
        return np.zeros((0, 0), bool), None
    if full is not None:
        ys, xs = np.nonzero(full)
        if not len(xs):
            return np.zeros((0, 0), bool), None
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        return full[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))
    ys, xs = np.nonzero(local)
    if not len(xs):
        return np.zeros((0, 0), bool), None
    lx0, ly0, lx1, ly1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return local[ly0:ly1, lx0:lx1].astype(bool), (int(x0 + lx0), int(y0 + ly0), int(x0 + lx1), int(y0 + ly1))


def downsample_labels(native: np.ndarray, size: Tuple[int, int], priority: List[int]) -> np.ndarray:
    present = [i for i in priority if i == 0 or (native == i).any()]
    best = np.full((size[1], size[0]), -1.0, np.float32)
    label = np.zeros((size[1], size[0]), np.uint8)
    for index in present:
        soft = cv2.resize((native == index).astype(np.float32), size, interpolation=cv2.INTER_AREA)
        win = soft >= best
        label[win] = index
        best[win] = soft[win]
    return label


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def write_outputs(
    out: Path,
    results: List[StripResult],
    classes: List[Dict],
    priority: List[str],
    split: Dict[str, str],
    overlap: int,
    px_per_mm: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    tiles = pd.DataFrame([t for r in results for t in r.tiles]).sort_values("tile_id")
    strips = pd.DataFrame([r.strip for r in results]).sort_values("strip")
    objects = pd.DataFrame([o for r in results for o in r.objects])
    wood = [w for r in results for w in r.wood_components]
    tiles.to_csv(out / "tiles.csv", index=False)
    strips.to_csv(out / "strips.csv", index=False)
    if len(objects):
        objects.to_csv(out / "objects.csv", index=False)
    (out / "wood_components.json").write_text(json.dumps(wood, indent=1))
    (out / "classes.json").write_text(json.dumps(sorted(classes, key=lambda c: c["index"]), indent=2))
    counts = tiles.groupby(["split", "species"]).size().unstack(fill_value=0)
    board_counts = strips.drop_duplicates("board").groupby(["split", "species"]).size().unstack(fill_value=0)
    native = {
        sp: float(np.median([px_per_mm[b] for b in strips.board[strips.species == sp].unique()]))
        for sp in sorted(strips.species.unique())
    }
    snapshot = {"path": str(args.project_dir)}
    if args.snapshot_info:
        snapshot.update(json.loads(args.snapshot_info.read_text()))
    dataset = {
        "name": out.name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_commit": _git_commit(),
        "source_snapshot": snapshot,
        "annotated_piece_rule": args.annotated_piece_rule
        or f"--annotated {args.annotated}"
        + (f" --annotated_tag {args.annotated_tag}" if args.annotated == "tag" else ""),
        "excluded_tags": args.exclude_tag,
        "target_px_per_mm": args.target_px_per_mm,
        "native_px_per_mm": native,
        "tile_size": args.tile,
        "overlap_px": overlap,
        "stride_px": args.tile - overlap,
        "knot_p99_extent": {
            "mm": args.knot_p99_mm,
            "px_target": None if args.knot_p99_mm is None else round(args.knot_p99_mm * args.target_px_per_mm, 1),
        },
        "wood_rule": {
            "rule": "N largest non-black components per piece, holes filled",
            "black_threshold": args.black_threshold,
            **WOOD_N,
        },
        "class_priority": priority,
        "mask_encoding": (
            "uint8 class index per pixel (classes.json), 255 = ignore padding; rasterised at native resolution with "
            "class_priority (later wins), one binary map per class area-averaged to target resolution, argmax with "
            "ties to the higher priority; images area-resampled"
        ),
        "split": {
            "seed": args.split_seed,
            "ratios": args.split_ratios,
            "stratify": "species",
            "split_file": str(args.split_file) if args.split_file else None,
            "boards": {name: sorted(b for b, s in split.items() if s == name) for name in ("train", "val", "test")},
        },
        "counts": {
            "tiles": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in counts.iterrows()},
            "boards": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in board_counts.iterrows()},
            "strips": len(strips),
            "objects": int(len(objects)),
            "objects_not_whole_in_any_tile": int((~objects.whole_in_tile).sum()) if len(objects) else 0,
        },
    }
    (out / "DATASET.json").write_text(json.dumps(dataset, indent=2))
    LOGGER.info("tiles per split x species:\n%s", counts)


def _read_meta(project_dir: Path) -> Dict:
    path = project_dir / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _read_px_per_mm(manifest: Path) -> Dict[str, float]:
    frame = pd.read_csv(manifest)
    frame = frame[frame.px_per_mm.notna()]
    return dict(zip(frame.board, frame.px_per_mm.astype(float)))


def _hex_to_rgb(colour: str) -> List[int]:
    colour = colour.lstrip("#")
    return [int(colour[i : i + 2], 16) for i in (0, 2, 4)]


def _default_colour(index: int) -> List[int]:
    hue = (index * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return [int(bgr[2]), int(bgr[1]), int(bgr[0])]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


if __name__ == "__main__":
    main()
