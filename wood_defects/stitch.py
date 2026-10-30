import argparse
import collections
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from wood_defects.labels import (
    BY_ID,
    LABEL_WIDTH,
    Box,
    find_bboxes,
    find_segmentation,
    image_id,
    list_images,
    load_image,
    load_semantic_map,
    read_bboxes,
    write_bboxes,
)

logger = logging.getLogger(__name__)

INDEX_DIGITS = 5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch consecutive Kodytek images, semantic maps and boxes along the board."
    )
    parser.add_argument("--images", type=pathlib.Path, required=True)
    parser.add_argument("--segmentation", type=pathlib.Path, default=None)
    parser.add_argument("--bboxes", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seam_tol_px", type=float, default=1.0)
    parser.add_argument(
        "--max_shift_px",
        type=int,
        default=300,
        help="Largest sideways shift searched between consecutive chunks.",
    )
    parser.add_argument(
        "--seam_warn",
        type=float,
        default=6.0,
        help="Warn when the aligned seam differs by more grey levels than this.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    images = {image_id(p): p for p in list_images([args.images])}
    for sub in ("images", "semantic_maps", "bboxes"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    for run in split_runs(sorted(images)):
        stitch_run(run, images, args)


def split_runs(sample_ids: List[str]) -> List[List[str]]:
    runs: List[List[str]] = []
    previous: Optional[Tuple[str, int]] = None
    for sample_id in sorted(sample_ids, key=parse_id):
        prefix, index = parse_id(sample_id)
        if previous is None or previous != (prefix, index - 1):
            runs.append([])
        runs[-1].append(sample_id)
        previous = (prefix, index)
    return runs


def parse_id(sample_id: str) -> Tuple[str, int]:
    if not sample_id.isdigit() or len(sample_id) <= INDEX_DIGITS:
        msg = "Expected a numeric Kodytek id <prefix><%d-digit index>, got %r" % (
            INDEX_DIGITS,
            sample_id,
        )
        raise ValueError(msg)
    return sample_id[:-INDEX_DIGITS], int(sample_id[-INDEX_DIGITS:])


def stitch_run(run: List[str], images: Dict[str, pathlib.Path], args: argparse.Namespace) -> None:
    name = "%s-%s" % (run[0], run[-1])
    chunks = []
    for sample_id in run:
        image, width = load_image(images[sample_id])
        if width < LABEL_WIDTH:
            logger.info("%s: %d px wide, padded right to %d", sample_id, width, LABEL_WIDTH)
        chunks.append(image)
    heights = [c.shape[0] for c in chunks]
    shifts, residuals = estimate_shifts(chunks, args.max_shift_px)
    x_offsets = place(shifts)
    if shifts:
        logger.info(
            "%s: seam x-shifts %s px, seam residual max %.1f grey levels",
            name,
            shifts,
            max(residuals),
        )
    for above, below, residual in zip(run, run[1:], residuals):
        if residual > args.seam_warn:
            logger.warning(
                "%s -> %s: seam does not match (%.1f grey levels after the best shift)", above, below, residual
            )
    Image.fromarray(compose(chunks, x_offsets)).save(args.output / "images" / ("%s.png" % name))
    canvas_width = LABEL_WIDTH + max(x_offsets)
    del chunks

    if args.segmentation is not None:
        maps = [stitch_map_chunk(args.segmentation, s, h) for s, h in zip(run, heights)]
        colored = colorize(compose(maps, x_offsets))
        Image.fromarray(colored).save(args.output / "semantic_maps" / ("%s_segm.png" % name))

    if args.bboxes is not None:
        per_chunk = []
        for sample_id in run:
            path = find_bboxes(args.bboxes, sample_id)
            if path is None:
                logger.warning("%s: no bbox file in %s, treated as empty", sample_id, args.bboxes)
            per_chunk.append(read_bboxes(path) if path is not None else [])
        boxes, counts = merge_seams(per_chunk, heights, x_offsets, canvas_width, args.seam_tol_px)
        write_bboxes(args.output / "bboxes" / ("%s_anno.txt" % name), boxes)
        logger.info(
            "%s: %d chunks, %d px tall, %d boxes (%d seam merges, %d unpaired seam boxes, %d class mismatches)",
            name,
            len(run),
            sum(heights),
            len(boxes),
            counts["merged"],
            counts["unpaired"],
            counts["mismatch"],
        )
    else:
        logger.info("%s: %d chunks, %d px tall", name, len(run), sum(heights))


def estimate_shifts(chunks: List[np.ndarray], max_shift: int) -> Tuple[List[int], List[float]]:
    shifts = []
    residuals = []
    for above, below in zip(chunks, chunks[1:]):
        last = above[-1].mean(axis=-1)
        first = below[0].mean(axis=-1)
        window = slice(max_shift, LABEL_WIDTH - max_shift)
        scores = {
            dx: float(np.abs(last[window] - first[window.start - dx : window.stop - dx]).mean())
            for dx in range(-max_shift, max_shift + 1)
        }
        best = min(scores, key=scores.get)
        shifts.append(best)
        residuals.append(scores[best])
    return shifts, residuals


def place(shifts: List[int]) -> List[int]:
    cumulative = np.concatenate([[0], np.cumsum(shifts)]).astype(int)
    return [int(v) for v in cumulative - cumulative.min()]


def compose(chunks: List[np.ndarray], x_offsets: List[int]) -> np.ndarray:
    height = sum(c.shape[0] for c in chunks)
    width = max(o + c.shape[1] for c, o in zip(chunks, x_offsets))
    canvas = np.zeros((height, width, *chunks[0].shape[2:]), dtype=chunks[0].dtype)
    y = 0
    for chunk, x in zip(chunks, x_offsets):
        canvas[y : y + chunk.shape[0], x : x + chunk.shape[1]] = chunk
        y += chunk.shape[0]
    return canvas


def stitch_map_chunk(directory: pathlib.Path, sample_id: str, height: int) -> np.ndarray:
    path = find_segmentation(directory, sample_id)
    if path is None:
        logger.warning("%s: no semantic map in %s, treated as empty", sample_id, directory)
        return np.zeros((height, LABEL_WIDTH), dtype=np.uint8)
    class_ids = load_semantic_map(path)
    if class_ids.shape[0] != height:
        msg = "%s: semantic map height %d does not match image height %d" % (
            sample_id,
            class_ids.shape[0],
            height,
        )
        raise ValueError(msg)
    return class_ids


def colorize(class_ids: np.ndarray) -> np.ndarray:
    palette = np.zeros((256, 3), dtype=np.uint8)
    for label_id, cls in BY_ID.items():
        palette[label_id] = cls.color
    return palette[class_ids]


def merge_seams(
    per_chunk: List[List[Box]],
    heights: List[int],
    x_offsets: List[int],
    canvas_width: int,
    tol_px: float,
) -> Tuple[List[Box], Dict[str, int]]:
    offsets = np.concatenate([[0], np.cumsum(heights)])
    pixel_boxes = []
    chunk_of = []
    for k, boxes in enumerate(per_chunk):
        for b in boxes:
            pixel_boxes.append(
                (
                    b.name,
                    x_offsets[k] + b.x1 * LABEL_WIDTH,
                    offsets[k] + b.y1 * heights[k],
                    x_offsets[k] + b.x2 * LABEL_WIDTH,
                    offsets[k] + b.y2 * heights[k],
                )
            )
            chunk_of.append(k)

    parent = list(range(len(pixel_boxes)))
    counts: Dict[str, int] = collections.Counter()
    for k in range(len(heights) - 1):
        seam = offsets[k + 1]
        paired = set()
        above = [i for i, c in enumerate(chunk_of) if c == k and pixel_boxes[i][4] >= seam - tol_px]
        below = [i for i, c in enumerate(chunk_of) if c == k + 1 and pixel_boxes[i][2] <= seam + tol_px]
        for i in above:
            for j in below:
                if min(pixel_boxes[i][3], pixel_boxes[j][3]) <= max(pixel_boxes[i][1], pixel_boxes[j][1]):
                    continue
                if pixel_boxes[i][0] != pixel_boxes[j][0]:
                    counts["mismatch"] += 1
                    continue
                parent[find(parent, i)] = find(parent, j)
                paired.update((i, j))
                counts["merged"] += 1
        counts["unpaired"] += sum(1 for i in (*above, *below) if i not in paired)

    groups: Dict[int, List[int]] = collections.defaultdict(list)
    for i in range(len(pixel_boxes)):
        groups[find(parent, i)].append(i)
    total = float(offsets[-1])
    merged = []
    for members in sorted(groups.values(), key=lambda m: min(pixel_boxes[i][2] for i in m)):
        merged.append(
            Box(
                pixel_boxes[members[0]][0],
                min(pixel_boxes[i][1] for i in members) / canvas_width,
                min(pixel_boxes[i][2] for i in members) / total,
                max(pixel_boxes[i][3] for i in members) / canvas_width,
                max(pixel_boxes[i][4] for i in members) / total,
            )
        )
    return merged, counts


def find(parent: List[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


if __name__ == "__main__":
    main()
