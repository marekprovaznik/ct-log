import argparse
import logging
import pathlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import plotly.express as px

from wood_defects.labels import (
    BY_ID,
    LABEL_WIDTH,
    Box,
    canonical,
    find_bboxes,
    find_segmentation,
    image_id,
    list_images,
    load_image,
    load_semantic_map,
    read_bboxes,
)

logger = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
BOX_THICKNESS = 2
FIGURE_MARGIN = 40


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay Kodytek semantic maps and bounding boxes on images.")
    parser.add_argument(
        "--images",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="Image directories or files.",
    )
    parser.add_argument(
        "--segmentation",
        type=pathlib.Path,
        default=None,
        help="Directory of <id>_segm.(bmp|png).",
    )
    parser.add_argument("--bboxes", type=pathlib.Path, default=None, help="Directory of <id>_anno.txt.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Save PNGs here instead of showing them.",
    )
    parser.add_argument("--ids", type=str, nargs="+", default=None, help="Only these sample ids.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor applied before drawing.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Opacity of the mask fill.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.segmentation is None and args.bboxes is None:
        parser.error("give --segmentation, --bboxes or both")

    images = list_images(args.images)
    if args.ids is not None:
        wanted = set(args.ids)
        images = [p for p in images if image_id(p) in wanted]
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        parser.error("no images selected")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    for path in images:
        sample_id = image_id(path)
        image, width = load_image(path)
        if width < LABEL_WIDTH:
            logger.info("%s: %d px wide, padded right to %d", sample_id, width, LABEL_WIDTH)
        rendered = render(
            image=image,
            class_ids=load_segmentation(args.segmentation, sample_id),
            boxes=load_boxes(args.bboxes, sample_id),
            scale=args.scale,
            alpha=args.alpha,
        )
        if args.output is None:
            figure = px.imshow(rendered, title=sample_id, binary_string=True)
            figure.update_layout(
                width=rendered.shape[1] + 2 * FIGURE_MARGIN,
                height=rendered.shape[0] + 2 * FIGURE_MARGIN,
                margin={
                    "l": FIGURE_MARGIN,
                    "r": FIGURE_MARGIN,
                    "t": FIGURE_MARGIN,
                    "b": FIGURE_MARGIN,
                },
            )
            figure.show()
        else:
            out_path = args.output / ("%s.png" % sample_id)
            Image.fromarray(rendered).save(out_path)
            logger.info("wrote %s", out_path)


def load_segmentation(directory: Optional[pathlib.Path], sample_id: str) -> Optional[np.ndarray]:
    if directory is None:
        return None
    path = find_segmentation(directory, sample_id)
    if path is None:
        logger.warning("%s: no semantic map in %s", sample_id, directory)
        return None
    return load_semantic_map(path)


def load_boxes(directory: Optional[pathlib.Path], sample_id: str) -> Optional[List[Box]]:
    if directory is None:
        return None
    path = find_bboxes(directory, sample_id)
    if path is None:
        logger.warning("%s: no bbox file in %s", sample_id, directory)
        return None
    return read_bboxes(path)


def render(
    image: np.ndarray,
    class_ids: Optional[np.ndarray],
    boxes: Optional[List[Box]],
    scale: float,
    alpha: float,
) -> np.ndarray:
    if class_ids is not None and class_ids.shape != image.shape[:2]:
        msg = "Semantic map %s does not match image %s" % (
            class_ids.shape,
            image.shape[:2],
        )
        raise ValueError(msg)
    if scale != 1.0:
        size = (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        )
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if class_ids is not None:
            class_ids = cv2.resize(class_ids, size, interpolation=cv2.INTER_NEAREST)
    out = image.copy()
    if class_ids is not None:
        draw_segments(out, class_ids, alpha, labels=boxes is None)
    if boxes is not None:
        draw_boxes(out, boxes)
    return out


def draw_segments(out: np.ndarray, class_ids: np.ndarray, alpha: float, labels: bool) -> None:
    for label_id in np.unique(class_ids):
        if label_id == 0:
            continue
        cls = BY_ID[int(label_id)]
        mask = (class_ids == label_id).astype(np.uint8)
        color = np.array(cls.color, dtype=np.float32)
        out[mask > 0] = ((1 - alpha) * out[mask > 0] + alpha * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, cls.color, 1)
        if not labels:
            continue
        n_components, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for component in range(1, n_components):
            x, y, _, h = stats[component, :4]
            put_label(out, cls.name, (int(x), int(y + h)), cls.color, below=True)


def draw_boxes(out: np.ndarray, boxes: List[Box]) -> None:
    height, width = out.shape[:2]
    for box in boxes:
        cls = canonical(box.name)
        top_left = (round(box.x1 * width), round(box.y1 * height))
        bottom_right = (round(box.x2 * width) - 1, round(box.y2 * height) - 1)
        cv2.rectangle(out, top_left, bottom_right, cls.color, BOX_THICKNESS)
        put_label(out, cls.name, top_left, cls.color, below=False)


def put_label(
    out: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    color: Tuple[int, int, int],
    below: bool,
) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    box_h = text_h + baseline + 4
    x = min(max(anchor[0], 0), max(out.shape[1] - text_w - 4, 0))
    y = anchor[1] if below else anchor[1] - box_h
    y = min(max(y, 0), max(out.shape[0] - box_h, 0))
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
    cv2.rectangle(out, (x, y), (x + text_w + 4, y + box_h), color, cv2.FILLED)
    cv2.putText(
        out,
        text,
        (x + 2, y + text_h + 2),
        FONT,
        FONT_SCALE,
        text_color,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
