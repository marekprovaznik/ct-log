import dataclasses
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = (".bmp", ".png")
SEGMENTATION_SUFFIXES = ("_segm.png", "_segm.bmp")
BBOX_SUFFIX = "_anno.txt"
LABEL_WIDTH = 2800

Image.MAX_IMAGE_PIXELS = None


@dataclasses.dataclass(frozen=True)
class KodytekClass:
    label_id: int
    name: str
    color: Tuple[int, int, int]
    aliases: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Box:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float


CLASSES: List[KodytekClass] = [
    KodytekClass(1, "live_knot", (0x00, 0xFF, 0x00), ("live_knot",)),
    KodytekClass(2, "dead_knot", (0xFF, 0x00, 0x00), ("dead_knot", "death_know")),
    KodytekClass(3, "knot_missing", (0xFF, 0x64, 0x00), ("knot_missing",)),
    KodytekClass(4, "knot_with_crack", (0xFF, 0xAF, 0x00), ("knot_with_crack",)),
    KodytekClass(5, "crack", (0xFF, 0x00, 0x64), ("crack",)),
    KodytekClass(6, "quartzite", (0x64, 0x00, 0x64), ("quartzite", "quartzity")),
    KodytekClass(7, "resin", (0xFF, 0x00, 0xFF), ("resin",)),
    KodytekClass(8, "marrow", (0x00, 0x00, 0xFF), ("marrow",)),
    KodytekClass(10, "blue_stain", (0x10, 0xFF, 0xFF), ("blue_stain",)),
    KodytekClass(11, "overgrown", (0x00, 0x40, 0x00), ("overgrown",)),
]

BY_ID: Dict[int, KodytekClass] = {c.label_id: c for c in CLASSES}
BY_ALIAS: Dict[str, KodytekClass] = {alias: c for c in CLASSES for alias in c.aliases}


def canonical(name: str) -> KodytekClass:
    key = name.strip().lower()
    if key not in BY_ALIAS:
        msg = "Unknown Kodytek class name: %r" % name
        raise ValueError(msg)
    return BY_ALIAS[key]


def read_bboxes(path: pathlib.Path) -> List[Box]:
    boxes = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 5:
            msg = "%s:%d: expected 5 tab-separated fields, got %d" % (
                path,
                line_no,
                len(fields),
            )
            raise ValueError(msg)
        x1, y1, x2, y2 = (float(v) for v in fields[1:])
        boxes.append(Box(canonical(fields[0]).name, x1, y1, x2, y2))
    return boxes


def write_bboxes(path: pathlib.Path, boxes: List[Box]) -> None:
    lines = ["%s\t%.6f\t%.6f\t%.6f\t%.6f" % (b.name, b.x1, b.y1, b.x2, b.y2) for b in boxes]
    path.write_text("".join(line + "\n" for line in lines))


def semantic_map_to_ids(rgb: np.ndarray) -> np.ndarray:
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    ids = np.zeros(rgb.shape[:2], dtype=np.uint8)
    known = packed == 0
    for c in CLASSES:
        hit = packed == ((c.color[0] << 16) | (c.color[1] << 8) | c.color[2])
        ids[hit] = c.label_id
        known |= hit
    if not known.all():
        unknown = np.unique(packed[~known])[:5]
        msg = "Semantic map has colours outside the spec: %s" % ", ".join("%06X" % v for v in unknown)
        raise ValueError(msg)
    return ids


def image_id(path: pathlib.Path) -> str:
    name = path.name
    for suffix in (*SEGMENTATION_SUFFIXES, BBOX_SUFFIX, *IMAGE_SUFFIXES):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def find_segmentation(directory: pathlib.Path, sample_id: str) -> Optional[pathlib.Path]:
    for suffix in SEGMENTATION_SUFFIXES:
        candidate = directory / (sample_id + suffix)
        if candidate.exists():
            return candidate
    return None


def find_bboxes(directory: pathlib.Path, sample_id: str) -> Optional[pathlib.Path]:
    candidate = directory / (sample_id + BBOX_SUFFIX)
    return candidate if candidate.exists() else None


def list_images(sources: List[pathlib.Path]) -> List[pathlib.Path]:
    images = []
    for source in sources:
        if source.is_dir():
            images.extend(p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES)
        elif source.is_file():
            images.append(source)
        else:
            msg = "Image path does not exist: %s" % source
            raise ValueError(msg)
    return images


def load_image(path: pathlib.Path) -> Tuple[np.ndarray, int]:
    image = np.asarray(Image.open(path).convert("RGB"))
    width = image.shape[1]
    if width < LABEL_WIDTH:
        image = np.pad(image, ((0, 0), (0, LABEL_WIDTH - width), (0, 0)))
    return image, width


def load_semantic_map(path: pathlib.Path) -> np.ndarray:
    class_ids = semantic_map_to_ids(np.asarray(Image.open(path).convert("RGB")))
    width = class_ids.shape[1]
    return np.pad(class_ids, ((0, 0), (0, max(LABEL_WIDTH - width, 0))))
