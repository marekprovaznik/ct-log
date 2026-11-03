import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

OVERLAY: Dict[int, Tuple[int, int, int]] = {
    2: (0, 200, 255),
    3: (255, 60, 0),
    255: (128, 128, 128),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image | annotation | prediction panels for a window of a timber strip."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--strip", required=True)
    parser.add_argument("--x0", type=int, default=0)
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out_width_px", type=int, default=2400)
    parser.add_argument("--alpha", type=float, default=0.75)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    image, mask = assemble_strip(args.dataset_root, args.strip)
    pred = np.array(Image.open(args.pred_dir / f"{args.strip}.png"))
    if pred.shape != mask.shape:
        msg = f"Prediction shape {pred.shape} differs from strip shape {mask.shape}"
        raise ValueError(msg)

    window = slice(args.x0, args.x0 + args.width)
    panels = [
        image[:, window],
        overlay(image[:, window], mask[:, window], args.alpha),
        overlay(image[:, window], pred[:, window], args.alpha),
    ]
    write_figure(panels, ["image", "annotation", "prediction"], args.out, args.out_width_px)
    logger.info("Wrote %s (%s, x %d-%d px at 4 px/mm)", args.out, args.strip, args.x0, args.x0 + panels[0].shape[1])


def assemble_strip(dataset_root: Path, strip: str) -> Tuple[np.ndarray, np.ndarray]:
    tiles = read_tiles(dataset_root / "tiles.csv", strip)
    if not tiles:
        msg = f"No tiles for strip {strip}"
        raise ValueError(msg)
    height = max(int(t["y"]) + int(t["h"]) for t in tiles)
    width = max(int(t["x"]) + int(t["w"]) for t in tiles)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    for tile in tiles:
        x, y = int(tile["x"]), int(tile["y"])
        tile_image = np.array(Image.open(dataset_root / "images" / f"{tile['tile_id']}.png").convert("RGB"))
        tile_mask = np.array(Image.open(dataset_root / "masks" / f"{tile['tile_id']}.png"))
        h = min(tile_image.shape[0], height - y)
        w = min(tile_image.shape[1], width - x)
        image[y : y + h, x : x + w] = tile_image[:h, :w]
        mask[y : y + h, x : x + w] = tile_mask[:h, :w]
    return image, mask


def read_tiles(tiles_csv: Path, strip: str) -> List[Dict[str, str]]:
    with tiles_csv.open() as f:
        rows = [row for row in csv.DictReader(f) if row["strip"] == strip]
    return [row for row in rows if int(row.get("pad", 0) or 0) == 0] or rows


def overlay(image: np.ndarray, classes: np.ndarray, alpha: float) -> np.ndarray:
    out = image.astype(np.float32)
    for value, colour in OVERLAY.items():
        sel = classes == value
        out[sel] = (1 - alpha) * out[sel] + alpha * np.array(colour, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def write_figure(panels: List[np.ndarray], titles: List[str], out: Path, out_width_px: int) -> None:
    scale = out_width_px / panels[0].shape[1]
    small = [
        np.array(Image.fromarray(p).resize((out_width_px, round(p.shape[0] * scale)), Image.Resampling.LANCZOS))
        for p in panels
    ]
    panel_h = small[0].shape[0]
    title_px = 40
    fig = make_subplots(
        rows=len(small), cols=1, subplot_titles=titles, vertical_spacing=title_px / (len(small) * (panel_h + title_px))
    )
    for row, panel in enumerate(small, start=1):
        fig.add_trace(go.Image(z=panel), row=row, col=1)
        fig.update_xaxes(visible=False, row=row, col=1)
        fig.update_yaxes(visible=False, row=row, col=1)
    fig.update_annotations(font={"family": "Arial", "size": 22})
    fig.update_layout(
        width=out_width_px,
        height=len(small) * (panel_h + title_px) + 20,
        margin={"l": 0, "r": 0, "t": title_px, "b": 0},
        paper_bgcolor="white",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out))


if __name__ == "__main__":
    main()
