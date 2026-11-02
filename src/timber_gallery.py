import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IGNORE_INDEX = 255
DISPLAY_COLOURS = {"knot": (255, 60, 0), "resin_pocket": (0, 200, 255)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plotly gallery of reassembled strips: image, ground truth, prediction."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--pred_dir", type=str, required=True, help="<strip>.png class maps from eval_timber strips mode."
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4, help="Downscale factor of the panels.")
    parser.add_argument("--title", type=str, default="timber v1 predictions")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    tiles = pd.read_csv(root / "tiles.csv", dtype={"tile_id": str, "strip": str})
    tiles = tiles[tiles["split"] == args.split]
    classes = json.loads((root / "classes.json").read_text())
    palette = colour_table(classes)
    legend = ", ".join(
        "%s rgb%s" % (c["name"], tuple(DISPLAY_COLOURS.get(c["name"], c["colour"])))
        for c in classes
        if c["name"] not in ("background", "wood", "ignore")
    )
    parts: List[str] = []
    for index, (strip, rows) in enumerate(tiles.groupby("strip", sort=True)):
        pred_path = Path(args.pred_dir) / f"{strip}.png"
        if not pred_path.exists():
            continue
        image, mask = reassemble(root, rows)
        prediction = np.asarray(Image.open(pred_path))
        figure = strip_figure(strip, image, mask, prediction, palette, args.scale)
        parts.append(figure.to_html(full_html=False, include_plotlyjs="cdn" if index == 0 else False))
    html = (
        "<html><head><meta charset='utf-8'><title>%s</title></head><body><h2>%s</h2><p>Split %s, %d strips. Overlay colours: %s; grey = ignored.</p>%s</body></html>"
        % (
            args.title,
            args.title,
            args.split,
            len(parts),
            legend,
            "\n".join(parts),
        )
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print("wrote %s (%d strips)" % (args.out, len(parts)))


def colour_table(classes: List[Dict[str, object]]) -> np.ndarray:
    table = np.zeros((256, 4), dtype=np.float32)
    for entry in classes:
        if entry["name"] in ("background", "wood"):
            continue
        rgb = [float(v) for v in DISPLAY_COLOURS.get(str(entry["name"]), entry["colour"])]
        table[int(entry["index"])] = [*rgb, 0.6]
    table[IGNORE_INDEX] = [128.0, 128.0, 128.0, 0.5]
    return table


def reassemble(root: Path, rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    height = int((rows["y"] + rows["h"]).max())
    width = int((rows["x"] + rows["w"]).max())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.full((height, width), IGNORE_INDEX, dtype=np.uint8)
    for row in rows.itertuples():
        tile = np.asarray(Image.open(root / "images" / f"{row.tile_id}.png").convert("RGB"))
        tile_mask = np.asarray(Image.open(root / "masks" / f"{row.tile_id}.png"))
        image[row.y : row.y + row.h, row.x : row.x + row.w] = tile
        mask[row.y : row.y + row.h, row.x : row.x + row.w] = tile_mask
    return image, mask


def overlay(image: np.ndarray, labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgba = palette[labels]
    alpha = rgba[..., 3:4]
    return (image * (1.0 - alpha) + rgba[..., :3] * alpha).astype(np.uint8)


def strip_figure(
    strip: str, image: np.ndarray, mask: np.ndarray, prediction: np.ndarray, palette: np.ndarray, scale: int
) -> go.Figure:
    small = image[::scale, ::scale]
    panels = [
        ("image", small),
        ("ground truth", overlay(small, mask[::scale, ::scale], palette)),
        ("prediction", overlay(small, prediction[::scale, ::scale], palette)),
    ]
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=[p[0] for p in panels]
    )
    for row, (_, panel) in enumerate(panels, start=1):
        figure.add_trace(go.Image(source=png_data_uri(panel)), row=row, col=1)
    height_px = small.shape[0] * 3 + 120
    figure.update_layout(
        title=strip, height=min(max(height_px, 400), 1400), margin={"l": 10, "r": 10, "t": 60, "b": 10}
    )
    return figure


def png_data_uri(panel: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(panel).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    main()
