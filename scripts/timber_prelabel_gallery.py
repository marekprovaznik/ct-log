import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

from camera_slider.training_set import rasterise_object
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

Image.MAX_IMAGE_PIXELS = None
ADDED_COLOURS = {"knot_unclassified": (255, 60, 0), "resin_pocket": (0, 200, 255)}
EXISTING_COLOUR = (40, 220, 40)
WOOD_EDGE_COLOUR = (255, 255, 0)
ALPHA = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plotly gallery of merged pre-label annotations from a timber_prelabel_upload dry run."
    )
    parser.add_argument("--dry_run_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--pieces", type=str, nargs="+", required=True)
    parser.add_argument("--wood_class", type=str, default="wood")
    parser.add_argument("--width", type=int, default=2400, help="Panel width in px.")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    counts = {row["name"]: row for row in json.loads((Path(args.dry_run_dir) / "counts.json").read_text())}
    panels: List[Tuple[str, np.ndarray]] = []
    for name in args.pieces:
        ann = json.loads((Path(args.dry_run_dir) / "ann" / (name + ".json")).read_text())
        image = cv2.cvtColor(cv2.imread(str(Path(args.images_dir) / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        row = counts[name]
        title = "%s | bumaska %s | existing %s | added %s" % (name, row["reviewed"], row["existing"], row["added"])
        panels.append((title, render(image, ann, len(ann["objects"]) - sum(row["added"].values()), args)))
    figure = make_subplots(rows=len(panels), cols=1, subplot_titles=[t for t, _ in panels], vertical_spacing=0.03)
    for index, (_, panel) in enumerate(panels, start=1):
        figure.add_trace(go.Image(source=to_data_uri(panel)), row=index, col=1)
        figure.update_xaxes(visible=False, row=index, col=1)
        figure.update_yaxes(visible=False, row=index, col=1)
    height = sum(int(p.shape[0]) for _, p in panels) + 90 * len(panels)
    figure.update_layout(
        title="KAR pre-labels (dry run): added knot_unclassified orange, added resin_pocket cyan, "
        "existing objects green, wood outline yellow",
        width=args.width + 80,
        height=height,
        margin={"l": 20, "r": 20, "t": 90, "b": 20},
    )
    figure.write_html(args.out, include_plotlyjs="cdn")
    print(args.out)


def render(image: np.ndarray, ann: Dict, n_existing: int, args: argparse.Namespace) -> np.ndarray:
    height, width = image.shape[:2]
    scale = args.width / width
    size = (args.width, max(1, round(height * scale)))
    canvas = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    for index, obj in enumerate(ann["objects"]):
        mask, bbox = rasterise_object(obj, height, width)
        if bbox is None:
            continue
        x0, y0 = int(bbox[0] * scale), int(bbox[1] * scale)
        x1 = min(size[0], max(x0 + 1, round(bbox[2] * scale)))
        y1 = min(size[1], max(y0 + 1, round(bbox[3] * scale)))
        small = np.zeros((size[1], size[0]), bool)
        coverage = cv2.resize(mask.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        small[y0:y1, x0:x1] = coverage > 0
        if obj["classTitle"] == args.wood_class:
            edge = small & ~cv2.erode(small.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            canvas[edge] = WOOD_EDGE_COLOUR
            continue
        colour = EXISTING_COLOUR if index < n_existing else ADDED_COLOURS.get(obj["classTitle"], (255, 0, 255))
        canvas[small] = (1 - ALPHA) * canvas[small] + ALPHA * np.asarray(colour, np.float32)
    return canvas.clip(0, 255).astype(np.uint8)


def to_data_uri(panel: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(panel).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    main()
