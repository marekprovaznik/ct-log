import argparse
import base64
import json
from pathlib import Path
import random
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IGNORE_INDEX = 255


def main() -> None:
    parser = argparse.ArgumentParser(description="Plotly gallery of timber training tiles with mask overlays.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: <root>/gallery.html")
    parser.add_argument("--per_group", type=int, default=6, help="random tiles per split x species")
    parser.add_argument("--per_class", type=int, default=6, help="tiles per defect class")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--thumb", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build_gallery(args)


def build_gallery(args: argparse.Namespace) -> None:
    tiles = pd.read_csv(args.root / "tiles.csv")
    classes = json.loads((args.root / "classes.json").read_text())
    rng = random.Random(args.seed)
    sections = []
    for (split, species), group in tiles.groupby(["split", "species"]):
        ids = group.tile_id.tolist()
        sections.append((f"random: {split} / {species}", rng.sample(ids, min(args.per_group, len(ids)))))
    for cls in classes:
        column = f"n_{cls['name']}"
        if column not in tiles:
            continue
        ids = tiles.tile_id[tiles[column] > 0].tolist()
        if ids:
            sections.append(
                (
                    f"class: {cls['name']} ({len(ids)} tiles contain a whole object)",
                    rng.sample(ids, min(args.per_class, len(ids))),
                )
            )
    lut = colour_lut(classes)
    parts = [legend_html(classes)]
    for title, ids in sections:
        fig = section_figure(args.root, title, ids, tiles.set_index("tile_id"), lut, args)
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if len(parts) == 1 else False))
    out = args.out or args.root / "gallery.html"
    out.write_text(
        f"<html><head><meta charset='utf-8'><title>{args.root.name} gallery</title></head><body>{''.join(parts)}</body></html>"
    )
    print(out)


def section_figure(
    root: Path, title: str, ids: List[str], tiles: pd.DataFrame, lut: np.ndarray, args: argparse.Namespace
) -> go.Figure:
    rows = (len(ids) + args.cols - 1) // args.cols
    titles = []
    for tile_id in ids:
        row = tiles.loc[tile_id]
        counts = ", ".join(f"{c[2:]}={int(row[c])}" for c in tiles.columns if c.startswith("n_") and row[c] > 0)
        titles += [f"{tile_id}<br>{row.split} wood={row.wood_frac:.2f} {counts}"]
    fig = make_subplots(
        rows=rows,
        cols=args.cols * 2,
        subplot_titles=[t for t in titles for _ in (0, 1)],
        horizontal_spacing=0.01,
        vertical_spacing=0.06,
    )
    for i, tile_id in enumerate(ids):
        image = cv2.imread(str(root / "images" / f"{tile_id}.png"), cv2.IMREAD_COLOR)[..., ::-1]
        mask = cv2.imread(str(root / "masks" / f"{tile_id}.png"), cv2.IMREAD_GRAYSCALE)
        overlay = image.copy()
        labelled = mask > 1
        overlay[labelled] = ((1 - args.alpha) * image[labelled] + args.alpha * lut[mask[labelled]]).astype(np.uint8)
        edges = cv2.morphologyEx((mask == 1).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(
            bool
        )
        overlay[edges] = lut[1]
        r, c = i // args.cols + 1, (i % args.cols) * 2 + 1
        fig.add_trace(go.Image(source=_data_uri(image, args.thumb)), row=r, col=c)
        fig.add_trace(go.Image(source=_data_uri(overlay, args.thumb)), row=r, col=c + 1)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_annotations(font_size=9)
    fig.update_layout(
        title=title, height=330 * rows + 80, width=360 * args.cols * 2, margin={"l": 10, "r": 10, "t": 80, "b": 10}
    )
    return fig


def colour_lut(classes: List[Dict]) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for cls in classes:
        lut[cls["index"]] = cls["colour"]
    return lut


def legend_html(classes: List[Dict]) -> str:
    items = "".join(
        f"<span style='display:inline-block;margin-right:14px'><span style='display:inline-block;width:14px;height:14px;background:rgb({c['colour'][0]},{c['colour'][1]},{c['colour'][2]});border:1px solid #888'></span> {c['index']} {c['name']}{' (ignored)' if c['ignored'] else ''}</span>"
        for c in classes
    )
    return f"<div style='font-family:sans-serif;margin:8px'>{items}<br><small>left: image, right: defects filled, wood outline</small></div>"


def _data_uri(image: np.ndarray, size: int) -> str:
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        msg = "JPEG encoding failed"
        raise ValueError(msg)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


if __name__ == "__main__":
    main()
