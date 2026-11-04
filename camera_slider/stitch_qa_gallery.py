import argparse
import base64
import html
import json
from pathlib import Path
import re
from typing import List

import cv2
import pandas as pd
import plotly.graph_objects as go

from camera_slider.stitch_qa import TYPES

COLOURS = {
    "face_order": "#e45756",
    "face_count": "#b279a2",
    "stray_segment": "#f58518",
    "edge_step": "#eeca3b",
    "face_shift": "#54a24b",
    "bad_seam": "#4c78a8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plotly gallery of pieces flagged by camera_slider.stitch_qa.")
    parser.add_argument("--qa_dir", type=Path, required=True, help="stitch_qa output with pieces.csv")
    parser.add_argument("--pieces_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: <qa_dir>/gallery.html")
    parser.add_argument("--per_type", type=int, default=25, help="most suspicious pieces shown per failure type")
    parser.add_argument("--width", type=int, default=1100)
    args = parser.parse_args()
    build_gallery(args)


def build_gallery(args: argparse.Namespace) -> None:
    pieces = pd.read_csv(args.qa_dir / "pieces.csv", keep_default_na=False, dtype={"order_cycle": str})
    flagged = pieces[pieces.types != ""]
    parts = [f"<h1>{html.escape(args.qa_dir.name)}: {len(flagged)} of {len(pieces)} pieces flagged</h1>"]
    parts.append(legend())
    first = True
    for kind in TYPES:
        group = flagged[flagged.types.str.split(";").apply(lambda ts, k=kind: k in ts)]
        if group.empty:
            continue
        group = group.sort_values(f"r_{kind}", ascending=False)
        shown = group.head(args.per_type)
        parts.append(f"<h2>{kind}: {len(group)} pieces (showing {len(shown)})</h2>")
        for _, row in shown.iterrows():
            fig = piece_figure(row, kind, args)
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
            first = False
    out = args.out or args.qa_dir / "gallery.html"
    out.write_text(
        "<html><head><meta charset='utf-8'><title>Stitch QA gallery</title>"
        "<style>body{font-family:sans-serif;margin:16px}</style></head><body>" + "".join(parts) + "</body></html>"
    )
    print(out)


def piece_figure(row: pd.Series, kind: str, args: argparse.Namespace) -> go.Figure:
    image = cv2.imread(str(args.pieces_dir / f"{row.piece}.jpg"), cv2.IMREAD_REDUCED_COLOR_8)
    scale = args.width / image.shape[1]
    image = cv2.resize(image, (args.width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    source = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()
    fig = go.Figure(go.Image(source=source))
    s = scale / 8
    for shape in overlay_shapes(row, kind, s, image.shape[0]):
        fig.add_shape(**shape)
    title = "<b>%s</b> [%s] score %.2f, tags: %s<br><span style='font-size:11px'>%s</span>" % (
        row.piece,
        row.types,
        row.score,
        row.get("tags", "") or "-",
        html.escape(row.evidence),
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 13}},
        width=args.width + 40,
        height=image.shape[0] + 110,
        margin={"l": 10, "r": 10, "t": 80, "b": 10},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def overlay_shapes(row: pd.Series, kind: str, s: float, height: int) -> List[dict]:
    colour = COLOURS[kind]
    shapes = []
    if kind == "stray_segment":
        for x, y, w, h in json.loads(row.stray_boxes or "[]"):
            pad = 6
            shapes.append(box(x * s - pad, y * s - pad, (x + w) * s + pad, (y + h) * s + pad, colour))
    if kind == "edge_step" and row.edge_step_at:
        x = int(re.search(r"x=(\d+)", row.edge_step_at).group(1)) * s
        shapes.append(vline(x, height, colour))
    if kind == "face_shift" and row.get("shift_lags"):
        for _, lag, x in json.loads(row.shift_lags or "[]"):
            if abs(lag) >= 5:
                shapes.append(vline(x * s, height, colour))
    if kind == "bad_seam" and row.get("frame_pair"):
        found = re.search(r"x=(-?\d+)", row.frame_pair)
        if found:
            shapes.append(vline(int(found.group(1)) * s, height, colour))
    return shapes


def box(x0: float, y0: float, x1: float, y1: float, colour: str) -> dict:
    return {"type": "rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "line": {"color": colour, "width": 2}}


def vline(x: float, height: int, colour: str) -> dict:
    return {
        "type": "line",
        "x0": x,
        "x1": x,
        "y0": 0,
        "y1": height,
        "line": {"color": colour, "width": 2, "dash": "dot"},
    }


def legend() -> str:
    items = "".join(f"<span style='color:{c};margin-right:16px'>&#9632; {k}</span>" for k, c in COLOURS.items())
    return f"<p>{items}</p>"


if __name__ == "__main__":
    main()
