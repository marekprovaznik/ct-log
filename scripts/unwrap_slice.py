import argparse
import base64
import io
import json
import pathlib
from typing import Dict, Tuple
import zlib

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage as ndi


def main() -> None:
    parser = argparse.ArgumentParser(description="Unwrap a whole CT slice around the pith: angle on x, radius on y.")
    parser.add_argument("--log_dir", type=pathlib.Path, default=pathlib.Path("/mnt/D/datasets/ct_log/kwp_ds_v3/10"))
    parser.add_argument("--page", default="page_133")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("report_2026-10/unwrap"))
    parser.add_argument("--deg_step", type=float, default=0.5)
    parser.add_argument("--pad_px", type=int, default=25)
    args = parser.parse_args()

    gray = np.array(Image.open(args.log_dir / "img" / ("%s.tiff" % args.page)))[..., 0].astype(np.float32)
    ann = json.loads((args.log_dir / "ann" / ("%s.tiff.json" % args.page)).read_text())
    knots, wood, pith = read_labels(ann, gray.shape)
    centre = pith if pith is not None else ndi.center_of_mass(wood)
    r_max = outline_radius(wood, centre) + args.pad_px
    theta = np.arange(0.0, 360.0, args.deg_step)
    radius = np.arange(0.0, r_max, 1.0)
    strip = sample_polar(gray, centre, theta, radius, order=1)
    knot_strip = sample_polar(knots.astype(np.float32), centre, theta, radius, order=0)
    wood_strip = sample_polar(wood.astype(np.float32), centre, theta, radius, order=0)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("%s_%s_unwrap.html" % (args.log_dir.name, args.page))
    figure(strip, knot_strip, wood_strip, theta, radius, "log %s %s" % (args.log_dir.name, args.page)).write_html(
        path, include_plotlyjs="cdn"
    )
    print(path)


def read_labels(ann: Dict[str, object], shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    knots = np.zeros(shape, bool)
    wood = np.zeros(shape, bool)
    pith = None
    for obj in ann["objects"]:
        if obj["geometryType"] == "bitmap":
            raw = zlib.decompress(base64.b64decode(obj["bitmap"]["data"]))
            mask = np.array(Image.open(io.BytesIO(raw)))
            mask = mask[..., -1] if mask.ndim == 3 else mask
            x0, y0 = obj["bitmap"]["origin"]
            h, w = mask.shape
            target = knots if obj["classTitle"] == "Knot" else wood
            target[y0 : y0 + h, x0 : x0 + w] |= mask > 0
        elif obj["classTitle"] == "Pith":
            x, y = obj["points"]["exterior"][0]
            pith = (float(y), float(x))
    return knots, wood | knots, pith


def outline_radius(wood: np.ndarray, centre: Tuple[float, float]) -> float:
    ys, xs = np.nonzero(wood)
    return float(np.hypot(ys - centre[0], xs - centre[1]).max())


def sample_polar(
    img: np.ndarray, centre: Tuple[float, float], theta: np.ndarray, radius: np.ndarray, order: int
) -> np.ndarray:
    t = np.deg2rad(theta)[None, :]
    r = radius[:, None]
    rows = centre[0] + r * np.sin(t)
    cols = centre[1] + r * np.cos(t)
    return ndi.map_coordinates(img, [rows, cols], order=order, mode="constant", cval=0.0)


def figure(
    strip: np.ndarray, knot_strip: np.ndarray, wood_strip: np.ndarray, theta: np.ndarray, radius: np.ndarray, title: str
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("CT", "CT with human labels (yellow knots, outline of wood)"),
    )
    grey = [[0, "black"], [1, "white"]]
    fig.add_trace(
        go.Heatmap(
            z=strip,
            x=theta,
            y=radius,
            colorscale=grey,
            showscale=False,
            hovertemplate="angle %{x}°<br>radius %{y} px<br>value %{z:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=strip, x=theta, y=radius, colorscale=grey, showscale=False, hoverinfo="skip"), row=2, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=np.where(knot_strip > 0, 1.0, np.nan),
            x=theta,
            y=radius,
            colorscale=[[0, "rgba(255,220,0,0.55)"], [1, "rgba(255,220,0,0.55)"]],
            showscale=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Contour(
            z=wood_strip,
            x=theta,
            y=radius,
            contours={"start": 0.5, "end": 0.5, "size": 1, "coloring": "lines"},
            line={"color": "#7fd17f", "width": 1},
            showscale=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="distance from pith, px (= mm)", autorange=True)
    fig.update_xaxes(title_text="angle, degrees (0 = right, 90 = down)", row=2, col=1, dtick=30)
    fig.update_layout(
        title=title,
        height=2 * max(320, len(radius)) + 160,
        width=1600,
        plot_bgcolor="black",
        margin={"l": 70, "r": 20, "t": 60, "b": 50},
    )
    return fig


if __name__ == "__main__":
    main()
