from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import plotly.graph_objects as go

DPI = 300
WIDTH_MM = 80.0
WIDTH_PX = round(WIDTH_MM / 25.4 * DPI)
FONT_PT = 8.0
FONT_PX = FONT_PT / 72 * DPI
FONT = "Arial"
LEGEND_PX = round(FONT_PX * 2.2)
ALPHA = 0.75
BOX_MARGIN = 10
OVERLAP = (255, 230, 0)
MODEL_ONLY = (255, 0, 255)
HUMAN_ONLY = (0, 230, 255)
KNOT_LEGEND = [(OVERLAP, "overlap"), (MODEL_ONLY, "M5 only"), (HUMAN_ONLY, "human only")]


def save_frame(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    bbox: Optional[Sequence[int]],
    path: Path,
) -> None:
    rgb = overlay(grey(image), [(gt & pred, OVERLAP), (pred & ~gt, MODEL_ONLY), (gt & ~pred, HUMAN_ONLY)])
    fig = frame_figure(rgb, KNOT_LEGEND)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        fig.add_shape(
            type="rect",
            x0=x0 - BOX_MARGIN - 0.5,
            y0=y0 - BOX_MARGIN - 0.5,
            x1=x1 + BOX_MARGIN - 0.5,
            y1=y1 + BOX_MARGIN - 0.5,
            line={"color": "white", "width": 3},
        )
    write_png(fig, rgb.shape[0], path)


def frame_figure(rgb: np.ndarray, legend: List[Tuple[Tuple[int, int, int], str]]) -> go.Figure:
    size = rgb.shape[0]
    fig = go.Figure(go.Image(z=rgb, hoverinfo="skip"))
    fig.update_xaxes(visible=False, range=[-0.5, size - 0.5], constrain="domain")
    fig.update_yaxes(visible=False, range=[size - 0.5, -0.5], scaleanchor="x", constrain="domain")
    fig.update_layout(
        width=WIDTH_PX,
        height=WIDTH_PX + LEGEND_PX,
        margin={"l": 0, "r": 0, "t": 0, "b": LEGEND_PX},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": FONT, "size": FONT_PX, "color": "black"},
        showlegend=False,
    )
    add_legend(fig, legend)
    return fig


def add_legend(fig: go.Figure, legend: List[Tuple[Tuple[int, int, int], str]]) -> None:
    swatch = FONT_PX * 0.8
    gap = FONT_PX * 0.4
    x = FONT_PX * 0.5
    y_mid = -LEGEND_PX / 2 / WIDTH_PX
    for color, label in legend:
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x / WIDTH_PX,
            x1=(x + swatch) / WIDTH_PX,
            y0=y_mid - swatch / 2 / WIDTH_PX,
            y1=y_mid + swatch / 2 / WIDTH_PX,
            fillcolor="rgb(%d,%d,%d)" % color,
            line={"color": "black", "width": 1},
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=(x + swatch + gap) / WIDTH_PX,
            y=y_mid,
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
        )
        x += swatch + gap + len(label) * FONT_PX * 0.55 + FONT_PX * 1.2


def grey(image: np.ndarray) -> np.ndarray:
    plane = image[..., 0] if image.ndim == 3 else image
    return np.repeat(plane[..., None], 3, axis=2).astype(np.float32)


def overlay(rgb: np.ndarray, layers: List[Tuple[np.ndarray, Tuple[int, int, int]]]) -> np.ndarray:
    out = rgb.copy()
    for mask, color in layers:
        out[mask] = (1 - ALPHA) * out[mask] + ALPHA * np.array(color, np.float32)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def write_png(fig: go.Figure, size: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(path, format="png", width=WIDTH_PX, height=WIDTH_PX + LEGEND_PX, scale=1)
    with Image.open(path) as img:
        img.convert("RGB").save(path, dpi=(DPI, DPI))
