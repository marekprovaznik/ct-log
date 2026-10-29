import argparse
import html
import json
import logging
import math
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from scripts.radial.common import FOUND_IOU, N_SLICES, OUT, page_name, read_png, write_png

logger = logging.getLogger(__name__)
YELLOW = (255, 220, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)
GREEN = (0, 220, 0)
RED = (255, 40, 40)
ALPHA = 0.55
AXIAL_MARGIN = 16
AXIAL_MIN = 80
AXIAL_HEIGHT = 240
RADIAL_MARGIN = 15
RADIAL_DZ = 10
RADIAL_SCALE = 2
N_CONTROLS = 6
GROUPS = [
    ("missed", "Thin GT missed by M4 (IoU < 0.3)"),
    ("poor", "Thin GT, poor (0.3 <= IoU < 0.5)"),
    ("good", "Thin GT, good (IoU >= 0.5)"),
    ("m4_thin_gt_thick", "Thin M4 component matched to a thick GT knot"),
    ("m4_extra", "Thin M4 component with no GT match (extra)"),
    ("thick", "Thick GT controls"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Axial vs radial thin-knot gallery for log 10.")
    parser.add_argument("--root", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = json.loads((args.root / "knots.json").read_text())
    planes = json.loads((args.root / "radial" / "planes.json").read_text())
    by_id = {k["id"]: k for k in data["knots"]}
    tiles = select(data["knots"], by_id)
    gt_pith = np.array(data["gt_pith_xy"])
    m4_pith = np.array(data["m4_pith_xy"])
    labelled = set(json.loads((args.root / "meta.json").read_text())["labelled_pages"])

    out_dir = args.root / "gallery"
    rows: List[Dict[str, object]] = []
    for group, knot in tiles:
        rows.append(
            render_tile(args.root, out_dir, group, knot, by_id, gt_pith, m4_pith, planes["radius_px"], labelled)
        )
    (out_dir / "tiles.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "index.html").write_text(index_html(rows, data["summary"]))
    logger.info("%d tiles -> %s", len(rows), out_dir / "index.html")


def select(knots: List[Dict[str, object]], by_id: Dict[str, Dict[str, object]]) -> List[Tuple[str, Dict[str, object]]]:
    tiles: List[Tuple[str, Dict[str, object]]] = []
    for k in knots:
        if k["source"] == "gt" and k["thin"]:
            group = "missed" if k["iou"] < FOUND_IOU else "poor" if k["iou"] < 0.5 else "good"
            tiles.append((group, k))
        elif k["source"] == "m4" and k["thin"]:
            if k["partner_id"] is None:
                tiles.append(("m4_extra", k))
            elif not by_id[k["partner_id"]]["thin"]:
                tiles.append(("m4_thin_gt_thick", by_id[k["partner_id"]]))
    thick = [k for k in knots if k["source"] == "gt" and not k["thin"] and 10 <= k["minor_axis_px"] <= 40]
    picks = np.linspace(0, len(thick) - 1, N_CONTROLS).round().astype(int)
    tiles.extend(("thick", thick[i]) for i in picks)
    order = [g for g, _ in GROUPS]
    return sorted(tiles, key=lambda t: (order.index(t[0]), t[1]["z"], t[1]["id"]))


def render_tile(
    root: pathlib.Path,
    out_dir: pathlib.Path,
    group: str,
    knot: Dict[str, object],
    by_id: Dict[str, Dict[str, object]],
    gt_pith: np.ndarray,
    m4_pith: np.ndarray,
    radius: int,
    labelled: set,
) -> Dict[str, object]:
    z = int(knot["z"])
    page = page_name(z)
    gray = read_png(root / "images" / ("%s.png" % page))
    gt_path = root / "annotations" / ("%s.png" % page)
    gt = read_png(gt_path) == 2 if gt_path.exists() else np.zeros_like(gray, bool)
    pred = read_png(root / "predictions" / "m4" / ("%s_mask.png" % page)) > 0
    ys, xs = axial_window(knot["bbox_xyxy"])
    markers = [(gt_pith[z], "square"), (m4_pith[z], "cross")]
    axial = strip(gray[ys, xs], gt[ys, xs], pred[ys, xs], [(p[0] - xs.start, p[1] - ys.start, m) for p, m in markers])
    scale = max(1, int(round(AXIAL_HEIGHT / axial.shape[0])))
    axial = np.kron(axial, np.ones((scale, scale, 1), np.uint8))

    plane_dir = root / "radial" / ("theta_%05.1f" % (round(float(knot["theta_deg"]), 1) % 360.0))
    c0 = max(0, radius + int(math.floor(knot["r_min_px"])) - RADIAL_MARGIN)
    c1 = min(2 * radius + 1, radius + int(math.ceil(knot["r_max_px"])) + RADIAL_MARGIN + 1)
    z0, z1 = max(0, z - RADIAL_DZ), min(N_SLICES - 1, z + RADIAL_DZ)
    r_ct = read_png(plane_dir / "ct_x10.png")[10 * z0 : 10 * z1 + 1, c0:c1]
    r_gt = read_png(plane_dir / "gt_x10.png")[10 * z0 : 10 * z1 + 1, c0:c1] == 2
    r_pred = read_png(plane_dir / "m4_mask_x10.png")[10 * z0 : 10 * z1 + 1, c0:c1] > 0
    theta = math.radians(float(knot["theta_deg"]))
    m4_offset = (m4_pith[z] - gt_pith[z]) @ np.array([math.cos(theta), math.sin(theta)])
    row = 10 * (z - z0)
    radial = strip(
        r_ct,
        r_gt,
        r_pred,
        [(radius - c0, row, "square"), (radius + m4_offset - c0, row, "cross")],
        tick_row=row,
    )
    radial = np.kron(radial, np.ones((RADIAL_SCALE, RADIAL_SCALE, 1), np.uint8))

    native_gt = read_png(plane_dir / "gt.png")[z0 : z1 + 1, c0:c1] == 2
    native_pred = read_png(plane_dir / "m4_mask.png")[z0 : z1 + 1, c0:c1] > 0
    rows_ok = np.array([page_name(zz) in labelled for zz in range(z0, z1 + 1)])
    radial_iou = iou(native_gt[rows_ok], native_pred[rows_ok])
    row_iou = iou(native_gt[z - z0], native_pred[z - z0])

    name = "%s__%s" % (group, knot["id"])
    write_png(out_dir / "tiles" / ("%s_axial.png" % name), axial)
    write_png(out_dir / "tiles" / ("%s_radial.png" % name), radial)
    partner = by_id.get(knot["partner_id"]) if knot["partner_id"] else None
    return {
        "group": group,
        "id": knot["id"],
        "page": page,
        "z": z,
        "minor_axis_px": knot["minor_axis_px"],
        "partner_minor_axis_px": partner["minor_axis_px"] if partner else None,
        "area": knot["area"],
        "theta_deg": knot["theta_deg"],
        "r_range_px": [knot["r_min_px"], knot["r_max_px"]],
        "axial_iou": knot["iou"],
        "radial_iou": radial_iou,
        "radial_row_iou": row_iou,
        "radial_rows": [z0, z1],
        "neighbours": knot["neighbours"],
        "axial_png": "tiles/%s_axial.png" % name,
        "radial_png": "tiles/%s_radial.png" % name,
    }


def axial_window(bbox: List[int]) -> Tuple[slice, slice]:
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0) + 2 * AXIAL_MARGIN
    side = max(side, AXIAL_MIN)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx = min(max(cx - side // 2, 0), 778 - side)
    sy = min(max(cy - side // 2, 0), 778 - side)
    return slice(sy, sy + side), slice(sx, sx + side)


def strip(
    gray: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    markers: List[Tuple[float, float, str]],
    tick_row: Optional[int] = None,
) -> np.ndarray:
    raw = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    gt_panel = raw.copy()
    edge = gt & ~erode(gt)
    gt_panel[edge] = CYAN
    pred_panel = raw.copy()
    for mask, colour in ((gt & pred, YELLOW), (pred & ~gt, MAGENTA), (gt & ~pred, CYAN)):
        pred_panel[mask] = (1 - ALPHA) * pred_panel[mask] + ALPHA * np.array(colour, np.float32)
    panels = []
    for panel in (raw, gt_panel, pred_panel):
        panel = panel.astype(np.uint8)
        for x, y, kind in markers:
            draw_marker(panel, x, y, kind)
        if tick_row is not None:
            panel[tick_row, :3] = (255, 255, 255)
            panel[tick_row, -3:] = (255, 255, 255)
        panels.append(panel)
    sep = np.full((gray.shape[0], 3, 3), 40, np.uint8)
    return np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)


def erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    return padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]


def draw_marker(panel: np.ndarray, x: float, y: float, kind: str) -> None:
    cx, cy = int(round(x)), int(round(y))
    h, w = panel.shape[:2]
    if kind == "square":
        pts = [(cx + dx, cy + dy) for dx in range(-2, 3) for dy in range(-2, 3) if max(abs(dx), abs(dy)) == 2]
        colour = GREEN
    else:
        pts = [(cx + d, cy + s * d) for d in range(-2, 3) for s in (-1, 1)]
        colour = RED
    for px, py in pts:
        if 0 <= px < w and 0 <= py < h:
            panel[py, px] = colour


def iou(gt: np.ndarray, pred: np.ndarray) -> Optional[float]:
    union = np.logical_or(gt, pred).sum()
    return round(float(np.logical_and(gt, pred).sum() / union), 4) if union else None


def index_html(rows: List[Dict[str, object]], summary: Dict[str, object]) -> str:
    parts = [HEAD % (summary["n_gt_thin"], summary["gt_thin_found"], summary["n_m4_thin"])]
    for group, title in GROUPS:
        items = [r for r in rows if r["group"] == group]
        if not items:
            continue
        parts.append('<h2 id="%s">%s <span class="n">(%d)</span></h2>' % (group, html.escape(title), len(items)))
        for r in items:
            parts.append(tile_html(r))
    parts.append("</body></html>")
    return "\n".join(parts)


def tile_html(r: Dict[str, object]) -> str:
    nb = r["neighbours"]

    def flag(v: Optional[bool]) -> str:
        return "–" if v is None else ("✓" if v else "✗")

    radial = "n/a" if r["radial_iou"] is None else "%.2f" % r["radial_iou"]
    row = "n/a" if r["radial_row_iou"] is None else "%.2f" % r["radial_row_iou"]
    partner = "" if r["partner_minor_axis_px"] is None else " (partner %.1f px)" % r["partner_minor_axis_px"]
    return (
        '<div class="tile"><div class="cap"><b>%s</b> · %s · z=%d · width %.1f px%s · θ %.1f° · r %.0f–%.0f px'
        " · axial IoU <b>%.2f</b> · radial IoU <b>%s</b> (rows %d–%d), row z only <b>%s</b>"
        "<br>z−1: GT %s M4 %s &nbsp; z+1: GT %s M4 %s</div>"
        '<div class="lab">axial: raw | GT | M4</div><img src="%s">'
        '<div class="lab">radial (knot-aligned, ×10 in z, row z ticked): raw | GT | M4</div><img src="%s"></div>'
    ) % (
        html.escape(r["id"]),
        r["page"],
        r["z"],
        r["minor_axis_px"],
        partner,
        r["theta_deg"],
        r["r_range_px"][0],
        r["r_range_px"][1],
        r["axial_iou"],
        radial,
        r["radial_rows"][0],
        r["radial_rows"][1],
        row,
        flag(nb["gt_z-1"]),
        flag(nb["m4_z-1"]),
        flag(nb["gt_z+1"]),
        flag(nb["m4_z+1"]),
        r["axial_png"],
        r["radial_png"],
    )


HEAD = (
    """<!doctype html><html><head><meta charset="utf-8"><title>Radial thin-knot gallery</title>
<style>
body{background:#111;color:#ddd;font:13px/1.4 system-ui,sans-serif;margin:16px}
h1{font-size:18px}h2{font-size:15px;margin-top:28px;border-bottom:1px solid #444}
.n{color:#888}.tile{display:inline-block;vertical-align:top;margin:8px;padding:8px;background:#1b1b1b;border:1px solid #333}
.cap{max-width:900px;margin-bottom:4px}.lab{color:#888;font-size:11px;margin-top:4px}
img{image-rendering:pixelated;display:block;max-width:100%%}
.legend span{padding:0 6px;margin-right:6px;color:#000}
</style></head><body>
<h1>Log 10: thin knots in axial vs knot-aligned radial views (M4 = two v4 U-Nets averaged)</h1>
<p>%d thin GT knots (minor axis &lt; 8 px), %d found by M4 (IoU ≥ 0.3); %d thin M4 components.
Radial planes are straightened reformats through the human pith of each slice, at the angle pith → knot centroid;
columns = distance from pith (1 px = 1 mm), rows = slices upsampled ×10 (nearest for labels) so the aspect is correct.
Radial IoU = GT vs M4 knot pixels inside the radial crop on native rows (unlabelled pages excluded); it includes other knots in the crop, e.g. the z±1 partner. Row z only = the same restricted to the knot's own slice, i.e. 1-D IoU along the ray. Qualitative only.</p>
<p class="legend"><span style="background:#ffdc00">GT ∩ M4</span><span style="background:#f0f">M4 only</span>
<span style="background:#0ff">GT only / GT outline</span><span style="background:#0d0">□ human pith</span>
<span style="background:#ff2828">× M4 pith</span></p>
<p>"""
    + " · ".join('<a href="#%s" style="color:#9cf">%s</a>' % (g, t) for g, t in GROUPS)
    + "</p>"
)


if __name__ == "__main__":
    main()
