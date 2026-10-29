import argparse
import json
import logging
import math
import pathlib
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi
from scripts.radial.common import (
    FOUND_IOU,
    N_SLICES,
    OUT,
    THIN_PX,
    components,
    interpolate_pith,
    match,
    minor_axis,
    page_name,
    read_png,
)

logger = logging.getLogger(__name__)
NEIGHBOUR_DILATE_PX = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="List GT and M4 knot instances of log 10 with thinness and matching.")
    parser.add_argument("--root", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gt_vol, pred_vol, labelled = load_volumes(args.root)
    human = json.loads((args.root / "annotations" / "pith.json").read_text())["pith"]
    model = json.loads((args.root / "predictions" / "m4" / "pith.json").read_text())["pith"]
    gt_pith = interpolate_pith({int(p[5:]): tuple(xy) for p, xy in human.items()}, N_SLICES)
    m4_pith = np.array([model[page_name(z)] for z in range(N_SLICES)], np.float64)

    records: List[Dict[str, object]] = []
    for z in range(N_SLICES):
        if not labelled[z]:
            continue
        gt = components(gt_vol[z])
        pred = components(pred_vol[z])
        gt_iou, gt_to_pred = match(gt, pred)
        pred_to_gt = {int(j): i for i, j in enumerate(gt_to_pred) if j >= 0}
        for i, mask in enumerate(gt):
            j = int(gt_to_pred[i])
            records.append(
                describe(
                    mask, "gt", z, i, gt_pith[z], gt_vol, pred_vol, labelled, float(gt_iou[i]), j if j >= 0 else None
                )
            )
        for j, mask in enumerate(pred):
            i = pred_to_gt.get(j)
            iou = float(gt_iou[i]) if i is not None else 0.0
            records.append(describe(mask, "m4", z, j, gt_pith[z], gt_vol, pred_vol, labelled, iou, i))
    summary = summarise(records)
    out = {
        "definitions": {
            "id": "<source>_<page>_<index>; index = order of scipy.ndimage.label (8-connected) in that slice",
            "theta_deg": "atan2(dy, dx) from the human pith (interpolated where unlabelled) to the centroid, image axes (0 = +x right, 90 = +y down)",
            "r_px": "distance pith -> centroid; r_min/r_max = extent of the knot pixels' distance to pith",
            "iou": "Hungarian matching per slice on IoU; unmatched = 0",
            "neighbours": "z +- 1: any GT / M4 knot pixel within the knot footprint dilated by %d px; null = slice unlabelled or out of range"
            % NEIGHBOUR_DILATE_PX,
        },
        "summary": summary,
        "gt_pith_xy": gt_pith.tolist(),
        "m4_pith_xy": m4_pith.tolist(),
        "knots": records,
    }
    (args.root / "knots.json").write_text(json.dumps(out, indent=1))
    logger.info(json.dumps(summary, indent=1))


def load_volumes(root: pathlib.Path):
    gt = np.zeros((N_SLICES, 778, 778), bool)
    pred = np.zeros_like(gt)
    labelled = np.zeros(N_SLICES, bool)
    for z in range(N_SLICES):
        ann = root / "annotations" / ("%s.png" % page_name(z))
        if ann.exists():
            gt[z] = read_png(ann) == 2
            labelled[z] = True
        pred[z] = read_png(root / "predictions" / "m4" / ("%s_mask.png" % page_name(z))) > 0
    return gt, pred, labelled


def describe(
    mask: np.ndarray,
    source: str,
    z: int,
    index: int,
    pith_xy: np.ndarray,
    gt_vol: np.ndarray,
    pred_vol: np.ndarray,
    labelled: np.ndarray,
    iou: float,
    partner: Optional[int],
) -> Dict[str, object]:
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    dist = np.hypot(xs - pith_xy[0], ys - pith_xy[1])
    width = minor_axis(mask)
    foot = ndi.binary_dilation(mask, iterations=NEIGHBOUR_DILATE_PX)
    neighbours: Dict[str, Optional[bool]] = {}
    for dz in (-1, 1):
        zz = z + dz
        inside = 0 <= zz < N_SLICES
        neighbours["gt_z%+d" % dz] = bool(gt_vol[zz][foot].any()) if inside and labelled[zz] else None
        neighbours["m4_z%+d" % dz] = bool(pred_vol[zz][foot].any()) if inside else None
    other = "m4" if source == "gt" else "gt"
    return {
        "id": "%s_%s_%02d" % (source, page_name(z), index),
        "source": source,
        "page": page_name(z),
        "z": z,
        "area": int(mask.sum()),
        "centroid_xy": [round(cx, 2), round(cy, 2)],
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "minor_axis_px": round(width, 3),
        "thin": width < THIN_PX,
        "theta_deg": round(math.degrees(math.atan2(cy - pith_xy[1], cx - pith_xy[0])) % 360.0, 2),
        "r_px": round(float(np.hypot(cx - pith_xy[0], cy - pith_xy[1])), 2),
        "r_min_px": round(float(dist.min()), 2),
        "r_max_px": round(float(dist.max()), 2),
        "iou": round(iou, 4),
        "partner_id": "%s_%s_%02d" % (other, page_name(z), partner) if partner is not None else None,
        "neighbours": neighbours,
    }


def summarise(records: List[Dict[str, object]]) -> Dict[str, object]:
    gt = [r for r in records if r["source"] == "gt"]
    thin = [r for r in gt if r["thin"]]
    pred_thin = [r for r in records if r["source"] == "m4" and r["thin"]]
    return {
        "n_gt_knots": len(gt),
        "gt_mean_per_knot_iou": round(float(np.mean([r["iou"] for r in gt])), 4),
        "n_gt_thin": len(thin),
        "gt_thin_found": sum(r["iou"] >= FOUND_IOU for r in thin),
        "gt_thin_iou0": sum(r["iou"] == 0 for r in thin),
        "gt_thin_mean_iou": round(float(np.mean([r["iou"] for r in thin])), 4),
        "n_m4_knots": sum(r["source"] == "m4" for r in records),
        "n_m4_thin": len(pred_thin),
        "n_m4_thin_unmatched": sum(r["partner_id"] is None for r in pred_thin),
    }


if __name__ == "__main__":
    main()
