import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from src.timber_gallery import IGNORE_INDEX, reassemble

S8 = np.ones((3, 3), dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sizes of matched, missed and false instances on reassembled strips.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--pred_dir", type=str, required=True, help="<strip>.png class maps from eval_timber strips mode."
    )
    parser.add_argument("--class_index", type=int, default=3)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_pred_area", type=int, default=16, help="Drop predicted components below this (px).")
    parser.add_argument("--px_per_mm", type=float, default=4.0)
    parser.add_argument("--small_mm2", type=float, default=20.0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    tiles = pd.read_csv(root / "tiles.csv", dtype={"tile_id": str, "strip": str})
    tiles = tiles[tiles["split"] == args.split]
    areas: Dict[str, List[float]] = {"matched_gt": [], "missed_gt": [], "false_pred": []}
    px_area = 1.0 / args.px_per_mm**2
    for strip, rows in tiles.groupby("strip", sort=True):
        _, mask = reassemble(root, rows)
        prediction = np.asarray(Image.open(Path(args.pred_dir) / f"{strip}.png"))
        strip_areas = match_instances(mask, prediction, args.class_index, args.iou_threshold, args.min_pred_area)
        for key, values in strip_areas.items():
            areas[key] += [v * px_area for v in values]
    summary = {
        "split": args.split,
        "pred_dir": args.pred_dir,
        "class_index": args.class_index,
        "iou_threshold": args.iou_threshold,
        "min_pred_area_px": args.min_pred_area,
        "px_per_mm": args.px_per_mm,
        "small_mm2": args.small_mm2,
        "matching": "a GT instance is matched if its best IoU with any predicted instance >= iou_threshold; a predicted instance is false if its best IoU with any GT instance < iou_threshold",
    }
    for key, values in areas.items():
        array = np.array(values)
        summary[key] = {
            "count": int(array.size),
            "median_mm2": float(np.median(array)) if array.size else None,
            "share_below_small": float((array < args.small_mm2).mean()) if array.size else None,
        }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def match_instances(
    mask: np.ndarray, prediction: np.ndarray, class_index: int, iou_threshold: float, min_pred_area: int
) -> Dict[str, List[float]]:
    """Areas (px) of matched and missed GT instances and of false predicted instances of one class.

    Args:
        mask: [H, W] GT class ids, IGNORE_INDEX where unsupervised.
        prediction: [H, W] predicted class ids.
        class_index: Evaluated class.
        iou_threshold: IoU for a match.
        min_pred_area: Predicted components below this many px are dropped.

    Returns:
        Dict[str, List[float]]: matched_gt, missed_gt and false_pred areas in px.
    """
    gt, n_gt = ndi.label(mask == class_index, structure=S8)
    pred, n_pred = ndi.label((prediction == class_index) & (mask != IGNORE_INDEX), structure=S8)
    pred_area = np.bincount(pred.ravel(), minlength=n_pred + 1)
    keep = pred_area >= min_pred_area
    keep[0] = False
    pred = np.where(keep[pred], pred, 0)
    pred_area = np.where(keep, pred_area, 0)
    gt_area = np.bincount(gt.ravel(), minlength=n_gt + 1)
    inter = np.zeros((n_gt + 1, n_pred + 1))
    np.add.at(inter, (gt.ravel(), pred.ravel()), 1)
    iou = inter[1:, 1:] / (gt_area[1:, None] + pred_area[None, 1:] - inter[1:, 1:] + 1e-9)
    gt_best = iou.max(axis=1) if n_pred else np.zeros(n_gt)
    pred_best = iou.max(axis=0) if n_gt else np.zeros(n_pred)
    out: Dict[str, List[float]] = {"matched_gt": [], "missed_gt": [], "false_pred": []}
    for area, best in zip(gt_area[1:], gt_best):
        out["matched_gt" if best >= iou_threshold else "missed_gt"].append(float(area))
    for area, best, kept in zip(pred_area[1:], pred_best, keep[1:]):
        if kept and best < iou_threshold:
            out["false_pred"].append(float(area))
    return out


if __name__ == "__main__":
    main()
