import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from article.render import save_frame
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import measure
from src.dataset.kwp_mask import KwpMaskBuilder
from src.eval.knot_instances import S8, THIN_MINOR_AXIS, iou_matrix

HELDOUT_LOG = "10"
FILL_BIN = "0.3"
TOLERANCE = 0.01
TRAINING_WARNING = "Training log %s was used to train M5: this image illustrates the metric, not a test result."


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-frame per-knot IoU gallery of M5 (GT vs M5 overlay).")
    parser.add_argument(
        "--index", type=Path, default=Path("/home/mary/dev/ct-log/report_2026-10/site/img/iou/index.json")
    )
    parser.add_argument("--gt_root", type=Path, default=Path("/mnt/D/datasets/ct_log/human_reviewed_20260929"))
    parser.add_argument("--img_root", type=Path, default=Path("/mnt/D/datasets/ct_log/m5_fp32_export_input"))
    parser.add_argument("--model_root", type=Path, default=Path("/mnt/D/datasets/ct_log/m5_fp32_export"))
    parser.add_argument("--out", type=Path, default=Path("article/figures/iou_gallery"))
    args = parser.parse_args()

    examples = json.loads(args.index.read_text())
    gallery = Gallery(args.gt_root, args.img_root, args.model_root)
    rows: Dict[str, List[dict]] = {"heldout_log10": [], "training_logs": []}
    for example in examples:
        knot = gallery.find(example["log"], example["page"], example["gt_area"])
        folder = "heldout_log10" if example["log"] == HELDOUT_LOG else "training_logs"
        rows[folder].append(
            gallery.render(example["file"], example["log"], example["page"], knot, args.out / folder, example)
        )
    if not any(r["bin"] == FILL_BIN for r in rows["heldout_log10"]):
        log, page, knot = gallery.nearest(HELDOUT_LOG, float(FILL_BIN))
        name = "iou_%s_%s_1.png" % (FILL_BIN, "thin" if knot["thin"] else "thick")
        rows["heldout_log10"].append(gallery.render(name, log, page, knot, args.out / "heldout_log10", None))
    for folder, items in rows.items():
        items.sort(key=lambda r: (float(r["bin"]), r["file"]))
        (args.out / folder / "index.json").write_text(json.dumps(items, indent=1))
        print("%s: %d examples" % (folder, len(items)))
    off = [r for items in rows.values() for r in items if r["index_iou"] is not None and not r["within_tolerance"]]
    print("recomputed IoU outside +-%.2f of index.json: %d" % (TOLERANCE, len(off)))
    for r in off:
        print("  %s %s/%s recomputed %.3f index %.3f" % (r["file"], r["log"], r["page"], r["iou"], r["index_iou"]))


class Gallery:
    def __init__(self, gt_root: Path, img_root: Path, model_root: Path) -> None:
        self.gt_root = gt_root
        self.img_root = img_root
        self.model_root = model_root
        self.builder = KwpMaskBuilder()
        self.knot_id = KwpMaskBuilder.class_names().index("knot")

    def slice(self, log: str, page: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stem = page.split(".")[0]
        annotation = json.loads((self.gt_root / log / "ann" / f"{stem}.tiff.json").read_text())
        gt = self.builder.build(annotation).numpy() == self.knot_id
        pred = np.array(Image.open(self.model_root / log / f"{stem}.cls.png")) == 2
        image = np.array(Image.open(self.img_root / log / "img" / f"{stem}.tiff"))
        return image, gt, pred

    def knots(self, log: str, page: str) -> List[dict]:
        _, gt, pred = self.slice(log, page)
        gt_labels, n_gt = ndi.label(gt, structure=S8)
        pred_labels, n_pred = ndi.label(pred, structure=S8)
        iou = iou_matrix(gt_labels, n_gt, pred_labels, n_pred)
        best = np.zeros(n_gt)
        if n_gt and n_pred:
            rows, cols = linear_sum_assignment(-iou)
            best[rows] = iou[rows, cols]
        out = []
        for label, region in enumerate(measure_regions(gt_labels), start=1):
            out.append(
                {
                    "label": label,
                    "iou": float(best[label - 1]),
                    "gt_area": int(region["area"]),
                    "width_px": round(float(region["minor"]), 1),
                    "thin": bool(region["minor"] < THIN_MINOR_AXIS),
                    "bbox": region["bbox"],
                }
            )
        return out

    def find(self, log: str, page: str, gt_area: int) -> dict:
        candidates = [k for k in self.knots(log, page) if k["gt_area"] == gt_area]
        if len(candidates) != 1:
            msg = "log %s %s: %d GT knots with area %d" % (log, page, len(candidates), gt_area)
            raise ValueError(msg)
        return candidates[0]

    def nearest(self, log: str, target: float) -> Tuple[str, str, dict]:
        best: Optional[Tuple[float, str, str, dict]] = None
        for ann in sorted((self.gt_root / log / "ann").glob("*.json")):
            page = ann.name[: -len(".json")]
            for knot in self.knots(log, page):
                key = abs(knot["iou"] - target)
                if best is None or key < best[0]:
                    best = (key, log, page, knot)
        if best is None:
            msg = "log %s has no GT knots" % log
            raise ValueError(msg)
        return best[1], best[2], best[3]

    def render(self, name: str, log: str, page: str, knot: dict, out_dir: Path, example: Optional[dict]) -> dict:
        image, gt, pred = self.slice(log, page)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = name[: -len(".png")]
        bin_ = stem.split("_")[1]
        save_frame(image, gt, pred, None, out_dir / f"{stem}.png")
        save_frame(image, gt, pred, knot["bbox"], out_dir / f"{stem}_box.png")
        index_iou = example["iou"] if example is not None else None
        return {
            "file": f"{stem}.png",
            "file_box": f"{stem}_box.png",
            "bin": bin_,
            "log": log,
            "page": page,
            "iou": round(knot["iou"], 3),
            "index_iou": index_iou,
            "within_tolerance": None if index_iou is None else bool(abs(knot["iou"] - index_iou) <= TOLERANCE),
            "thin": knot["thin"],
            "width_px": knot["width_px"],
            "gt_area": knot["gt_area"],
            "bbox_xyxy": knot["bbox"],
            "source": "meeting page gallery" if example is not None else "nearest log-10 knot to IoU %s" % bin_,
            "caption": caption(log, page, knot),
        }


def measure_regions(labels: np.ndarray) -> List[dict]:
    return [
        {"area": r.area, "minor": r.axis_minor_length, "bbox": [r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]]}
        for r in measure.regionprops(labels)
    ]


def caption(log: str, page: str, knot: dict) -> str:
    width = "thin" if knot["thin"] else "thick"
    slice_no = int(page.split(".")[0].split("_")[1])
    where = "Held-out log 10" if log == HELDOUT_LOG else "Training log %s" % log
    text = (
        "Fig. X: %s, slice %d. M5 knot prediction vs. human annotation, full 778 x 778 px slice; "
        "per-knot IoU %.2f for the %s knot (%.1f px wide) marked by the white rectangle in the _box variant. "
        "Yellow: overlap; magenta: M5 only; cyan: human only."
    ) % (where, slice_no, knot["iou"], width, knot["width_px"])
    if log != HELDOUT_LOG:
        text += " " + TRAINING_WARNING % log
    return text


if __name__ == "__main__":
    main()
