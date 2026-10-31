import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional

SOURCES = {
    "instance_metrics.json": "/home/mary/dev/ct-log/report_2026-10/scores/instance_metrics.json",
    "generalisation.json": "/home/mary/dev/ct-log/report_2026-10/scores/generalisation.json",
    "nnunet.json": "/home/mary/dev/ct-log/report_2026-10/scores/nnunet.json",
    "ensemble_dec512.json": "/home/mary/dev/ct-log/report_2026-10/scores/ensemble_dec512.json",
    "ens_dec512.json": "/home/mary/dev/ct-log/report_2026-10/euler_exports/ensemble_dec512/ens_dec512.json",
    "pith_wood_dec512.json": "/home/mary/dev/ct-log/report_2026-10/euler_exports/ensemble_dec512/pith_wood_dec512.json",
}
MILESTONES = [
    ("PL1", "May", "MedSAM2 propagation from human anchor frames"),
    ("PL2", "May", "YOLO-OBB detector + MedSAM2, slice by slice"),
    ("PL3", "Jul-Sep", "Anchor-free OBB propagation"),
    ("M1", "20-25 Sep", "DINOv3 ViT-L, frozen, 320 px"),
    ("PL4", "30 Sep", "YOLO11n-seg 640 px + snapped OBB pith"),
    ("M2", "2-5 Oct", "DINOv3 ViT-L, 784 px, fine-tuned"),
    ("PL5", "7 Oct", "YOLO11n-seg 800 px + snapped OBB pith"),
    ("M3", "7 Oct", "U-Net ResNet-50, 800 px"),
    ("M4", "9 Oct", "Two U-Nets averaged (EfficientNetV2-S + ConvNeXt-S)"),
    ("nnU-Net 2D", "11-12 Oct", "nnU-Net v2 2D, external baseline (no pith output)"),
    ("M5", "12-13 Oct", "One U-Net, EfficientNetV2-S, wide decoder, 48 epochs (default)"),
]
KEYS = [
    "knot_iou",
    "mean_per_gt_iou",
    "map50",
    "map50_95",
    "inst_f1_30",
    "precision",
    "recall",
    "thin_recall",
    "pith_mean_px_rounded",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model development PL1 -> M5 on held-out log 10 from the logged JSONs."
    )
    parser.add_argument("--sources_out", type=Path, default=Path("article/data/sources"))
    parser.add_argument("--out", type=Path, default=Path("article/data/development.json"))
    args = parser.parse_args()

    args.sources_out.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, path in SOURCES.items():
        shutil.copyfile(path, args.sources_out / name)
        data[name] = json.loads(Path(path).read_text())
    rows = [row(mid, when, what, data) for mid, when, what in MILESTONES]
    args.out.write_text(json.dumps({"log": "10", "keys": KEYS, "rows": rows}, indent=1))
    print(markdown(rows))


def row(mid: str, when: str, what: str, data: Dict[str, dict]) -> dict:
    values: Dict[str, Optional[float]] = dict.fromkeys(KEYS)
    sources: Dict[str, str] = {}
    inst = data["instance_metrics.json"].get(mid)
    if inst is not None:
        block = inst["iou_0.3"]
        values.update(
            mean_per_gt_iou=block["mean_per_gt_iou"],
            precision=block["precision"],
            recall=block["recall"],
            thin_recall=inst["recall_thin_iou0.3"]["recall"],
            inst_f1_30=block["f1"],
        )
        for key in ("mean_per_gt_iou", "inst_f1_30", "precision", "recall", "thin_recall"):
            sources[key] = "instance_metrics.json %s" % mid
    gen = data["generalisation.json"].get(mid, {}).get("splits", {})
    test = gen.get("test") or gen.get("val=test")
    if test is not None:
        split = "test" if "test" in gen and gen["test"] is not None else "val=test"
        for key in ("knot_iou", "inst_f1_30", "map50", "map50_95", "pith_mean_px_rounded"):
            if test.get(key) is not None:
                values[key] = test[key]
                sources[key] = "generalisation.json %s.splits.%s" % (mid, split)
    if mid == "nnU-Net 2D":
        nn = data["nnunet.json"]["2d"]["10"]
        for key in ("knot_iou", "inst_f1_30", "map50", "map50_95", "thin_recall"):
            values[key] = nn[key]
            sources[key] = "nnunet.json 2d.10"
        values["mean_per_gt_iou"] = data["nnunet.json"]["mean_per_gt_iou_log10"]["2d"]
        sources["mean_per_gt_iou"] = "nnunet.json mean_per_gt_iou_log10.2d"
    if mid == "M5":
        m5 = data["ens_dec512.json"]["log10"]["e1_ep48"]
        for key in ("knot_iou", "mean_per_gt_iou", "inst_f1_30", "map50", "map50_95", "thin_recall", "pith_mean_px_rounded"):
            values[key] = m5[key]
            sources[key] = "ens_dec512.json log10.e1_ep48"
    return {"id": mid, "when": when, "what": what, "values": values, "sources": sources}


def markdown(rows: List[dict]) -> str:
    lines = ["| ID | " + " | ".join(KEYS) + " |", "|---|" + "---:|" * len(KEYS)]
    for r in rows:
        cells = ["" if r["values"][k] is None else "%.3f" % r["values"][k] for k in KEYS]
        lines.append("| %s | %s |" % (r["id"], " | ".join(cells)))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
