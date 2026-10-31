import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scripts.viz_pith_gallery import render

LOG = "10"
PREFIX = Path("data/kwp_ds_v3")


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 pith on held-out log 10: error statistics and gallery.")
    parser.add_argument("--gt_root", type=Path, default=Path("/mnt/D/datasets/ct_log/human_reviewed_20260929"))
    parser.add_argument("--img_root", type=Path, default=Path("/mnt/D/datasets/ct_log/m5_fp32_export_input"))
    parser.add_argument("--model_root", type=Path, default=Path("/mnt/D/datasets/ct_log/m5_fp32_export"))
    parser.add_argument(
        "--points",
        type=Path,
        default=Path(""),
        help="Optional logged points [{path, gt, pred}] instead of --model_root (default: --model_root).",
    )
    parser.add_argument("--n_random", type=int, default=24)
    parser.add_argument("--stats_out", type=Path, default=Path("article/data/pith_m5_log10.json"))
    parser.add_argument("--out", type=Path, default=Path("article/figures/pith_gallery_m5_log10"))
    args = parser.parse_args()

    if args.points.name:
        rows = json.loads(args.points.read_text())
        for row in rows:
            row["path"] = str(PREFIX / Path(row["path"]).relative_to(Path(row["path"]).parents[2]))
            row["dist"] = float(np.hypot(row["pred"][0] - row["gt"][0], row["pred"][1] - row["gt"][1]))
        source = str(args.points)
    else:
        rows = points(args.gt_root / LOG / "ann", json.loads((args.model_root / f"{LOG}.json").read_text())["pith"])
        source = str(args.model_root)
    dist = np.array([r["dist"] for r in rows])
    stats = {
        "log": LOG,
        "model": "M5",
        "source": source,
        "n_slices": len(rows),
        "mean_px": round(float(dist.mean()), 3),
        "exact_pct": round(100 * float((dist == 0).mean()), 1),
        "le_1px_pct": round(100 * float((dist <= 1).mean()), 1),
        "le_2px_pct": round(100 * float((dist <= 2).mean()), 1),
        "p90_px": round(float(np.percentile(dist, 90)), 2),
        "max_px": round(float(dist.max()), 2),
        "n_over_1px": int((dist > 1).sum()),
        "error": "Euclidean distance between the rounded predicted pixel and the human click",
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, indent=1))

    worst = sorted((r for r in rows if r["dist"] > 1), key=lambda r: -r["dist"])
    rest = [r for r in rows if r["dist"] <= 1]
    picks = [rest[i] for i in np.linspace(0, len(rest) - 1, args.n_random).astype(int)]
    figure = render(worst + sorted(picks, key=lambda r: r["path"]), args.img_root, len(worst))
    figure.update_layout(
        title="Log 10 pith, 41x41 px around the click (x8). Green square = human click, red cross = M5.<br>"
        "First %d: all slices with error > 1 px (1.41 px = diagonal neighbour); rest: evenly spaced sample."
        % len(worst),
        font_family="Arial",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(str(args.out) + ".png", width=1100, height=figure.layout.height, scale=1.5)
    print(json.dumps(stats))


def points(ann_dir: Path, pith: Dict[str, List[int]]) -> List[dict]:
    rows = []
    for ann in sorted(ann_dir.glob("*.json")):
        page = ann.name[: -len(".json")]
        for obj in json.loads(ann.read_text()).get("objects", []):
            if obj.get("classTitle", "").lower() == "pith":
                gt = [int(round(v)) for v in obj["points"]["exterior"][0]]
                pred = pith[page]
                rows.append(
                    {
                        "path": str(PREFIX / LOG / "img" / page),
                        "gt": gt,
                        "pred": pred,
                        "dist": float(np.hypot(pred[0] - gt[0], pred[1] - gt[1])),
                    }
                )
    return rows


if __name__ == "__main__":
    main()
