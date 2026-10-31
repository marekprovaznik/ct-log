import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List

import numpy as np
from PIL import Image
from scripts.eval_generalisation import predict_member
from scripts.export_knot_probs import load_stack
from scripts.export_report_preds import load_member
import torch

PAGE_RE = re.compile(r"page_(\d+)")
NATIVE = 778


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One model run per slice: 3-class argmax map, knot probability and pith point (build_kwp_dataset --model_root)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Model name recorded in META.json (e.g. M5).")
    parser.add_argument("--data_root", type=Path, required=True, help="Directory with <log>/img/page_*.tiff.")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true", help="Disable the config's bf16 autocast and run in float32.")
    args = parser.parse_args()

    device = torch.device(args.device)
    member = load_member("%s=%s" % (args.config, args.checkpoint), device)
    if member["pith_head"] is None:
        msg = "Model %s has no pith output" % args.checkpoint
        raise ValueError(msg)
    if args.fp32:
        member["config"].backbone_bf16 = False
        if hasattr(member["seg_head"], "bf16"):
            member["seg_head"].bf16 = False
    args.out_root.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for log in args.logs:
        points = export_log(args.data_root / log / "img", member, args.out_root / log, args.batch_size, device)
        (args.out_root / f"{log}.json").write_text(json.dumps({"pith": points}, indent=0))
        counts[log] = len(points)
        print("log %s: %d slices" % (log, len(points)), flush=True)

    meta = {
        "name": args.name,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "tta": "none",
        "precision": "fp32" if args.fp32 else ("bf16 autocast" if member["config"].backbone_bf16 else "fp32"),
        "input": "3-slice stack [z-1, z, z+1] clamped at the ends, native 778 -> model resolution bilinear, ImageNet normalisation",
        "outputs": {
            "<log>/page_XXX.cls.png": "uint8 argmax of the 3-class softmax at native 778 (0 background, 1 wood, 2 knot)",
            "<log>/page_XXX.png": "uint8 = round(p_knot * 255) at native 778 (same format as export_ensemble_probs)",
            "<log>.json": "pith point per slice: rint(soft-argmax * 778) clipped to [0, 777], no snap (same format as export_pith)",
        },
        "resize": "softmax interpolated bilinear to native 778 before argmax",
        "slices_per_log": counts,
    }
    (args.out_root / "META.json").write_text(json.dumps(meta, indent=2))
    print("wrote %s" % args.out_root)


def export_log(
    img_dir: Path, member: dict, out_dir: Path, batch_size: int, device: torch.device
) -> Dict[str, List[int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted((int(PAGE_RE.search(p.stem).group(1)), p) for p in img_dir.glob("*.tiff") if PAGE_RE.search(p.stem))
    paths = [path for _, path in pages]
    points: Dict[str, List[int]] = {}
    for start in range(0, len(paths), batch_size):
        positions = list(range(start, min(start + batch_size, len(paths))))
        stacks = torch.stack([load_stack(paths, position, member["config"].window) for position in positions])
        probs, pith = predict_member(member, stacks, device)
        for i, position in enumerate(positions):
            stem = paths[position].name.split(".")[0]
            Image.fromarray(probs[i].argmax(axis=0).astype(np.uint8)).save(out_dir / f"{stem}.cls.png")
            knot = np.round(np.clip(probs[i, 2], 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(knot).save(out_dir / f"{stem}.png")
            x, y = np.clip(np.rint(pith[i]), 0, NATIVE - 1).astype(int)
            points[paths[position].name] = [int(x), int(y)]
    return points


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
