import argparse
import json
import logging
import pathlib
from typing import Dict, List

import numpy as np
from scipy import ndimage as ndi
from scripts.radial.common import (
    FOUND_IOU,
    N_SLICES,
    OUT,
    SRC_PRED,
    components,
    match,
    minor_axis,
    page_name,
    read_png,
    read_pred_npz,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z-consistency post-step: lower the knot threshold where M4 has a knot in z +- 1 at the same place."
    )
    parser.add_argument("--root", type=pathlib.Path, default=OUT)
    parser.add_argument("--t_low", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--dilate", type=int, nargs="+", default=[3, 6])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    prob, base, wood, gt, labelled, pith = load(args.root)
    results = [evaluate("baseline (p >= 0.5)", base, gt, labelled)]
    for d in args.dilate:
        support = neighbour_support(base, pith, d)
        for t in args.t_low:
            mask = base | ((prob >= t) & support & wood)
            results.append(evaluate("t_low %.2f, dilate %d px" % (t, d), mask, gt, labelled))
    for r in results:
        logger.info(json.dumps(r))
    (args.root / "z_consistency.json").write_text(json.dumps(results, indent=1))


def load(root: pathlib.Path):
    shape = (N_SLICES, 778, 778)
    prob = np.zeros(shape, np.float32)
    base = np.zeros(shape, bool)
    wood = np.zeros(shape, bool)
    gt = np.zeros(shape, bool)
    labelled = np.zeros(N_SLICES, bool)
    for z in range(N_SLICES):
        page = page_name(z)
        prob[z] = read_png(root / "predictions" / "m4" / ("%s_prob.png" % page)) / 255.0
        knot, wood[z], _ = read_pred_npz(SRC_PRED / ("%s.tiff.npz" % page))
        base[z] = knot
        ann = root / "annotations" / ("%s.png" % page)
        if ann.exists():
            gt[z] = read_png(ann) == 2
            labelled[z] = True
    model = json.loads((root / "predictions" / "m4" / "pith.json").read_text())["pith"]
    pith = np.array([model[page_name(z)] for z in range(N_SLICES)], np.float64)
    return prob, base, wood, gt, labelled, pith


def neighbour_support(base: np.ndarray, pith: np.ndarray, dilate: int) -> np.ndarray:
    support = np.zeros_like(base)
    for z in range(N_SLICES):
        for zz in (z - 1, z + 1):
            if not 0 <= zz < N_SLICES or not base[zz].any():
                continue
            dx, dy = np.rint(pith[z] - pith[zz]).astype(int)
            support[z] |= np.roll(base[zz], (dy, dx), axis=(0, 1))
        support[z] = ndi.binary_dilation(support[z], iterations=dilate) & ~base[z]
    return support


def evaluate(name: str, mask: np.ndarray, gt: np.ndarray, labelled: np.ndarray) -> Dict[str, object]:
    ious: List[float] = []
    thin_ious: List[float] = []
    n_pred = 0
    n_pred_unmatched = 0
    for z in np.nonzero(labelled)[0]:
        g = components(gt[z])
        p = components(mask[z])
        iou, gt_to_pred = match(g, p)
        ious.extend(iou)
        thin_ious.extend(iou[i] for i, c in enumerate(g) if minor_axis(c) < 8.0)
        n_pred += len(p)
        n_pred_unmatched += len(p) - int((gt_to_pred >= 0).sum())
    m = mask[labelled]
    t = gt[labelled]
    inter = float((m & t).sum())
    return {
        "name": name,
        "per_knot_iou": round(float(np.mean(ious)), 4),
        "thin_found": int(sum(i >= FOUND_IOU for i in thin_ious)),
        "n_thin": len(thin_ious),
        "thin_mean_iou": round(float(np.mean(thin_ious)), 4),
        "pixel_iou": round(inter / float((m | t).sum()), 4),
        "pixel_precision": round(inter / float(m.sum()), 4),
        "pixel_recall": round(inter / float(t.sum()), 4),
        "n_pred_components": n_pred,
        "n_pred_unmatched": n_pred_unmatched,
    }


if __name__ == "__main__":
    main()
