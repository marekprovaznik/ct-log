import argparse
import json
import logging
import pathlib
import shutil
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scripts.radial.common import (
    N_SLICES,
    OUT,
    SRC_LOG,
    SRC_PRED,
    SRC_PROB,
    page_name,
    read_annotation,
    read_pred_npz,
    sha256,
    write_png,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the radial-experiment data layout for log 10.")
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for sub in ("images", "annotations", "predictions/m4", "predictions/m5", "radial", "gallery"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    human_pith, labelled, unlabelled = export_axial(args.out)
    m4_pith = export_m4(args.out)
    convention = check_pith_convention(human_pith, m4_pith)
    (args.out / "annotations" / "pith.json").write_text(
        json.dumps({"convention": "[x, y] = [column, row]", "unlabelled": unlabelled, "pith": human_pith}, indent=1)
    )
    (args.out / "predictions" / "m4" / "pith.json").write_text(
        json.dumps({"convention": "[x, y] = [column, row]", "pith": m4_pith}, indent=1)
    )
    write_meta(args.out, labelled, unlabelled, convention)
    logger.info("%d slices, %d labelled, unlabelled %s", N_SLICES, len(labelled), unlabelled)
    logger.info("pith convention: %s", convention)


def export_axial(out: pathlib.Path) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
    human_pith: Dict[str, List[float]] = {}
    labelled: List[str] = []
    unlabelled: List[str] = []
    for z in range(N_SLICES):
        page = page_name(z)
        rgb = np.array(Image.open(SRC_LOG / "img" / ("%s.tiff" % page)))
        if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
            msg = "R != G != B in %s" % page
            raise ValueError(msg)
        write_png(out / "images" / ("%s.png" % page), rgb[..., 0])
        ann_path = SRC_LOG / "ann" / ("%s.tiff.json" % page)
        if not ann_path.exists():
            unlabelled.append(page)
            continue
        label, pith = read_annotation(ann_path)
        write_png(out / "annotations" / ("%s.png" % page), label)
        labelled.append(page)
        if pith is not None:
            human_pith[page] = [pith[0], pith[1]]
    return human_pith, labelled, unlabelled


def export_m4(out: pathlib.Path) -> Dict[str, List[int]]:
    pith: Dict[str, List[int]] = {}
    for z in range(N_SLICES):
        page = page_name(z)
        shutil.copyfile(SRC_PROB / ("%s.png" % page), out / "predictions" / "m4" / ("%s_prob.png" % page))
        knot, _, raw = read_pred_npz(SRC_PRED / ("%s.tiff.npz" % page))
        write_png(out / "predictions" / "m4" / ("%s_mask.png" % page), (knot * 255).astype(np.uint8))
        pith[page] = [raw[0], raw[1]]
    return pith


def check_pith_convention(human: Dict[str, List[float]], model: Dict[str, List[int]]) -> Dict[str, float]:
    pages = sorted(set(human) & set(model))
    h = np.array([human[p] for p in pages])
    m = np.array([model[p] for p in pages], np.float64)
    as_xy = np.hypot(*(h - m).T)
    as_rc = np.hypot(*(h - m[:, ::-1]).T)
    if np.median(as_xy) > np.median(as_rc):
        msg = "pred_m8 pith looks like (row, col), not (x, y)"
        raise ValueError(msg)
    return {
        "npz_pith_order": "x, y",
        "n_pages": len(pages),
        "median_err_px_as_xy": float(np.median(as_xy)),
        "mean_err_px_as_xy": float(as_xy.mean()),
        "median_err_px_as_row_col": float(np.median(as_rc)),
    }


def write_meta(out: pathlib.Path, labelled: List[str], unlabelled: List[str], convention: Dict[str, float]) -> None:
    sources = {
        "ct_tiff_dir": str(SRC_LOG / "img"),
        "ann_json_dir": str(SRC_LOG / "ann"),
        "m4_prob_dir": str(SRC_PROB),
        "m4_prob_meta": str(SRC_PROB.parent / "META.json"),
        "m4_pred_npz_dir": str(SRC_PRED),
        "m4_pred_meta": str(SRC_PRED / "META.json"),
    }
    hashes: Dict[str, Dict[str, str]] = {"ct": {}, "ann": {}, "m4_prob": {}, "m4_npz": {}}
    for z in range(N_SLICES):
        page = page_name(z)
        hashes["ct"][page] = sha256(SRC_LOG / "img" / ("%s.tiff" % page))
        hashes["m4_prob"][page] = sha256(SRC_PROB / ("%s.png" % page))
        hashes["m4_npz"][page] = sha256(SRC_PRED / ("%s.tiff.npz" % page))
        ann = SRC_LOG / "ann" / ("%s.tiff.json" % page)
        if ann.exists():
            hashes["ann"][page] = sha256(ann)
    hashes["meta"] = {
        "m4_prob_meta": sha256(SRC_PROB.parent / "META.json"),
        "m4_pred_meta": sha256(SRC_PRED / "META.json"),
    }
    meta = {
        "log": "10",
        "n_slices": N_SLICES,
        "image_size_px": [778, 778],
        "spacing_mm": {"x": 1.0, "y": 1.0, "z": 10.0},
        "pages": [page_name(z) for z in range(N_SLICES)],
        "labelled_pages": labelled,
        "unlabelled_pages": unlabelled,
        "sources": sources,
        "sha256": hashes,
        "conventions": {
            "image": "row = y, column = x; z = slice index = page number",
            "points": "[x, y] = [column, row] in pixels",
            "annotation_png": "0 background, 1 wood, 2 knot (knot painted over wood)",
            "m4_prob_png": "uint8 = round(p * 255), mean knot softmax of the two v4 U-Nets",
            "m4_mask_png": "0/255, mean prob >= 0.5 clipped to predicted wood (from pred_m8 npz)",
            "m4_pith": convention,
            "knot_instance": "8-connected component per axial slice",
            "thin_knot": "minor axis = 4 * sqrt(smallest eigenvalue of pixel-coordinate covariance) < 8 px",
            "colours": "yellow GT&pred, magenta pred only, cyan GT only, green square human pith, red x pred pith",
        },
        "models": {"m4": "two v4 U-Nets averaged (pred_m8)", "m5": "not exported yet"},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
