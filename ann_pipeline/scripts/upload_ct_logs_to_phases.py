"""Upload image-only CT logs to Phase1 (blank) and Phase2 (anchor-free auto).

For logs that have NO human anchors. Phase1 gets the images with empty
annotations (a canvas for annotators). Phase2 gets anchor-free OBB-only
propagation knots + peel wood + YOLO pith.

Expects, per log:
  - images at /mnt/D/datasets/ct_log/375492_SM_2025/<name>/img/page_*.tiff
  - propagation npz at experiments/sm2025_subset4_propagate/out/result_ct<name>_obb_only.npz

Auth via $SUPERVISELY_TOKEN.

Example:
    set -a && source .env && set +a
    python -m ann_pipeline.scripts.upload_ct_logs_to_phases --logs 05 06 08 09 10
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from src.utils.mask import mask_to_base64
import supervisely as sly
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ann_pipeline.pith.detectors import snap_to_darkest
from ann_pipeline.utils.review_tag import REVIEW_TAG_META
from ann_pipeline.wood.detectors import threshold_peel

PHASE1_ID = 377327
PHASE2_ID = 377328
CLASS_IDS = {"Wood": 1, "Knot": 2, "Pith": 3}
SM_ROOT = "/mnt/D/datasets/ct_log/375492_SM_2025"
NPZ_DIR = "experiments/sm2025_subset4_propagate/out"
DEFAULT_YOLO_WEIGHTS = "ann_pipeline/out/knot_runs/yolo11n_obb_2cls_holdout_v3/weights/best.pt"

CLASS_REGISTRY = [
    {"title": "Knot", "description": "", "shape": "any", "color": "#FF9D00", "geometry_config": {}, "hotkey": ""},
    {"title": "Wood", "description": "", "shape": "any", "color": "#86FF00", "geometry_config": {}, "hotkey": ""},
    {"title": "Pith", "description": "", "shape": "point", "color": "#FF03D6", "geometry_config": {}, "hotkey": ""},
]


def _largest_cc_fill(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask.astype(np.uint8))
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return ndi.binary_fill_holes(lab == largest).astype(np.uint8)


def _bitmap_object(class_title: str, mask: np.ndarray) -> Optional[dict]:
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    patch = mask[y0 : y1 + 1, x0 : x1 + 1].astype(bool)
    b64 = mask_to_base64(torch.from_numpy(patch))
    return {
        "classTitle": class_title,
        "description": "",
        "tags": [],
        "bitmap": {"origin": [int(x0), int(y0)], "data": b64},
        "shape": "bitmap",
        "geometryType": "bitmap",
        "labelerLogin": "auto-pipeline",
    }


def _point_object(class_title: str, x: float, y: float) -> dict:
    return {
        "classTitle": class_title,
        "description": "",
        "tags": [],
        "points": {"exterior": [[int(round(x)), int(round(y))]], "interior": []},
        "shape": "point",
        "geometryType": "point",
        "labelerLogin": "auto-pipeline",
    }


def _load_gray_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        gray = arr[..., :3].mean(axis=-1).astype(np.uint8)
        rgb = arr[..., :3].astype(np.uint8)
    else:
        gray = arr.astype(np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)
    return gray, rgb


def _yolo_pith(
    model: YOLO, img_rgb: np.ndarray, conf: float = 0.10, snap_window: int = 5
) -> Optional[Tuple[float, float]]:
    """Highest-confidence pith (cls=1) from a 2-class OBB detector, snapped to the darkest pixel.

    With snap_window > 0 the box centre is moved to the darkest pixel in a snap_window x snap_window
    window and returned as an integer pixel index (what Supervisely points use); 0 returns the raw
    continuous box centre.
    """
    res = model.predict(img_rgb, conf=conf, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return None
    cls = res.obb.cls.cpu().numpy().astype(int)
    confs = res.obb.conf.cpu().numpy()
    xywhr = res.obb.xywhr.cpu().numpy()
    pith_idx = np.where(cls == 1)[0]
    if len(pith_idx) == 0:
        return None
    best = pith_idx[np.argmax(confs[pith_idx])]
    x, y = float(xywhr[best, 0]), float(xywhr[best, 1])
    if snap_window <= 0:
        return x, y
    sx, sy = snap_to_darkest(img_rgb[..., :3].mean(axis=-1), x, y, window=snap_window)
    return float(sx), float(sy)


def build_auto_annotation(gray: np.ndarray, rgb: np.ndarray, pred_frame: np.ndarray, yolo_model: YOLO) -> dict:
    h, w = gray.shape
    objects: List[dict] = []

    wood_peel = threshold_peel(gray, 30, 60, 5).astype(bool)
    prop_knot = pred_frame == CLASS_IDS["Knot"]
    wood_final = _largest_cc_fill(wood_peel | prop_knot).astype(bool)

    final_pith = _yolo_pith(yolo_model, rgb)

    lab, n_cc = ndi.label(prop_knot, structure=np.ones((3, 3), dtype=np.uint8))
    for k in range(1, n_cc + 1):
        clipped = (lab == k) & wood_final
        if clipped.sum() < 150:
            continue
        if final_pith is not None:
            px, py = int(round(final_pith[0])), int(round(final_pith[1]))
            inside = 0 <= py < h and 0 <= px < w and bool(clipped[py, px])
            ys, xs = np.nonzero(clipped)
            dist = float(np.hypot(xs.mean() - final_pith[0], ys.mean() - final_pith[1]))
            if inside or dist < 25.0:
                continue
        from skimage import measure as _measure

        props = _measure.regionprops(clipped.astype(np.uint8))
        if props and (props[0].eccentricity < 0.7 or props[0].solidity < 0.85):
            continue
        clipped = ndi.binary_fill_holes(clipped)
        obj = _bitmap_object("Knot", clipped.astype(np.uint8))
        if obj is not None:
            objects.append(obj)

    obj = _bitmap_object("Wood", wood_final.astype(np.uint8))
    if obj is not None:
        objects.append(obj)
    if final_pith is not None:
        objects.append(_point_object("Pith", *final_pith))

    return {"size": {"height": h, "width": w}, "description": "", "tags": [], "objects": objects}


def _meta(include_tag: bool) -> dict:
    return {
        "classes": CLASS_REGISTRY,
        "tags": [REVIEW_TAG_META] if include_tag else [],
        "projectType": "images",
        "projectSettings": {"multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False}},
    }


def _upload_dataset(api: sly.Api, project_id: int, ds_name: str, img_paths: List[Path], anns: List[dict]) -> int:
    existing = api.dataset.get_info_by_name(project_id, ds_name)
    if existing is not None:
        msg = "dataset %r already exists in project %d (id=%d); delete it first" % (ds_name, project_id, existing.id)
        raise ValueError(msg)
    ds = api.dataset.create(project_id, ds_name)
    names = [p.name for p in img_paths]
    img_infos = api.image.upload_paths(ds.id, names, [str(p) for p in img_paths])
    api.annotation.upload_jsons([i.id for i in img_infos], anns)
    return ds.id


def process_log(api: sly.Api, yolo_model: YOLO, name: str, phases: List[str]) -> None:
    img_dir = Path(SM_ROOT) / name / "img"
    npz_path = Path(NPZ_DIR) / ("result_ct%s_obb_only.npz" % name)
    if not img_dir.exists():
        msg = "image dir not found: %s" % img_dir
        raise ValueError(msg)
    if not npz_path.exists():
        msg = "propagation npz not found: %s" % npz_path
        raise ValueError(msg)

    img_paths = sorted(img_dir.glob("*.tiff"))
    data = np.load(npz_path, allow_pickle=False)
    page_to_idx = {int(p): i for i, p in enumerate(data["pages"])}
    pred = data["pred"]

    phase1_anns: List[dict] = []
    phase2_anns: List[dict] = []
    for img_path in tqdm(img_paths, desc="log %s" % name):
        gray, rgb = _load_gray_rgb(str(img_path))
        h, w = gray.shape
        phase1_anns.append({"size": {"height": h, "width": w}, "description": "", "tags": [], "objects": []})
        page_num = int(img_path.stem.replace("page_", ""))
        idx = page_to_idx.get(page_num)
        pred_frame = pred[idx] if idx is not None else np.zeros((h, w), dtype=np.uint8)
        phase2_anns.append(build_auto_annotation(gray, rgb, pred_frame, yolo_model))

    p1 = _upload_dataset(api, PHASE1_ID, name, img_paths, phase1_anns) if "1" in phases else None
    p2 = _upload_dataset(api, PHASE2_ID, name, img_paths, phase2_anns) if "2" in phases else None
    print("log %s: Phase1 ds=%s, Phase2 ds=%s (%d frames)" % (name, p1, p2, len(img_paths)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True, help="CT log names, e.g. 05 06 08 09 10")
    parser.add_argument("--yolo_weights", default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["1", "2"],
        choices=["1", "2"],
        help="Which projects to (re)upload to. Default both; use '2' to refresh only Phase2 auto annotations.",
    )
    parser.add_argument("--server", default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"))
    args = parser.parse_args()

    token = os.environ.get("SUPERVISELY_TOKEN")
    if not token:
        msg = "SUPERVISELY_TOKEN not set; source .env first"
        raise ValueError(msg)

    api = sly.Api(server_address=args.server, token=token)
    print("connected as %s" % api.user.get_my_info().login)
    if "1" in args.phases:
        api.project.update_meta(PHASE1_ID, _meta(include_tag=False))
    if "2" in args.phases:
        api.project.update_meta(PHASE2_ID, _meta(include_tag=True))

    yolo_model = YOLO(args.yolo_weights)
    for name in args.logs:
        process_log(api, yolo_model, name, args.phases)

    print("done.")


if __name__ == "__main__":
    main()
