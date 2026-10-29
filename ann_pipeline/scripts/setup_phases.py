"""Create Phase1 and Phase2 Supervisely projects for the two-phase annotation workflow.

Phase1: sparse human annotations (anchors). All frames uploaded, only anchors
        have annotations. Annotators add more anchors over time.
Phase2: auto-generated annotations for human review. Anchor frames are copied
        verbatim from Phase1 and tagged 'bumaska'. Non-anchor frames get
        OBB-augmented propagation output. Annotators correct auto annotations
        and tag corrected frames with 'bumaska'.

Usage (dry-run):
    set -a && source .env && set +a
    conda run -n ct-log python -m ann_pipeline.scripts.setup_phases \
        --subset 4 --out_dir /tmp/phases_test --dry_run

Upload:
    conda run -n ct-log python -m ann_pipeline.scripts.setup_phases \
        --subset 4 --out_dir /tmp/phases_test --upload
"""

import argparse
import json
import os
import pathlib
import shutil
from typing import List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage as ndi
from src.utils.mask import mask_to_base64
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ann_pipeline.pith.detectors import snap_to_darkest
from ann_pipeline.utils.review_tag import HUMAN_TAG, REVIEW_TAG_META
from ann_pipeline.wood.detectors import threshold_largest_cc

CLASS_REGISTRY = [
    {"title": "Knot", "shape": "any", "color": "#FF9D00"},
    {"title": "Wood", "shape": "any", "color": "#86FF00"},
    {"title": "Pith", "shape": "point", "color": "#FF03D6"},
]

CLASS_IDS = {"Wood": 1, "Knot": 2, "Pith": 3}

DEFAULT_YOLO_WEIGHTS = "ann_pipeline/out/knot_runs/yolo11n_v2_all45/weights/best.pt"
SAM_CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
SAM_MODEL_CFG = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"


def _project_meta_json(include_bumaska: bool) -> dict:
    tags = [REVIEW_TAG_META] if include_bumaska else []
    return {
        "classes": [
            {
                "title": c["title"],
                "description": "",
                "shape": c["shape"],
                "color": c["color"],
                "geometry_config": {},
                "hotkey": "",
            }
            for c in CLASS_REGISTRY
        ],
        "tags": tags,
        "projectType": "images",
        "projectSettings": {"multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False}},
    }


def _load_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.uint8)


def _load_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def _bitmap_with_origin(mask: np.ndarray) -> Optional[dict]:
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    patch = mask[y_min : y_max + 1, x_min : x_max + 1].astype(bool)
    b64 = mask_to_base64(torch.from_numpy(patch))
    return {"origin": [int(x_min), int(y_min)], "data": b64}


def _make_bitmap_object(class_title: str, mask: np.ndarray) -> Optional[dict]:
    bmp = _bitmap_with_origin(mask)
    if bmp is None:
        return None
    return {
        "classTitle": class_title,
        "description": "",
        "tags": [],
        "bitmap": bmp,
        "shape": "bitmap",
        "geometryType": "bitmap",
        "labelerLogin": "auto-pipeline",
    }


def _make_point_object(class_title: str, x: float, y: float) -> dict:
    return {
        "classTitle": class_title,
        "description": "",
        "tags": [],
        "points": {"exterior": [[int(round(x)), int(round(y))]], "interior": []},
        "shape": "point",
        "geometryType": "point",
        "labelerLogin": "auto-pipeline",
    }


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    labelled, n = ndi.label(mask.astype(np.uint8))
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, labelled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labelled == largest).astype(np.uint8)


def _yolo_pith(model: YOLO, img_rgb: np.ndarray, conf: float, snap_window: int = 5) -> Optional[Tuple[float, float]]:
    res = model.predict(img_rgb, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    pith_idx = np.where(cls == 1)[0]
    if len(pith_idx) == 0:
        return None
    best = pith_idx[np.argmax(confs[pith_idx])]
    b = xyxy[best]
    x, y = float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0)
    if snap_window <= 0:
        return x, y
    sx, sy = snap_to_darkest(img_rgb[..., :3].mean(axis=-1), x, y, window=snap_window)
    return float(sx), float(sy)


def build_auto_annotation(
    img_gray: np.ndarray,
    img_rgb: np.ndarray,
    pred_frame: np.ndarray,
    yolo_model: YOLO,
    sam_predictor: SAM2ImagePredictor,
    wood_thresh: int = 30,
) -> dict:
    """Build a single auto annotation from propagation output + YOLO pith."""
    h, w = img_gray.shape
    objects: List[dict] = []

    prop_wood = pred_frame == CLASS_IDS["Wood"]
    prop_knot = pred_frame == CLASS_IDS["Knot"]
    prop_pith = pred_frame == CLASS_IDS["Pith"]

    lab, n_cc = ndi.label(prop_knot, structure=np.ones((3, 3), dtype=np.uint8))
    knot_masks = [(lab == k) for k in range(1, n_cc + 1)]

    combined_knot = np.zeros((h, w), dtype=bool)
    for km in knot_masks:
        combined_knot |= km

    thresh_wood = threshold_largest_cc(img_gray, thresh=wood_thresh).astype(bool)
    wood_union = prop_wood | combined_knot | prop_pith | thresh_wood
    wood_final = _largest_cc(wood_union).astype(bool)

    obj = _make_bitmap_object("Wood", wood_final.astype(np.uint8))
    if obj is not None:
        objects.append(obj)

    yolo_xy = _yolo_pith(yolo_model, img_rgb, conf=0.10)
    prop_xy: Optional[Tuple[float, float]] = None
    if prop_pith.any():
        ys, xs = np.nonzero(prop_pith)
        prop_xy = (float(xs.mean()), float(ys.mean()))

    final_pith = yolo_xy or prop_xy

    for km in knot_masks:
        clipped = km & wood_final
        if clipped.sum() < 150:
            continue
        if final_pith is not None:
            px, py = int(round(final_pith[0])), int(round(final_pith[1]))
            inside_cc = 0 <= py < h and 0 <= px < w and bool(clipped[py, px])
            ys, xs = np.nonzero(clipped)
            cc_cx, cc_cy = float(xs.mean()), float(ys.mean())
            dist_to_pith = float(np.hypot(cc_cx - final_pith[0], cc_cy - final_pith[1]))
            if inside_cc or dist_to_pith < 25.0:
                continue
        from skimage import measure as _measure

        props = _measure.regionprops(clipped.astype(np.uint8))
        if props:
            if props[0].eccentricity < 0.7:
                continue
            if props[0].solidity < 0.85:
                continue
        clipped = ndi.binary_fill_holes(clipped)
        obj = _make_bitmap_object("Knot", clipped.astype(np.uint8))
        if obj is not None:
            objects.append(obj)

    if final_pith is not None:
        objects.append(_make_point_object("Pith", *final_pith))

    return {
        "size": {"height": h, "width": w},
        "description": "",
        "tags": [],
        "objects": objects,
    }


def find_anchor_pages(ann_dir: pathlib.Path) -> Set[int]:
    """Return page numbers that have any annotations (knot, wood, or pith)."""
    anchors: Set[int] = set()
    for f in sorted(ann_dir.glob("*.tiff.json")):
        with open(f) as fh:
            ann = json.load(fh)
        if ann.get("objects"):
            page_num = int(f.name.replace("page_", "").replace(".tiff.json", ""))
            anchors.add(page_num)
    return anchors


def write_phase1(
    src_dir: pathlib.Path,
    out_dir: pathlib.Path,
    dataset_name: str,
) -> None:
    """Copy all images + annotations from source into Phase1 layout."""
    meta_path = out_dir / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(_project_meta_json(include_bumaska=False), indent=2))

    ds_dir = out_dir / dataset_name
    img_out = ds_dir / "img"
    ann_out = ds_dir / "ann"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    src_img_dir = src_dir / "img"
    src_ann_dir = src_dir / "ann"

    images = sorted(src_img_dir.glob("*.tiff"))
    for img_path in tqdm(images, desc="Phase1"):
        shutil.copy(img_path, img_out / img_path.name)
        ann_src = src_ann_dir / ("%s.json" % img_path.name)
        if ann_src.exists():
            shutil.copy(ann_src, ann_out / ann_src.name)
        else:
            empty_ann = {
                "size": {"height": 778, "width": 778},
                "description": "",
                "tags": [],
                "objects": [],
            }
            (ann_out / ("%s.json" % img_path.name)).write_text(json.dumps(empty_ann))


def write_phase2(
    src_dir: pathlib.Path,
    out_dir: pathlib.Path,
    dataset_name: str,
    npz_path: str,
    yolo_weights: str,
    anchor_pages: Set[int],
) -> None:
    """Generate Phase2: auto annotations + anchor passthrough with bumaska tag."""
    meta_path = out_dir / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(_project_meta_json(include_bumaska=True), indent=2))

    ds_dir = out_dir / dataset_name
    img_out = ds_dir / "img"
    ann_out = ds_dir / "ann"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=False)
    pages = data["pages"]
    pred = data["pred"]
    page_to_idx = {int(p): i for i, p in enumerate(pages)}
    print("loaded %d frames from %s" % (len(pages), npz_path))

    yolo_model = YOLO(yolo_weights)
    print("building SAM2 image predictor ...")
    sam = build_sam2("//" + SAM_MODEL_CFG, SAM_CHECKPOINT)
    sam_predictor = SAM2ImagePredictor(sam)

    src_img_dir = src_dir / "img"
    src_ann_dir = src_dir / "ann"
    images = sorted(src_img_dir.glob("*.tiff"))

    n_anchor = 0
    n_auto = 0
    for img_path in tqdm(images, desc="Phase2"):
        shutil.copy(img_path, img_out / img_path.name)
        page_num = int(img_path.stem.replace("page_", ""))

        if page_num in anchor_pages:
            ann_src = src_ann_dir / ("%s.json" % img_path.name)
            with open(ann_src) as fh:
                ann = json.load(fh)
            ann["tags"] = [{"name": HUMAN_TAG}]
            (ann_out / ("%s.json" % img_path.name)).write_text(json.dumps(ann))
            n_anchor += 1
        else:
            idx = page_to_idx.get(page_num)
            if idx is None:
                empty_ann = {
                    "size": {"height": 778, "width": 778},
                    "description": "",
                    "tags": [],
                    "objects": [],
                }
                (ann_out / ("%s.json" % img_path.name)).write_text(json.dumps(empty_ann))
                continue
            img_gray = _load_gray(str(img_path))
            img_rgb = _load_rgb(str(img_path))
            ann = build_auto_annotation(img_gray, img_rgb, pred[idx], yolo_model, sam_predictor)
            (ann_out / ("%s.json" % img_path.name)).write_text(json.dumps(ann))
            n_auto += 1

    print("Phase2: %d anchor frames (bumaska), %d auto frames" % (n_anchor, n_auto))


def upload_project(
    local_dir: pathlib.Path,
    project_name: str,
    dataset_name: str,
    workspace_id: int,
    server: str,
    token: str,
) -> int:
    """Create a Supervisely project, upload one dataset. Returns project ID."""
    import supervisely as sly

    api = sly.Api(server_address=server, token=token)
    me = api.user.get_my_info()
    print("connected as user_id=%d (%s)" % (me.id, me.login))

    project = api.project.create(workspace_id, project_name, description="Two-phase annotation workflow")
    print("created project id=%d name=%r" % (project.id, project.name))

    with open(local_dir / "meta.json") as f:
        meta_dict = json.load(f)
    api.project.update_meta(project.id, meta_dict)
    print("updated project meta (classes + tags)")

    ds = api.dataset.create(project.id, dataset_name)
    print("created dataset id=%d name=%r" % (ds.id, ds.name))

    img_dir = local_dir / dataset_name / "img"
    ann_dir = local_dir / dataset_name / "ann"
    image_paths = sorted(img_dir.glob("*.tiff"))
    names = [p.name for p in image_paths]
    paths = [str(p) for p in image_paths]

    print("uploading %d images ..." % len(image_paths))
    img_infos = api.image.upload_paths(ds.id, names, paths)
    image_ids = [info.id for info in img_infos]

    ann_paths = [str(ann_dir / ("%s.json" % name)) for name in names]
    print("uploading %d annotations ..." % len(ann_paths))
    api.annotation.upload_paths(image_ids, ann_paths)

    print("upload done for %s/%s" % (project_name, dataset_name))
    return project.id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Phase1/Phase2 Supervisely projects.")
    parser.add_argument("--subset", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="/tmp/phases")
    parser.add_argument(
        "--npz",
        default="experiments/sm2025_subset4_propagate/out/result_obb_aug_ellipse.npz",
    )
    parser.add_argument("--yolo_weights", default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--dry_run", action="store_true", help="Build local export only, skip upload even if --upload is set."
    )
    parser.add_argument("--server", default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"))
    parser.add_argument(
        "--workspace_id",
        type=int,
        default=None,
        help="Supervisely workspace ID. Auto-detected from SM_2025 project if not set.",
    )
    args = parser.parse_args()

    src_dir = pathlib.Path("/mnt/D/datasets/ct_log/375492_SM_2025/%d" % args.subset)
    if not src_dir.exists():
        msg = "source dir not found: %s" % src_dir
        raise ValueError(msg)

    anchor_pages = find_anchor_pages(src_dir / "ann")
    print(
        "subset %d: %d total frames, %d anchor frames"
        % (
            args.subset,
            len(list((src_dir / "img").glob("*.tiff"))),
            len(anchor_pages),
        )
    )

    out_dir = pathlib.Path(args.out_dir)
    phase1_dir = out_dir / "phase1"
    phase2_dir = out_dir / "phase2"
    dataset_name = str(args.subset)

    print("\n=== Phase1: copying source annotations ===")
    write_phase1(src_dir, phase1_dir, dataset_name)

    print("\n=== Phase2: generating auto annotations ===")
    write_phase2(src_dir, phase2_dir, dataset_name, args.npz, args.yolo_weights, anchor_pages)

    if args.upload and not args.dry_run:
        token = os.environ.get("SUPERVISELY_TOKEN")
        if not token:
            msg = "SUPERVISELY_TOKEN not set; source .env first"
            raise ValueError(msg)

        workspace_id = args.workspace_id
        if workspace_id is None:
            import supervisely as sly

            api = sly.Api(server_address=args.server, token=token)
            project = api.project.get_info_by_id(376641)
            workspace_id = project.workspace_id
            print("auto-detected workspace_id=%d from SM_2025 project" % workspace_id)

        print("\n=== Uploading Phase1 ===")
        p1_id = upload_project(phase1_dir, "Phase1", dataset_name, workspace_id, args.server, token)

        print("\n=== Uploading Phase2 ===")
        p2_id = upload_project(phase2_dir, "Phase2", dataset_name, workspace_id, args.server, token)

        print("\nDone. Phase1 project_id=%d, Phase2 project_id=%d" % (p1_id, p2_id))
    elif not args.upload:
        print("\nDry run complete. Add --upload to push to Supervisely.")


if __name__ == "__main__":
    main()
