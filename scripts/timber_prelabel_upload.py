import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from camera_slider.training_set import WOOD_N, wood_components
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import supervisely as sly

Image.MAX_IMAGE_PIXELS = None
S8 = np.ones((3, 3), dtype=bool)
REVIEW_TAG = "bumaska"
WOOD_COLOUR = [205, 170, 125]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload model pre-labels of board pieces to a new Supervisely project."
    )
    parser.add_argument(
        "--pred_dir", type=str, required=True, help="Directory with <piece>.png class maps and manifest.json."
    )
    parser.add_argument(
        "--meta_json", type=str, required=True, help="meta.json of the source project (classes and tags)."
    )
    parser.add_argument("--workspace_id", type=int, default=None, help="Create a new project here.")
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--project_id", type=int, default=None, help="Overwrite annotations of images already in it.")
    parser.add_argument("--wood_class", type=str, default=None, help="Also add wood (training-set rule) as this class.")
    parser.add_argument("--black_threshold", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, default="boards")
    parser.add_argument(
        "--class_map", type=str, nargs="+", required=True, help="index=class_title, e.g. 3=knot_unclassified"
    )
    parser.add_argument(
        "--min_area_target_px", type=int, default=16, help="Drop components smaller than this at 4 px/mm."
    )
    parser.add_argument("--server", type=str, default="https://app.supervisely.com")
    args = parser.parse_args()

    class_map = {int(k): v for k, v in (item.split("=", 1) for item in args.class_map)}
    meta = sly.ProjectMeta.from_json(json.loads(Path(args.meta_json).read_text()))
    for title in class_map.values():
        if meta.get_obj_class(title) is None:
            meta = meta.add_obj_class(sly.ObjClass(title, sly.Bitmap, color=[255, 60, 0]))
    if args.wood_class and meta.get_obj_class(args.wood_class) is None:
        meta = meta.add_obj_class(sly.ObjClass(args.wood_class, sly.Bitmap, color=WOOD_COLOUR))
    if meta.get_tag_meta(REVIEW_TAG) is None:
        msg = "Source meta has no %r tag; reviewers could not mark frames" % REVIEW_TAG
        raise ValueError(msg)

    api = sly.Api(server_address=args.server, token=os.environ["SUPERVISELY_TOKEN"])
    if args.project_id is not None:
        project = api.project.get_info_by_id(args.project_id)
        meta = sly.ProjectMeta.from_json(api.project.get_meta(project.id)).merge(meta)
        api.project.update_meta(project.id, meta.to_json())
        dataset = api.dataset.get_info_by_name(project.id, args.dataset_name)
        existing = {info.name: info for info in api.image.get_list(dataset.id)}
    elif args.workspace_id is not None and args.project_name:
        project = api.project.create(args.workspace_id, args.project_name, change_name_if_conflict=True)
        api.project.update_meta(project.id, meta.to_json())
        dataset = api.dataset.create(project.id, args.dataset_name)
        existing = {}
    else:
        msg = "Pass --project_id, or --workspace_id with --project_name"
        raise ValueError(msg)
    manifest = json.loads((Path(args.pred_dir) / "manifest.json").read_text())
    for entry in manifest:
        labels_map = np.asarray(Image.open(Path(args.pred_dir) / (Path(entry["name"]).stem + ".png")))
        min_area = int(round(args.min_area_target_px / entry["scale"] ** 2))
        labels = build_labels(labels_map, class_map, meta, min_area)
        if args.wood_class:
            labels = (
                wood_labels(entry["image"], entry["name"], meta.get_obj_class(args.wood_class), args.black_threshold)
                + labels
            )
        info = existing.get(entry["name"]) or api.image.upload_path(dataset.id, entry["name"], entry["image"])
        annotation = sly.Annotation(img_size=labels_map.shape[:2], labels=labels)
        api.annotation.upload_ann(info.id, annotation)
        counts: Dict[str, int] = {}
        for label in labels:
            counts[label.obj_class.name] = counts.get(label.obj_class.name, 0) + 1
        print("%s image %d objects %s" % (entry["name"], info.id, counts), flush=True)
    print("project %d %s" % (project.id, project.name))


def build_labels(
    labels_map: np.ndarray, class_map: Dict[int, str], meta: sly.ProjectMeta, min_area: int
) -> List[sly.Label]:
    """One bitmap label per 8-connected component of each mapped class.

    Args:
        labels_map: [H, W] predicted class indices at native resolution.
        class_map: Class index -> Supervisely class title.
        meta: Project meta holding those classes.
        min_area: Drop components with fewer native pixels.

    Returns:
        List[sly.Label]: Labels for the annotation.
    """
    labels: List[sly.Label] = []
    for index, title in class_map.items():
        obj_class = meta.get_obj_class(title)
        components, count = ndi.label(labels_map == index, structure=S8)
        for sl, number in zip(ndi.find_objects(components), range(1, count + 1)):
            mask = components[sl] == number
            if mask.sum() < min_area:
                continue
            bitmap = sly.Bitmap(mask, origin=sly.PointLocation(row=sl[0].start, col=sl[1].start))
            labels.append(sly.Label(bitmap, obj_class))
    return labels


def wood_labels(image_path: str, name: str, obj_class: sly.ObjClass, black_threshold: int) -> List[sly.Label]:
    """Wood as in the training set: N largest non-black components (4 spruce, 2 larch), holes filled.

    Args:
        image_path: Native piece image.
        name: Piece file name; its prefix is the species.
        obj_class: Supervisely wood class.
        black_threshold: Grey level above which a pixel is not background.

    Returns:
        List[sly.Label]: One label per wood component.
    """
    species = name.split("_", 1)[0]
    if species not in WOOD_N:
        msg = "Unknown species %r in %s" % (species, name)
        raise ValueError(msg)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    wood, _ = wood_components(image, WOOD_N[species], black_threshold)
    components, count = ndi.label(wood, structure=S8)
    labels: List[sly.Label] = []
    for sl, number in zip(ndi.find_objects(components), range(1, count + 1)):
        mask = components[sl] == number
        labels.append(
            sly.Label(sly.Bitmap(mask, origin=sly.PointLocation(row=sl[0].start, col=sl[1].start)), obj_class)
        )
    return labels


if __name__ == "__main__":
    main()
