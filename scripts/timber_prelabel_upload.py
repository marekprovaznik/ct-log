import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
VOLATILE_KEYS = {"updatedAt", "createdAt"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload model pre-labels of board pieces to Supervisely: to a new project, in place into a preview "
            "project (--project_id), or appended to a working project (--merge_backup_dir)."
        )
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
    parser.add_argument(
        "--merge_backup_dir",
        type=str,
        default=None,
        help=(
            "Backup of --project_id (scripts/sly_backup.py). Append-only: existing objects and tags are kept, "
            "pieces with the review tag get only wood, and nothing is written if any annotation differs from it."
        ),
    )
    parser.add_argument(
        "--dry_run_dir",
        type=str,
        default=None,
        help="With --merge_backup_dir: write merged annotations and counts here, no Supervisely calls.",
    )
    parser.add_argument(
        "--check_only", action="store_true", help="With --merge_backup_dir: compare the project with the backup only."
    )
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
    meta = extend_meta(
        sly.ProjectMeta.from_json(json.loads(Path(args.meta_json).read_text())), class_map, args.wood_class
    )
    manifest = json.loads((Path(args.pred_dir) / "manifest.json").read_text())
    if args.merge_backup_dir is not None:
        merge(args, manifest, class_map, meta)
        return
    if args.dry_run_dir is not None:
        msg = "--dry_run_dir needs --merge_backup_dir"
        raise ValueError(msg)
    replace(args, manifest, class_map, meta)


def extend_meta(meta: sly.ProjectMeta, class_map: Dict[int, str], wood_class: Optional[str]) -> sly.ProjectMeta:
    for title in class_map.values():
        if meta.get_obj_class(title) is None:
            meta = meta.add_obj_class(sly.ObjClass(title, sly.Bitmap, color=[255, 60, 0]))
    if wood_class and meta.get_obj_class(wood_class) is None:
        meta = meta.add_obj_class(sly.ObjClass(wood_class, sly.Bitmap, color=WOOD_COLOUR))
    if meta.get_tag_meta(REVIEW_TAG) is None:
        msg = "Source meta has no %r tag; reviewers could not mark frames" % REVIEW_TAG
        raise ValueError(msg)
    return meta


def replace(args: argparse.Namespace, manifest: List[Dict], class_map: Dict[int, str], meta: sly.ProjectMeta) -> None:
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
    for entry in manifest:
        labels, size = piece_labels(args, entry, class_map, meta, with_defects=True)
        info = existing.get(entry["name"]) or api.image.upload_path(dataset.id, entry["name"], entry["image"])
        api.annotation.upload_ann(info.id, sly.Annotation(img_size=size, labels=labels))
        print("%s image %d objects %s" % (entry["name"], info.id, count_labels(labels)), flush=True)
    print("project %d %s" % (project.id, project.name))


def merge(args: argparse.Namespace, manifest: List[Dict], class_map: Dict[int, str], meta: sly.ProjectMeta) -> None:
    backup = Path(args.merge_backup_dir) / args.dataset_name / "ann"
    backup_anns = {entry["name"]: json.loads((backup / (entry["name"] + ".json")).read_text()) for entry in manifest}
    if args.dry_run_dir is not None:
        write_dry_run(args, manifest, class_map, meta, backup_anns)
        return
    if args.project_id is None:
        msg = "--merge_backup_dir needs --project_id"
        raise ValueError(msg)
    api = sly.Api(server_address=args.server, token=os.environ["SUPERVISELY_TOKEN"])
    project = api.project.get_info_by_id(args.project_id)
    dataset = api.dataset.get_info_by_name(project.id, args.dataset_name)
    infos = {info.name: info for info in api.image.get_list(dataset.id)}
    missing = sorted(entry["name"] for entry in manifest if entry["name"] not in infos)
    if missing:
        msg = "Pieces not in project %d: %s" % (project.id, missing)
        raise ValueError(msg)
    current = {
        info.name: ann.annotation
        for info, ann in zip(
            [infos[entry["name"]] for entry in manifest],
            api.annotation.download_batch(dataset.id, [infos[entry["name"]].id for entry in manifest]),
        )
    }
    check_unchanged(current, backup_anns, args.wood_class)
    extra = sorted(set(infos) - set(current))
    print("%d annotations equal the backup; pieces in the project but not in the manifest: %s" % (len(current), extra))
    if args.check_only:
        return
    server_meta = sly.ProjectMeta.from_json(api.project.get_meta(project.id))
    api.project.update_meta(project.id, server_meta.merge(meta).to_json())
    meta = sly.ProjectMeta.from_json(api.project.get_meta(project.id))
    for entry in manifest:
        reviewed = is_reviewed(current[entry["name"]])
        labels, _ = piece_labels(args, entry, class_map, meta, with_defects=not reviewed)
        api.annotation.append_labels(infos[entry["name"]].id, labels)
        print(
            "%s image %d reviewed %s added %s"
            % (entry["name"], infos[entry["name"]].id, reviewed, count_labels(labels)),
            flush=True,
        )
    print("project %d %s: appended to %d pieces" % (project.id, project.name, len(manifest)))


def write_dry_run(
    args: argparse.Namespace,
    manifest: List[Dict],
    class_map: Dict[int, str],
    meta: sly.ProjectMeta,
    backup_anns: Dict[str, Dict],
) -> None:
    check_unchanged(backup_anns, backup_anns, args.wood_class)
    out_dir = Path(args.dry_run_dir)
    (out_dir / "ann").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta.to_json(), indent=2))
    summary = []
    for entry in manifest:
        source = backup_anns[entry["name"]]
        reviewed = is_reviewed(source)
        labels, _ = piece_labels(args, entry, class_map, meta, with_defects=not reviewed)
        merged = dict(source)
        merged["objects"] = list(source["objects"]) + [label.to_json() for label in labels]
        (out_dir / "ann" / (entry["name"] + ".json")).write_text(json.dumps(merged))
        existing: Dict[str, int] = {}
        for obj in source["objects"]:
            existing[obj["classTitle"]] = existing.get(obj["classTitle"], 0) + 1
        summary.append(
            {
                "name": entry["name"],
                "species": entry["name"].split("_", 1)[0],
                "reviewed": reviewed,
                "existing": existing,
                "added": count_labels(labels),
            }
        )
        print(
            "%s reviewed %s existing %s added %s" % (entry["name"], reviewed, existing, count_labels(labels)),
            flush=True,
        )
    (out_dir / "counts.json").write_text(json.dumps(summary, indent=2))


def check_unchanged(current: Dict[str, Dict], backup_anns: Dict[str, Dict], wood_class: Optional[str]) -> None:
    changed = sorted(name for name in backup_anns if strip_volatile(current[name]) != strip_volatile(backup_anns[name]))
    if changed:
        msg = "%d annotations differ from the backup, nothing written: %s" % (len(changed), changed[:20])
        raise ValueError(msg)
    with_wood = sorted(
        name for name, ann in current.items() if any(obj["classTitle"] == wood_class for obj in ann["objects"])
    )
    if with_wood:
        msg = "%d pieces already hold %r, nothing written: %s" % (len(with_wood), wood_class, with_wood[:20])
        raise ValueError(msg)


def strip_volatile(value: object) -> object:
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def is_reviewed(ann: Dict) -> bool:
    return any(tag["name"] == REVIEW_TAG for tag in ann.get("tags", []))


def piece_labels(
    args: argparse.Namespace, entry: Dict, class_map: Dict[int, str], meta: sly.ProjectMeta, with_defects: bool
) -> Tuple[List[sly.Label], Tuple[int, int]]:
    labels_map = np.asarray(Image.open(Path(args.pred_dir) / (Path(entry["name"]).stem + ".png")))
    labels: List[sly.Label] = []
    if args.wood_class:
        labels += wood_labels(entry["image"], entry["name"], meta.get_obj_class(args.wood_class), args.black_threshold)
    if with_defects:
        min_area = int(round(args.min_area_target_px / entry["scale"] ** 2))
        labels += build_labels(labels_map, class_map, meta, min_area)
    return labels, labels_map.shape[:2]


def count_labels(labels: List[sly.Label]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label.obj_class.name] = counts.get(label.obj_class.name, 0) + 1
    return counts


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
