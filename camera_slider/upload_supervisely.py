import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from ann_pipeline.utils.review_tag import with_review_tag
import supervisely as sly

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the stitched-board pieces to a Supervisely project.")
    parser.add_argument(
        "--project_dir", type=Path, required=True, help="local Supervisely project from camera_slider.dataset"
    )
    parser.add_argument("--dataset_name", default="boards")
    parser.add_argument("--project_name", default="camera_slider_v1")
    parser.add_argument("--meta_project_id", type=int, default=385546, help="project whose classes and tags are copied")
    parser.add_argument("--workspace_id", type=int, default=None, help="default: workspace of --meta_project_id")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--server", default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    upload(args)


def upload(args: argparse.Namespace) -> None:
    token = os.environ.get("SUPERVISELY_TOKEN")
    if not token:
        msg = "SUPERVISELY_TOKEN not set; run with with-secrets"
        raise ValueError(msg)
    api = sly.Api(server_address=args.server, token=token)
    LOGGER.info("Connected as %s", api.user.get_my_info().login)

    source = api.project.get_info_by_id(args.meta_project_id)
    workspace_id = args.workspace_id or source.workspace_id
    meta = with_review_tag(sly.ProjectMeta.from_json(api.project.get_meta(args.meta_project_id)))
    project = api.project.get_info_by_name(workspace_id, args.project_name)
    if project is None:
        if args.dry_run:
            LOGGER.info("Would create project %r in workspace %d", args.project_name, workspace_id)
        else:
            project = api.project.create(workspace_id, args.project_name, type=sly.ProjectType.IMAGES)
            api.project.update_meta(project.id, meta.to_json())
            LOGGER.info(
                "Created project %r (id %d) with the classes of project %d",
                args.project_name,
                project.id,
                args.meta_project_id,
            )

    images = sorted((args.project_dir / args.dataset_name / "img").glob("*.jpg"))
    existing = set()
    dataset = None
    if project is not None:
        dataset = api.dataset.get_info_by_name(project.id, args.dataset_name)
        if dataset is None and not args.dry_run:
            dataset = api.dataset.create(project.id, args.dataset_name)
        if dataset is not None:
            existing = {info.name for info in api.image.get_list(dataset.id)}
    todo = [p for p in images if p.name not in existing]
    LOGGER.info("%d pieces local, %d already uploaded, %d to upload", len(images), len(existing), len(todo))
    if args.dry_run or not todo:
        return
    for start in range(0, len(todo), args.batch):
        chunk: List[Path] = todo[start : start + args.batch]
        infos = api.image.upload_paths(dataset.id, [p.name for p in chunk], [str(p) for p in chunk])
        anns = [json.loads((p.parent.parent / "ann" / f"{p.name}.json").read_text()) for p in chunk]
        api.annotation.upload_jsons([i.id for i in infos], anns)
        LOGGER.info("Uploaded %d/%d", start + len(chunk), len(todo))
    LOGGER.info("Done: project %d, dataset %r", project.id, args.dataset_name)


if __name__ == "__main__":
    main()
