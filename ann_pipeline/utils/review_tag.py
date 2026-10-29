"""Keep the human-review tag (``bumaska``) in every CT-log annotation project.

Reviewers can only tick ``bumaska`` on a frame when the project meta defines the tag. A project created from
classes alone (e.g. ``sly.ProjectMeta(obj_classes=...)``) silently lacks it. Build new metas with
``with_review_tag`` and audit the workspace with this script after creating a project.

An annotation project is one whose classes include Knot, Wood and Pith. Phase1 is exempt: every frame there is
human-annotated by construction.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

import supervisely as sly

HUMAN_TAG = "bumaska"
REVIEW_TAG_META = {
    "name": HUMAN_TAG,
    "value_type": "none",
    "color": "#FF0000",
    "hotkey": "",
    "applicable_type": "all",
    "classes": [],
    "target_type": "all",
}
ANNOTATION_CLASSES = {"Knot", "Wood", "Pith"}
EXEMPT_PROJECT_IDS = {377327}


def with_review_tag(meta: sly.ProjectMeta) -> sly.ProjectMeta:
    if meta.get_tag_meta(HUMAN_TAG) is not None:
        return meta
    return meta.add_tag_meta(sly.TagMeta.from_json(REVIEW_TAG_META))


def ensure_review_tag(api: sly.Api, project_id: int, dry_run: bool = False) -> bool:
    meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    if meta.get_tag_meta(HUMAN_TAG) is not None:
        return False
    if not dry_run:
        api.project.update_meta(project_id, with_review_tag(meta).to_json())
    return True


def is_annotation_project(meta: sly.ProjectMeta) -> bool:
    return ANNOTATION_CLASSES <= {c.name for c in meta.obj_classes}


def audit(api: sly.Api, project_ids: Optional[List[int]], fix: bool) -> List[int]:
    if project_ids is None:
        projects = [
            p
            for team in api.team.get_list()
            for ws in api.workspace.get_list(team.id)
            for p in api.project.get_list(ws.id)
        ]
    else:
        projects = [api.project.get_info_by_id(pid) for pid in project_ids]

    missing = []
    for project in projects:
        meta = sly.ProjectMeta.from_json(api.project.get_meta(project.id))
        if project_ids is None and (project.id in EXEMPT_PROJECT_IDS or not is_annotation_project(meta)):
            continue
        has_tag = meta.get_tag_meta(HUMAN_TAG) is not None
        if not has_tag:
            missing.append(project.id)
            if fix:
                ensure_review_tag(api, project.id)
        status = "ok" if has_tag else ("ADDED" if fix else "MISSING")
        print("%-8s %d %s" % (status, project.id, project.name))

    return missing


def main() -> None:
    """Usage below.

    ```bash
    set -a && source .env && set +a
    python -m ann_pipeline.utils.review_tag                 # audit every Knot/Wood/Pith project, exit 1 if any lacks it
    python -m ann_pipeline.utils.review_tag --fix           # add the tag meta where missing
    python -m ann_pipeline.utils.review_tag --project_ids 384679 --fix
    ```

    Raises:
        ValueError: If the token is not set in the environment variables.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_ids", type=int, nargs="+", default=None, help="Default: all annotation projects.")
    parser.add_argument("--fix", action="store_true", help="Add the tag meta to projects that lack it.")
    parser.add_argument("--server", default=os.environ.get("SUPERVISELY_SERVER", "https://app.supervisely.com"))
    args = parser.parse_args()

    for handler in sly.logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)

    token = os.environ.get("SUPERVISELY_TOKEN")
    if not token:
        msg = "SUPERVISELY_TOKEN not set; source .env first"
        raise ValueError(msg)

    missing = audit(sly.Api(server_address=args.server, token=token), args.project_ids, args.fix)
    if missing and not args.fix:
        sys.exit(1)


if __name__ == "__main__":
    main()
