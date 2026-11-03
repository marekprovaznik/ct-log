import argparse
import datetime
import json
import os
from pathlib import Path

import supervisely as sly


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Back up the annotations, meta and image infos of a Supervisely project (no images)."
    )
    parser.add_argument("--project_id", type=int, required=True)
    parser.add_argument("--out_root", type=str, required=True, help="Backup goes to <out_root>/<name>_<stamp>_backup.")
    parser.add_argument("--server", type=str, default="https://app.supervisely.com")
    args = parser.parse_args()

    api = sly.Api(server_address=args.server, token=os.environ["SUPERVISELY_TOKEN"])
    project = api.project.get_info_by_id(args.project_id)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.out_root) / ("%s_%s_backup" % (project.name, stamp))
    if out_dir.exists():
        msg = "Backup directory %s already exists" % out_dir
        raise ValueError(msg)
    sly.download_project(api, project.id, str(out_dir), save_images=False, save_image_info=True, log_progress=False)
    after = api.project.get_info_by_id(args.project_id)
    info = {
        "project_id": project.id,
        "project_name": project.name,
        "downloaded_at": stamp,
        "project_updated_at_before": project.updated_at,
        "project_updated_at_after": after.updated_at,
        "images_count": project.items_count,
    }
    (out_dir / "BACKUP.json").write_text(json.dumps(info, indent=2))
    print(out_dir)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
