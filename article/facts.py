import argparse
import datetime
import json
from pathlib import Path
import re
from typing import Dict

import yaml

PAGE_RE = re.compile(r"^page_\d+\.tiff$")
REVIEW_TAG = "bumaska"
HUMAN_TRAIN_LOGS = ["1", "2", "4", "08"]
HELDOUT_LOGS = ["10"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset facts for the abstract: CT frames on disk, Phase2 bumaska counts, M5 training set."
    )
    parser.add_argument("--ct_root", type=Path, default=Path("/mnt/D/datasets/ct_log/CT"))
    parser.add_argument(
        "--phase2_tags",
        type=Path,
        required=True,
        help="JSON from ann_pipeline.utils.get_tag_info_from_supervisely (Phase2, project 377328).",
    )
    parser.add_argument("--phase2_date", required=True, help="Date the Phase2 tags were read (YYYY-MM-DD).")
    parser.add_argument("--dataset_json", type=Path, default=Path("/mnt/D/datasets/ct_log/kwp_ds_v4/DATASET.json"))
    parser.add_argument(
        "--config", type=Path, default=Path("src/configs/train_kwp_v4_unet_effv2s_w1_dice_ts_48ep_dec512.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("article/data/facts.json"))
    args = parser.parse_args()

    ct = ct_frames(args.ct_root)
    phase2 = phase2_counts(json.loads(args.phase2_tags.read_text()), args.phase2_date, ct)
    training = m5_training_set(json.loads(args.dataset_json.read_text()), yaml.safe_load(args.config.read_text()))
    facts = {
        "generated": datetime.date.today().isoformat(),
        "ct_frames_on_disk": ct,
        "phase2_bumaska": phase2,
        "m5_training_set": training,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=1))
    print("wrote %s" % args.out)


def ct_frames(root: Path) -> dict:
    logs = {
        d.name: sum(1 for p in d.iterdir() if PAGE_RE.match(p.name))
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name.isdigit()
    }
    return {
        "source": str(root),
        "n_logs": len(logs),
        "total_frames": sum(logs.values()),
        "min_frames_per_log": min(logs.values()),
        "max_frames_per_log": max(logs.values()),
        "frames_per_log": logs,
    }


def phase2_counts(tags: dict, date: str, ct: dict) -> dict:
    logs: Dict[str, dict] = {}
    other: Dict[str, int] = {}
    for ds in sorted(tags["datasets"], key=lambda d: int(d["dataset_name"]) if d["dataset_name"].isdigit() else 0):
        name = ds["dataset_name"]
        if not name.isdigit():
            other[name] = ds["images_count"]
            continue
        logs[name] = {
            "images": ds["images_count"],
            "bumaska": ds["tags"].get(REVIEW_TAG, {}).get("count", 0),
            "ct_frames_on_disk": ct["frames_per_log"].get(name.zfill(2)),
            "m5_role": m5_role(name),
        }
    return {
        "read_on": date,
        "project_id": tags["project_id"],
        "project_name": tags["project_name"],
        "tag": REVIEW_TAG,
        "n_logs": len(logs),
        "total_images": sum(v["images"] for v in logs.values()),
        "total_bumaska": sum(v["bumaska"] for v in logs.values()),
        "total_ct_frames_on_disk": sum(v["ct_frames_on_disk"] or 0 for v in logs.values()),
        "logs": logs,
        "other_datasets_not_counted": other,
    }


def m5_role(log: str) -> str:
    if log in HELDOUT_LOGS:
        return "held out (test)"
    if log in HUMAN_TRAIN_LOGS:
        return "train, human frames"
    return "train, model pre-labels only"


def m5_training_set(dataset: dict, config: dict) -> dict:
    train = [Path(p).name for p in config["train_logs"]]
    test = sorted({Path(p).name for p in config["test_logs"]})
    logs = dataset["logs"]
    human = {log: human_frames(logs[log]) for log in train if human_frames(logs[log])}
    return {
        "dataset_version": dataset["version"],
        "config": config_name(config),
        "n_train_logs": len(train),
        "train_frames": sum(logs[log]["frames"] for log in train),
        "train_human_frames": sum(human.values()),
        "train_human_frames_per_log": human,
        "train_auto_frames": sum(logs[log]["frames"] - human_frames(logs[log]) for log in train),
        "test_logs": test,
        "test_frames": {log: logs[log]["frames"] for log in test},
        "train_logs": sorted(train, key=int),
    }


def human_frames(entry: dict) -> int:
    return entry["frames"] if entry["kind"] == "human" else entry.get("human_frames", 0)


def config_name(config: dict) -> str:
    return Path(config["checkpoint_path"]).stem


if __name__ == "__main__":
    main()
