# Two-phase annotation workflow

## Overview

```
Phase1 (human anchors)          Phase2 (auto + human review)
┌──────────────────────┐        ┌──────────────────────────────┐
│ All frames, ~10-15%  │        │ All frames have annotations  │
│ annotated by humans  │───────>│ Anchors: copied from Phase1  │
│ (knots only needed)  │        │ Rest: auto-generated         │
│                      │        │ Tag 'bumaska' = human-verified│
└──────────────────────┘        └──────────────────────────────┘
```

## Phase1 — anchor collection

- One dataset per log (e.g. `Phase1/4` for subset 4)
- Annotators create knot masks on ~every 10-15th frame
- Wood and pith are NOT needed — computed automatically
- Target: ~20 anchors per log, denser in knot-heavy regions

## Phase2 — auto annotations for review

- Same images as Phase1, but every frame has annotations
- **Anchor frames**: copied verbatim from Phase1, tagged `bumaska`
- **Non-anchor frames**: OBB-augmented ellipse propagation + filters
- Annotators correct auto annotations, then tag the frame `bumaska`
- Untagged frames = untouched auto output

## The `bumaska` tag

Image-level tag. Meaning: "a human has verified this frame."

- Anchor frames arrive pre-tagged (they're already human-verified)
- Annotators add the tag after correcting an auto frame
- Lets us track coverage: % of frames with `bumaska` = review progress

## Auto annotation pipeline (Phase2)

Recipe: OBB-augmented anchor propagation v2.2 (ellipse seeds).

1. MedSAM2 video propagation seeded with anchor GT + YOLO-OBB ellipses
2. Per-frame filters: min 150px, pith exclusion 25px, eccentricity >0.7,
   solidity >0.85, fill_holes
3. Pith: YOLO bbox centroid (conf=0.10)
4. Wood: iterative peel (t=30, peel<60, n=5) ∪ propagated wood

## Supervisely projects

| project | ID | description |
|---|---|---|
| Phase1 | 377327 | Human anchors |
| Phase2 | 377328 | Auto + human review |
| Human-collection | 378193 | Accumulated `bumaska` (human-reviewed) frames from Phase2 |

Datasets: `4` (subset 4, 292 frames, 45 anchors).

## Human-collection: accumulating reviewed frames

`Human-collection` (378193) mirrors Phase2's dataset layout but holds only the
`bumaska`-tagged (human-reviewed) frames. It is the clean training pool.

- **Initial build** (create-only, errors if it exists):
  `python -m ann_pipeline.utils.build_human_collection`
- **Incremental sync** (append-only; never touches/overwrites existing frames,
  matches by image name, creates missing datasets, idempotent):
  `python -m ann_pipeline.utils.update_human_collection [--dry-run]`

## Retrain workflow (knot detectors)

The knot data-prep scripts read a **local** project root
(`/mnt/D/datasets/ct_log/375492_SM_2025/{subset}/{ann,img}/`), not Supervisely
directly. New reviewed frames must first be pulled to disk:

```bash
set -a && source .env && set +a

# 1. Pull bumaska frames from Phase2 to the local root (skips existing files;
#    --overwrite to refresh a subset whose reviewed set grew, e.g. subset 4).
python -m ann_pipeline.utils.download_all_tagged_images_from_supervisely \
    --subsets 08 10

# 2. Build datasets with a log-level holdout (whole logs -> val via
#    --val_subsets; all others -> train). Both detectors share the split.
python -m ann_pipeline.knot.data_prep_obb \
    --subsets 1 2 3 4 08 10 --val_subsets 2 10 \
    --out_dir ann_pipeline/out/knot_yolo_obb_v3
python -m ann_pipeline.knot.data_prep \
    --subsets 1 2 3 4 08 10 --val_subsets 2 10 \
    --out_dir ann_pipeline/out/knot_yolo_v3

# 3. Train (OBB needs an -obb pretrained model; task auto-detected).
conda run -n ct-log python -m ann_pipeline.knot.train \
    --data ann_pipeline/out/knot_yolo_obb_v3/knots_obb.yaml \
    --model yolo11n-obb.pt --name yolo11n_obb_holdout_v3
conda run -n ct-log python -m ann_pipeline.knot.train \
    --data ann_pipeline/out/knot_yolo_v3/knots.yaml \
    --model yolo11n.pt --name yolo11n_2cls_holdout_v3
```

**Recommended detector — single 2-class OBB (knot + pith):** one model replaces
the OBB-knot + axis-aligned-pith split and beats it on every metric (see
`CLAUDE.md`). Build with the 2-class OBB prep instead of the two preps above:

```bash
python -m ann_pipeline.knot.data_prep_obb_2cls \
    --subsets 1 2 3 4 08 10 --val_subsets 2 10 \
    --out_dir ann_pipeline/out/knot_yolo_obb_2cls_v3
conda run -n ct-log python -m ann_pipeline.knot.train \
    --data ann_pipeline/out/knot_yolo_obb_2cls_v3/knots_obb_2cls.yaml \
    --model yolo11n-obb.pt --name yolo11n_obb_2cls_holdout_v3
```

Log-level holdout (val = held-out logs, e.g. 2 + 10) gives an honest
generalization estimate to unseen logs, unlike the default per-frame random
split where correlated slices from one log leak across train/val.

## Running

```bash
# Build locally (dry run):
set -a && source .env && set +a
conda run -n ct-log python -m ann_pipeline.scripts.setup_phases \
    --subset 4 --out_dir /tmp/phases --dry_run

# Build + upload:
conda run -n ct-log python -m ann_pipeline.scripts.setup_phases \
    --subset 4 --out_dir /tmp/phases --upload
```

## 2026-10-09: Phase2 pre-labels replaced with the U-Net ensemble default

Unreviewed frames in Phase2 (377328) were overwritten:
- logs 3 (298 frames), 05 (299), 06 (299) and 09 (300), none of which had `bumaska` frames or edits since 2026-07-13;
- 271 untagged frames of 08.

The 27 `bumaska` frames of 08 were skipped and verified unchanged.

Labels:
- **Knots:** mean of the v4 U-Nets EffV2-S + ConvNeXt-S (no TTA), p >= 0.5, clipped to wood.
- **Pith:** ensemble pith (mean of the two soft-argmax points, rounded, no snap).
- **Wood:** `threshold_peel`.

Built with `build_kwp_dataset.py --knot_root /mnt/D/datasets/ct_log/knot_probs_ens_v4
--pith_root /mnt/D/datasets/ct_log/unet_pith_ens_v4`.
- Export tar sha256: knots `4f7d894f…a583`, pith `960ffeb6…6068`.
- Build record: `dataset_manifests/phase2-ens-v4.DATASET.json` (manifest sha256 `0c11eb80…9d11`).
- Log-10 check of the exports: knot IoU 0.763; pith 66 % exact / 95 % within 1 px / 99 % within 2 px,
  mean 0.39 px.

Upload: `logs/phase2_deploy_20261009/upload_prelabels.py` (skips `bumaska`). Every uploaded frame was
re-downloaded and matches the local build object for object.

Backup of the previous pre-labels (propagation pipeline, detector v3) and the project meta:
`/mnt/D/datasets/ct_log/backups/phase2_prelabels_before_ensemble_20261009_1803/<log>/<image>.json`.

## 2026-10-15: Phase2 pre-labels replaced with M5

Same unreviewed frames as on 2026-10-09 were overwritten: logs 3 (298), 05 (299), 06 (299), 09 (300) and the
271 untagged frames of 08. The 27 `bumaska` frames of 08 were skipped. Before the upload, every overwritten
frame still matched the 2026-10-09 ensemble build object for object (no untagged human edits were lost).

Labels:
- **Knots:** M5 = E1, the single EffV2-S U-Net with a 512-256-128-64-32 decoder, 48 epochs, EMA
  (`seg_head_epoch_47.pth`, sha256 `228ed105…88b0`), no TTA, p >= 0.5, clipped to wood.
- **Pith:** M5 soft-argmax point, rounded, no snap.
- **Wood:** `threshold_peel`.

On log 10, M5 is level with the M4 ensemble in pixel knot IoU (0.763 vs 0.763, +0.001 [−0.007, +0.009])
and better on instances: mean per-GT IoU +0.025 [+0.015, +0.038], mAP50 +0.034, thin recall +0.128
(`report_2026-10/bootstrap_instance_e1.json`). Pith: 66.9 % exact, 94.9 % within 1 px, mean 0.38 px.

Built with `build_kwp_dataset.py --version phase2-m5 --knot_root /mnt/D/datasets/ct_log/knot_probs_m5
--pith_root /mnt/D/datasets/ct_log/unet_pith_m5` into `/mnt/D/datasets/ct_log/phase2_m5_build`.
Build record: `dataset_manifests/phase2-m5.DATASET.json` (manifest sha256 `45115544…f218`).

Upload: `logs/phase2_deploy_20261009/upload_prelabels.py`. Every uploaded frame was re-downloaded and
matches the build object for object.

Backup of the previous (ensemble) pre-labels and the project meta:
`/mnt/D/datasets/ct_log/backups/phase2_prelabels_before_m5_20261015_1855/<log>/<image>.json`.

## 2026-10-16: Phase2 pre-labels replaced with M5, all classes from the model

Same unreviewed frames as before: logs 3 (298), 05 (299), 06 (299), 09 (300) and the 271 untagged frames of
08; the 27 `bumaska` frames of 08 were skipped. Before the upload every overwritten frame still matched the
2026-10-15 upload object for object.

Labels (builder defaults `--wood_source model --knot_rule argmax --pith_source model`): wood = largest filled
component of the M5 argmax in {wood, knot}, knots = argmax == knot, pith = M5 soft-argmax rounded, no snap.
Inputs: `scripts/export_model_outputs.py` on euler (bf16), `/mnt/D/datasets/ct_log/model_outputs_m5_euler`
(checkpoint sha256 `228ed105…88b0`). Build: `/mnt/D/datasets/ct_log/phase2_m5_model_build`, record
`dataset_manifests/phase2-m5-model.DATASET.json`.

Change vs the 2026-10-15 labels: wood −0.51 % to +0.07 % px per log, knots +0.05 % to +0.16 %.
Every uploaded frame was re-downloaded and matches the build object for object.

Backup of the previous pre-labels and the project meta:
`/mnt/D/datasets/ct_log/backups/phase2_20261016/<log>/<image>.json`.

## 2026-10-21: `bumaska` tag meta added to projects that lacked it

Reviewers refining in `ct-log-v5-preview` (384679) could not tick `bumaska`: the preview projects were created by
`logs/report_20261010/preview_upload.py` (Vostro, untracked) with `sly.ProjectMeta(obj_classes=...)` copied from
Phase2, which drops the tag metas. `upload_agreement.py` built its meta the same way. Phase2 and Human-collection
were never affected (tag id 32945609 unchanged since the 2026-10-09 backup, 1192 tagged frames).

`python -m ann_pipeline.utils.review_tag --fix` added the tag meta (no image or annotation changes) to 384366,
384578, 384679, 384683, 375492 and 376641. Run the audit (without `--fix`) after creating any project.
