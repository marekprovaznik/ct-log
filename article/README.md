# CT-log knot, wood and pith segmentation: facts and figures for the abstract

> **Warning: training-log images.** The per-knot IoU gallery has two folders.
> [`figures/iou_gallery/heldout_log10/`](figures/iou_gallery/heldout_log10/) is held-out log 10 (test data).
> [`figures/iou_gallery/training_logs/`](figures/iou_gallery/training_logs/) comes from logs 1, 2 and 4, which
> M5 was trained on: those images only show what an IoU value looks like and are **not test results**. All
> numbers below are on held-out log 10 unless stated otherwise.

Links: [meeting page](https://claude.ai/artifact/SugjP5D8V86jTLxP2CKJM3) (full history, interactive charts) ·
[MLflow on DagsHub](https://dagshub.com/AddictionLord/ct-log.mlflow/#/experiments) (all training runs; M5 is in
experiment `ct-log-kwp-teachersplit`, run `kwp-v4-unet-effv2s-w1-dice-ts-48ep-dec512`, run ID
`d23857a7e83041a0af89e1dc45e36c55`).

Numbers use a decimal point and a comma for thousands (author guidelines). Captions are drafts: Tab. captions go
above tables, Fig. captions below figures.

## Dataset

Tab. 1: CT data. A log is one scanned trunk section; a frame is one 778 x 778 px CT slice.

| | Logs | Frames |
|---|---:|---:|
| CT logs on disk | 54 | 16,171 (298-301 per log) |
| Logs in the Supervisely annotation project (Phase2) | 9 | 2,654 |
| Human-reviewed frames in Phase2 (`bumaska` tag), read 2026-10-31 | 8 | 1,867 |
| Human-annotated frames M5 was trained on (logs 1, 2, 4 and 27 frames of 08) | 4 | 899 |
| Model pre-labelled frames M5 was trained on (all training logs except 1, 2, 4) | 50 | 14,948 |
| Held-out test frames (log 10, human-annotated) | 1 | 293 |

M5 was trained on **fewer** human frames (899) than exist now (1,867). Logs 3, 05 and part of 06 were reviewed
after M5 was trained; M5 saw them only as model pre-labels.

Tab. 2: Phase2 logs, human-reviewed frames on 2026-10-31 and their role for M5.

| Log | Frames on disk | Phase2 images | Human-reviewed (`bumaska`) | Role for M5 |
|---|---:|---:|---:|---|
| 1 | 299 | 292 | 291 | training, human labels |
| 2 | 299 | 292 | 290 | training, human labels |
| 3 | 298 | 294 | 292 | training, pre-labels only (reviewed later) |
| 4 | 299 | 292 | 291 | training, human labels |
| 05 | 299 | 296 | 295 | training, pre-labels only (reviewed later) |
| 06 | 299 | 296 | 88 | training, pre-labels only (partly reviewed later) |
| 08 | 299 | 298 | 27 | training, 27 human frames, rest pre-labels |
| 09 | 300 | 300 | 0 | training, pre-labels only |
| 10 | 299 | 294 | 293 | **held out (test)** |
| Total | 2,691 | 2,654 | 1,867 | |

Sources: [data/facts.json](data/facts.json), made by `python -m article.facts` from the CT folder on disk, the
Supervisely Phase2 project (377328, read-only) and kwp-ds-v4 `DATASET.json` plus the M5 config
[`src/configs/train_kwp_v4_unet_effv2s_w1_dice_ts_48ep_dec512.yaml`](../src/configs/train_kwp_v4_unet_effv2s_w1_dice_ts_48ep_dec512.yaml).
Phase2 holds 292-300 images per log, a few fewer than on disk for most logs.

## Networks

- **YOLO-OBB detector** (YOLO11n-OBB, 2 classes: knot, pith): oriented boxes that seed propagation and give the
  pith point (box centre, later snapped to the darkest pixel).
- **MedSAM2** (SAM2-based medical video segmenter): propagates knot masks along the log from human anchor frames
  or detector seeds (PL1-PL3).
- **YOLO11n-seg** (640 / 800 px): per-slice knot instance segmentation trained on the human frames (PL4, PL5).
- **M1**: frozen DINOv3 ViT-L backbone at 320 px, light segmentation head, pith coordinate regression.
- **M2**: DINOv3 ViT-L at 784 px with the last four blocks fine-tuned, pith as a heatmap (soft-argmax).
- **M3**: U-Net with a ResNet-50 encoder, 800 px, soft Dice loss, pith heatmap channel.
- **M4**: probability average of two U-Nets (EfficientNetV2-S and ConvNeXt-S encoders).
- **M5** (default since 2026-10-16): one U-Net (segmentation_models_pytorch), EfficientNetV2-S encoder, decoder
  512-256-128-64-32, 48 epochs, EMA weights; input three neighbouring slices, 800 px; outputs background / wood /
  knot classes and a pith heatmap. One run gives knots, wood and pith.
- **nnU-Net v2 2D**: self-configuring U-Net trained from scratch on the same data as M4, external baseline (no
  pith output).

## Model development on held-out log 10

Tab. 3: Knot and pith results on held-out log 10 (293 human-annotated frames; PL1: 248 frames, its 45 anchor
frames excluded). Best value per column in bold; lower is better for `pith_mean_px_rounded`, higher for the rest.
Empty: not logged for that model.

| ID | When (2026) | Model | `knot_iou` | `mean_per_gt_iou` | `map50` | `map50_95` | `inst_f1_30` | `precision` | `recall` | `thin_recall` | `pith_mean_px_rounded` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PL1 | May | MedSAM2 from human anchors | | 0.390 | | | 0.726 | 0.902 | 0.608 | 0.062 | |
| PL2 | May | YOLO-OBB + MedSAM2 per slice | | 0.350 | | | 0.744 | 0.896 | 0.636 | 0.103 | |
| PL3 | Jul-Sep | Anchor-free OBB propagation | | 0.503 | | | 0.821 | **0.980** | 0.706 | 0.128 | |
| M1 | 20-25 Sep | DINOv3 frozen, 320 px | 0.416 | 0.266 | 0.203 | 0.040 | 0.521 | 0.520 | 0.522 | 0.026 | 5.802 |
| PL4 | 30 Sep | YOLO11n-seg 640 px | 0.745 | 0.690 | 0.853 | 0.471 | 0.909 | 0.914 | 0.904 | 0.718 | |
| M2 | 2-5 Oct | DINOv3 fine-tuned, 784 px | 0.676 | 0.612 | 0.771 | 0.343 | 0.865 | 0.838 | 0.893 | 0.641 | 0.993 |
| PL5 | 7 Oct | YOLO11n-seg 800 px + snapped pith | 0.750 | 0.711 | 0.874 | 0.485 | 0.932 | 0.930 | **0.934** | 0.795 | 0.443 |
| M3 | 7 Oct | U-Net ResNet-50 | 0.744 | 0.693 | 0.877 | 0.489 | 0.900 | 0.886 | 0.915 | 0.769 | **0.359** |
| M4 | 9 Oct | Two U-Nets averaged | 0.763 | 0.703 | 0.877 | 0.513 | 0.922 | 0.932 | 0.912 | 0.718 | 0.382 |
| nnU-Net | 11-12 Oct | nnU-Net v2 2D | **0.768** | 0.720 | 0.898 | **0.534** | 0.911 | | | 0.744 | |
| **M5** | 12-13 Oct | One U-Net, wide decoder | 0.763 | **0.728** | **0.911** | 0.527 | **0.933** | | | **0.846** | 0.366 |

Metric keys as logged:
- `knot_iou`: pixel IoU of the knot class pooled over all slices, inside the wood.
- `mean_per_gt_iou`: per-knot IoU. Knots are 8-connected regions per slice, matched one-to-one to predicted regions
  (Hungarian matching on IoU); the mean over human knots of the matched IoU, a knot matched below 0.3 or missed
  counting 0.
- `map50`, `map50_95`: COCO-style single-class AP at IoU 0.50 and averaged over 0.50:0.05:0.95, ranked by each
  predicted knot's mean probability.
- `inst_f1_30`, `precision`, `recall`: knot instances matched at IoU >= 0.3.
- `thin_recall`: recall at IoU >= 0.3 of thin knots (minor axis < 8 px); log 10 has 39 thin knots of 272. M5 finds
  33 of 39 (0.846).
- `pith_mean_px_rounded`: mean distance in px between the predicted pith (rounded to the pixel) and the human click.

Sources per row are recorded in [data/development.json](data/development.json) (made by
`python -m article.development`); the source files are copied to [data/sources/](data/sources/):
- PL1-PL3, M4 instance metrics: `instance_metrics.json`, key `<ID>.iou_0.3`.
- M1-M4, PL4, PL5 `knot_iou`, `map50`, `map50_95`, pith: `generalisation.json`, key `<ID>.splits.test` or
  `<ID>.splits.val=test`.
- nnU-Net: `nnunet.json`, keys `2d.10` and `mean_per_gt_iou_log10.2d`.
- M5: `ens_dec512.json`, key `log10.e1_ep48` (the M5 evaluation run on euler).
- `precision` and `recall` were not logged separately for nnU-Net and M5 (only `inst_f1_30`).
- The meeting page also lists `map50` / `map50_95` for PL1-PL3 (0.468 / 0.155, 0.327 / 0.067, 0.669 / 0.307); their
  source files are not in this repository.

M5 vs M4, paired 20-slice block bootstrap on log 10 (95 % interval; from `bootstrap_instance_e1.json`, which is on
the Vostro machine and was not re-checked here): `mean_per_gt_iou` +0.025 [+0.015, +0.038], `map50` +0.034 [+0.014,
+0.060], `map50_95` +0.014 [+0.001, +0.030], `thin_recall` +0.13 [+0.04, +0.27], `knot_iou` +0.001 [-0.007, +0.009].

**Wood** (M5: wood or knot class, largest component, holes filled; ground truth = human Wood ∪ Knot): IoU 0.992 on
log 10 (meeting page; recomputed here from a local M5 run: 0.9917).

Tab. 4: M5 pith error on held-out log 10 (293 slices), px between the rounded prediction and the human click.

| Mean | Exact pixel | <= 1 px | <= 2 px | 90th percentile | Max |
|---:|---:|---:|---:|---:|---:|
| 0.366 | 68.3 % | 95.6 % | 99.0 % | 1.00 | 8.06 |

Source: [data/pith_m5_log10.json](data/pith_m5_log10.json), from one float32 run of the M5 checkpoint
(`python -m scripts.export_model_outputs ... --fp32`, then `python -m article.pith_gallery`). This is the only source
for M5 pith numbers here and on the meeting page; Tab. 3 uses the same mean. Quote **0.37 px** in the text. Other M5
pith means in older files (0.357, 0.372, 0.383 px) come from separate evaluation runs whose predicted point differs
by 1 px on a few slices; do not mix them with this table.

## Figures

All figures are PNG, 80 mm wide at 300 dpi, Arial labels (8 pt), made with plotly.

**Per-knot IoU gallery**, `python -m article.iou_gallery`. One full 778 x 778 px slice per example with every knot
coloured: yellow overlap, magenta M5 only, cyan human only. Two variants per example: `<name>.png` without and
`<name>_box.png` with a white rectangle around the example knot. Bins IoU ≈ 0.1, 0.3, 0.5, 0.7, 0.9 as on the
meeting page; "thin" = minor axis < 8 px. Draft captions, IoU, slice and knot bounding box per image are in each
folder's `index.json`.

Tab. 5: Gallery examples on held-out log 10 ([figures/iou_gallery/heldout_log10/](figures/iou_gallery/heldout_log10/)).

| Bin | File | Slice | Knot | IoU |
|---|---|---:|---|---:|
| 0.1 | `iou_0.1_thick_1` | 297 | thick, 13.0 px | 0.07 |
| 0.3 | `iou_0.3_thick_1` | 246 | thick, 13.9 px | 0.24 |
| 0.5 | `iou_0.5_thick_1` | 240 | thick, 14.3 px | 0.50 |
| 0.5 | `iou_0.5_thin_1` | 295 | thin, 5.1 px | 0.52 |
| 0.7 | `iou_0.7_thick_1`, `_2` | 164, 195 | thick | 0.70, 0.70 |
| 0.7 | `iou_0.7_thin_1` … `_4` | 281, 111, 194, 79 | thin | 0.70, 0.71, 0.72, 0.73 |
| 0.9 | `iou_0.9_thick_1`, `_2` | 214, 210 | thick | 0.90, 0.90 |

On log 10 M5 has no thin knot near IoU 0.1 or 0.9, and no knot between IoU 0.25 and 0.35: the 0.3 example is the
log-10 knot closest to 0.3 (0.24). Log-10 per-knot IoU (`mean_per_gt_iou`, missed knot = 0) is 0.55 for thin knots
and 0.76 for the others (meeting page).

Draft caption, held-out example: *Fig. X: Held-out log 10, slice 240. M5 knot prediction vs. human annotation on
a full CT slice; per-knot IoU 0.50 for the marked knot. Yellow: overlap; magenta: M5 only; cyan: human only.*

Training-log examples ([figures/iou_gallery/training_logs/](figures/iou_gallery/training_logs/), 18 images, logs 1,
2, 4): **seen in training, they illustrate the metric, not a test result** (see the folder's README).

The IoUs are recomputed from a local float32 run of the same M5 checkpoint. 5 of the 29 meeting-page knots differ
from the meeting page by more than 0.01 (log 10 slices 297: 0.07 vs 0.08 and 79: 0.73 vs 0.72; log 2 slice 13:
0.10 vs 0.14; log 1 slice 259: 0.23 vs 0.27; log 4 slice 169: 0.89 vs 0.90); the captions use the recomputed value,
which matches the image.

**Pith gallery**, [figures/pith_gallery_m5_log10.png](figures/pith_gallery_m5_log10.png): *Fig. X: M5 pith on
held-out log 10, 41 x 41 px around the human click (x8): the 13 slices with error > 1 px, then 24 evenly spaced
slices. Green square: human click; red cross: M5.* This one is an overview sheet for reading, not print-sized.

## Reproducing

From the repository root (conda env `ct-log`):

```bash
python -m scripts.export_model_outputs --config src/configs/train_kwp_v4_unet_effv2s_w1_dice_ts_48ep_dec512.yaml \
    --checkpoint <M5 seg_head_epoch_47.pth> --name M5 --data_root <root with <log>/img/page_*.tiff> \
    --logs 10 1 2 4 --out_root <model_root> --batch_size 1 --fp32
python -m article.facts --phase2_tags <get_tag_info_from_supervisely JSON> --phase2_date <YYYY-MM-DD>
python -m article.development
python -m article.iou_gallery --model_root <model_root>
python -m article.pith_gallery
```

The M5 checkpoint is `seg_head_epoch_47.pth`, sha256 `228ed1059172e2e7eaaa6c996b45d9a25b03d9060c435bb1587be52127c688b0`.
