# CT Log

## Semi-automatic annotation options

Three pipelines, trading off human effort vs. quality:

### 1. Anchor-augmented propagation (best quality, ~20 anchors/log)

Human annotates ~20 knot-only anchor frames → MedSAM2 video propagator fills
the rest, seeded with YOLO-OBB detections at non-anchor frames. Two seed
variants:

- **Ellipse seeds (v2.2)**: slightly crisper shapes on softwood, fewer
  detections.
- **SAM point seeds (v2.3)**: natural shapes on both hardwood and softwood,
  +17% recall. Recommended universal recipe.

Post-filters clean up artifacts (size, pith exclusion, eccentricity, solidity,
fill_holes). Wood and pith are fully automatic.

### 2. Anchor-free OBB propagation (zero human effort)

YOLO-OBB detections → ellipse rasters → MedSAM2 video propagation in
overlapping windows. No human annotations at all. Works well on softwood.
Degrades to ellipse-shaped outputs on hardwood.

### 3. Per-frame YOLO-OBB + SAM2 (no propagation, no anchors)

YOLO-OBB detection → SAM2 image predictor per slice. Simplest pipeline, no
temporal propagation, no anchors. Lower frame coverage than propagation-based
options but every detection is high precision (conf=0.40 → precision 1.0).

### Recommendation

Option 1 (point seeds + 20 anchors) gives the best results on any wood type
and is the recommended deployment path. Option 2 is the zero-effort fallback
for softwood. Option 3 is the simplest code path but covers fewer frames.

## Auto-annotation default: all classes from M5 (decided 2026-10-16)

Pre-labels for knots, wood and pith come from one run of **M5**, a single smp U-Net trained on kwp-ds-v4
(teacher split: auto logs + human 1, 2, 4, 08; log 10 held out), no test-time augmentation:
- config `src/configs/train_kwp_v4_unet_effv2s_w1_dice_ts_48ep_dec512.yaml` (EfficientNetV2-S encoder,
  512-256-128-64-32 decoder, 48 epochs, EMA);
- checkpoint `ckpt_local/kwp_v4_unet_effv2s_w1_dice_ts_48ep_dec512/kwp-v4-unet-effv2s-w1-dice-ts-48ep-dec512_periodic/seg_head_epoch_47.pth`
  on euler (epoch 48 of 48), sha256 `228ed1059172…`; a copy is on Vostro at `/mnt/D/models/ct_log/m5/`.

Rules (the `build_kwp_dataset.py` defaults):
- **Knots** (`--knot_rule argmax`): 3-class argmax == knot.
- **Wood** (`--wood_source model`): argmax in {wood, knot}, largest component, holes filled.
- **Pith** (`--pith_source model`): the model's soft-argmax point, rounded to the pixel, no snap.

On held-out log 10 (native 778, human GT; `report_2026-10/m5_all_classes.json`, `bootstrap_instance_e1.json`):
- **Knots:** pixel IoU 0.763, level with the previous M4 ensemble of two v4 U-Nets (+0.001 [−0.007, +0.009]).
  Per-knot IoU is 0.728 vs 0.703 for M4 (+0.025 [+0.015, +0.038]), mAP50 +0.034, thin-knot recall +0.13.
  argmax vs softmax >= 0.5 differs by 165 px over 293 slices (metrics identical to 3 decimals).
- **Wood** (GT = human Wood ∪ Knot): IoU 0.9916 vs 0.9954 for `threshold_peel`, no slice below 0.95 for
  either. The human Wood was made by correcting peel pre-labels and keeps the bark shell inside Wood, which
  favours peel; the model trims the outer layer more tightly (it undershoots GT on train logs 1/2/4 by
  80–220 px per slice, peel overshoots by 130–260).
- **Pith:** mean 0.383 px, 67 % on the exact pixel, 95 % within 1 px (M4 ensemble 0.385; snapped OBB 0.443).

Producing the input: `scripts/export_model_outputs.py --config <cfg> --checkpoint <ckpt> --name M5
--data_root <root> --logs ... --out_root <model_root>` writes, from one model run per slice,
`<log>/page_XXX.cls.png` (argmax), `<log>/page_XXX.png` (knot probability, uint8) and `<log>.json` (pith).
The builder reads it with `--model_root`. DATASET.json records `wood_source`, `knot_rule`, `pith_source`
and the checkpoint sha256. Exports on euler use bf16 autocast; Vostro's GTX 1650 runs fp32, which moves a
few knot pixels and ~1 % of pith points by 1 px.

Fallbacks via flags:
- `--knot_rule prob` thresholds the knot probability (`--knot_threshold`, default 128) from `--knot_root`
  or `--model_root`; with neither, YOLO11n-seg @800 (`yolo11n_seg_v5_val10_800`, conf 0.25; provenance in
  `ann_pipeline/DETECTOR_PROVENANCE.md`).
- `--wood_source peel` (`threshold_peel`, see below) or `lcc` (`threshold_largest_cc`, t=30).
- `--pith_source model` reads `--pith_root` when given (e.g. `scripts/export_pith.py` or
  `scripts/export_ensemble_pith.py` output); `--pith_source obb` uses the snapped 2-class OBB pith.
- `--wood_source peel --knot_rule prob --knot_root <probs> --pith_root <pith>` reproduces the 2026-10-09
  (ensemble) and 2026-10-15 (M5) builds byte for byte; `--wood_source peel --knot_rule prob --pith_source obb`
  without roots is the older YOLO + OBB builder.

Phase2 history: the unreviewed frames (logs 3, 05, 06, 09 and the 271 untagged frames of 08) got the M4
ensemble on 2026-10-09, M5 knots + pith with peel wood on 2026-10-15, and all classes from M5 (the
current default, euler bf16 exports) on 2026-10-16 (backups and details in `ann_pipeline/PHASES.md`).
Supervisely previews for comparison: `ct-log-v4-ensemble-preview` (384366, M4), `ct-log-v5-preview`
(384679, M5 knots + peel wood), `ct-log-v5b-preview` (384683, all classes from M5).

## Detector: single 2-class OBB (knot + pith)

The recommended detector is one **2-class OBB model** (class 0 = knot, class 1 =
pith), replacing the older split of a single-class OBB knot model plus a
separate axis-aligned knot+pith model. Pith is emitted as a small square OBB
around each pith point.

Prep with `ann_pipeline/knot/data_prep_obb_2cls.py`, train `yolo11n-obb.pt`. On
a log-level holdout (val = whole held-out logs, e.g. 2+10) it beats the
two-model split on every metric:

| | knot mAP50 | pith mAP50 | pith median px err |
|---|---|---|---|
| two models (OBB knot + axis-aligned pith) | 0.906 | 0.942 | 1.52 |
| single 2-class OBB | 0.913 | 0.990 | 0.97 |

Pith localizes at the annotation floor (median 0.97px, 87% within 2px). Knot
AP degrades past IoU 0.75 (AP90≈0.06) because OBB boxes are fit to the knot
mask — fine for propagation seeding, which needs the seed on the knot, not a
pixel-tight trace. The old axis-aligned knot head was weak/unstable (0.35
mAP50) and is dropped.

**Retrain flow** (detectors read a local disk root, not Supervisely directly —
see `ann_pipeline/PHASES.md`): download reviewed frames → `data_prep_obb_2cls`
→ `knot/train`.

## Two-phase annotation workflow

Two Supervisely projects split the workflow:

- **Phase1** — all frames, ~10-15% annotated by humans (knots only). These
  serve as anchors for propagation.
- **Phase2** — all frames have annotations. Anchor frames copied from Phase1
  (tagged `bumaska`), rest auto-generated. Annotators correct auto output and
  tag corrected frames with `bumaska`.

The `bumaska` image-level tag tracks review progress: tagged = human-verified.

Every Supervisely project with Knot/Wood/Pith classes (Phase1 excepted) must define the `bumaska` tag meta,
or reviewers cannot tick it. When creating a project, copy the whole Phase2 meta or pass it through
`with_review_tag` (`ann_pipeline/utils/review_tag.py`); never build it from `obj_classes` alone. After any
project creation run `python -m ann_pipeline.utils.review_tag` (exits 1 if a project lacks the tag; `--fix`
adds it). On 2026-10-21 the previews 384366/384679/384683, agreement 384578 and SM_2025 375492/376641 lacked
it because `logs/report_20261010/preview_upload.py` (Vostro) copied only Phase2's classes.

Setup script: `python -m ann_pipeline.scripts.setup_phases --subset 4`

See `ann_pipeline/PHASES.md` for full details.

## Wood mask: iterative boundary peel

The default wood detector (`threshold_largest_cc`, t=30) overshoots into the
bark ring by ~1200px on average. `threshold_peel` (in
`ann_pipeline/wood/detectors.py`) fixes this by iteratively removing dark
boundary pixels (intensity < 60) — reduces bark overshoot to ~374px with
minimal wood loss. No model needed, ~11 it/s.

Peel is no longer the builder default (model wood is, since 2026-10-16); it stays available as
`--wood_source peel` and is what the Phase2 pre-labels uploaded up to 2026-10-15 used.

## Training runs and experiment tracking

Agents that launch, monitor or report training (locally or on euler.mendelu.cz) must follow
`TRAINING_AGENTS.md`. MLflow tracking is on DagsHub
(`https://dagshub.com/AddictionLord/ct-log.mlflow`, public); credentials live in `.env`.

### Monitoring long runs cheaply

Every wake-up of the main session re-sends its whole context, so do not poll with the model or
use periodic heartbeats. Watch a run with `scripts/euler_watch.sh` started as a background
command of the session that should react:

```bash
scripts/euler_watch.sh <job> <remote_run_script> <remote_train_info_yaml>
```

- Polling is a bash probe every 5 min (zero model tokens). The script exits, and so wakes the
  session, only when a decision is needed: FINISHED, FAILED (supervisor gave up), BUG, MLFLOW,
  STALL, UNREACHABLE, RECOVERY limit. Events are appended to `logs/euler_watch_<job>.log`.
- It recovers on its own from a container restart/recreation (re-stages the MLflow credentials
  from `.env` and relaunches the job, which resumes from `~/work`).
- A new traceback is triaged by a small model (`claude -p --model sonnet --tools ""`, headless,
  no tools, ~$0.02 and 4 s per call). TRANSIENT (OOM from another GPU user, network, signal) is
  left to the on-euler supervisor; BUG wakes the main session.
- Status questions ("jak je na tom trénink?") are answered with one probe on demand, not by
  keeping a heartbeat.
- Start each new experiment in a fresh session: context is the dominant token cost.
  `TRAINING_AGENTS.md` and `ann_pipeline/DETECTOR_PROVENANCE.md` carry the needed background.
