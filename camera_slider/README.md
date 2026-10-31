# CameraSlider stitcher

Turns the raw frames of one board (one pass per face) into an unrolled board surface: each face as one
image at a common scale, and all faces stacked along the board length. Raw frames are only read; every
output can be rebuilt from them.

```bash
python -m camera_slider.stitch --frames_dir <board_frames_dir> --out_dir <derived_root> [--marker_mm 20]
```

`--n_faces` (default 4) is the number of passes expected; `--allow_partial` stitches whatever was found.
`--marker_mm` adds an absolute px/mm to the report; `--px_per_mm` with it sets the output resolution.

## How it works

1. **Passes.** Frames are sorted by EXIF capture time and split into passes where the gap exceeds
   `--pass_gap_s` (the slider waits ~5 s and the board turns between passes).
2. **Markers.** ArUco `DICT_7X7_1000` markers on the strip are detected with sub-pixel corners. Pixel data
   is read with EXIF orientation ignored.
3. **Wood tie points.** SIFT keypoints on the wood only (HSV mask) are matched between each frame and its
   next two neighbours, with a RANSAC similarity check.
4. **Joint bundle adjustment** per pass ([registration.py](registration.py)): one homography per frame
   (image → board plane), the pose of every marker (square of unit side, so the strip sets the metric
   scale), and one height scale `k` per face. Marker corners and wood tie points share the frame
   homographies; wood points are first scaled about the image centre by `1/k`, which models wood lying at a
   different height than the strip (`k` = strip distance / wood distance). Radial lens distortion
   (`k1`, `k2`, shared by all frames of the pass, about the image centre, radius normalised by the half
   diagonal) is estimated in the same problem and applied to every observation before the homography;
   `--no_distortion` turns it off. Soft-L1 loss, then observations above `--outlier_px` are dropped and the
   problem is re-solved. Frames without usable markers are placed by their wood tie points alone.
5. **Board frame.** The strip axis becomes +x with marker IDs increasing along it, so faces from passes in
   opposite directions share the same board-end orientation; x = 0 is marker 0.
6. **Render.** Frames are undistorted, warped at the median marker size (px per marker side, the same for
   all faces of a board) and feather-blended.
7. **Board mask** ([wood.py](wood.py), full resolution). Wood colour mask, closing, largest component,
   holes filled; then per column the longest wood run. Near the top and bottom edge the run is cut at a
   near-black gap when that brings the edge closer to the board's edge line (median over ~8000 px), which
   drops the side face seen at an angle while keeping dark wood at the edge. Edge points further than 20 px
   from that line are interpolated from their neighbours, and both edges are smoothed.
8. **Layout.** Faces are stretched to their median length (`--no_stretch_faces` centres them instead), so
   the board ends line up, and stacked with `--gap_px` between faces and `--margin_px` around the board.

## Outputs (`<out_dir>/<board_id>/`)

| File | Content |
|---|---|
| `face_<n>.jpg` | Face `n`, timber only |
| `face_<n>_mask.png` | Timber mask of the face |
| `unrolled.jpg` | All faces stacked, aligned on the board ends |
| `preview.jpg` | Downscaled `unrolled.jpg` |
| `report.json` | Registration quality, see below |

`report.json`, per face: `height_scale_k`, `distortion_k1` / `distortion_k2`, `seam_median_px` / `seam_max_px` (median wood tie-point
disagreement between overlapping frames, in output px), `gaps` and `n_missing_estimate` (steps longer than
1.5× the median step), `file_number_gaps`, `marker_ids_not_used`, `uncovered_columns` (board columns no
frame covers) and `flagged_frames`. Per frame: `placed_by` (`markers`, `features`, or null when unplaced),
marker reprojection error, `outlier_markers` and `seam_px`. A frame is flagged when it is unplaced or its
worst seam exceeds `--max_seam_px`.

## Dataset builder

Builds the annotation dataset from whole scan folders: every board is matched to its ESP CSV, stitched,
laid out as one unrolled image and cut into pieces for Supervisely.

```bash
python -m camera_slider.dataset \
    --scan_root /mnt/D/datasets/camera_slider/Optical_scan_Marek /mnt/D/datasets/camera_slider/raw/board_64 \
    --out_root /mnt/D/datasets/camera_slider/derived/dataset_v1 \
    --labels SKEN_2810=spruce sken_3010=larch --workers 16
```

1. **Index** ([scan_index.py](scan_index.py)). All frames under the scan roots, sorted by EXIF time with
   subseconds (IMG numbers wrap, and the same file name occurs in both blocks). A gap over `--board_gap_s`
   starts a new board, over `--pass_gap_s` a new pass.
2. **Match boards to CSVs.** Frame segments are aligned in time order with every session number of a
   block (a number without a CSV is an aborted run); block order and session order are searched jointly.
   The CSV file time (camera clock = ESP + 1 h, CSV written 20–80 s after the last frame) is a cross-check
   only (`mtime_check` in `index.csv`). `board_ids.jpg` shows each board's first frame with its assigned ID
   for a visual check against the handwritten number.
3. **Frames to shots** ([shots.py](shots.py)). The camera position from the markers is fitted to the CSV
   `pos_mm` per block (rail pitch in mm per marker ID), and each frame gets its shot index, delay and
   position; skipped shots and late frames are counted per pass.
4. **Stitch** each pass as above; faces and directions come from the CSV, layout `ADCB` for 4 faces and
   `AB` for 2.
5. **Reject** a board (no pieces, status `rejected: ...` in `manifest.csv`) when its CSV match is not
   `ok`, a face has uncovered columns, or a seam exceeds `--max_reject_seam_px`. Exceptions give
   `failed: ...`. `--skip_existing` resumes a run.

Outputs under `--out_root`:

| Path | Content |
|---|---|
| `index.csv` | Every segment and CSV with its match and reason |
| `board_ids.jpg` | Contact sheet of first frames with assigned board IDs |
| `cache/` | Marker detections per session |
| `boards/<label>_<board>/` | `face_<area>.jpg` + mask, `unrolled.jpg` (`.tif` above 65500 px), `preview.jpg`, `report.json` with the frame-to-shot table; `render_<area>.jpg` with `--save_renders` |
| `manifest.csv` | One row per board: status, seams, height k, px/mm, shots, pieces |
| `supervisely/<project>/boards/` | `img/<label>_<board>_x<start>.jpg` pieces (`--pieces`, `--overlap_px`) with empty `ann/` |

Each unrolled board keeps the native resolution of its frames (spruce about 7.7 px/mm, larch about
13.7 px/mm). Upload the pieces with
`python -m camera_slider.upload_supervisely --project_dir <out_root>/supervisely/camera_slider`; it
creates the project with the classes and review tag of the meta project and uploads only missing pieces.

## Data

Scans by Matěj Šimek, 2026-10-28 (spruce) and 2026-10-30 (larch). The delivery is complete; there are no
further frames.

| Source | Content |
|---|---|
| `Optical_scan_Marek/SKEN_2810` | Spruce, 4 faces, 46 frames per face, 2976×1984; frame folders `28vcetne`, `3`–`6`, `zmatek do 28_tusmazat` (holds frames of real boards, so it is included), `kalibrace smrk`; one CSV per board |
| `Optical_scan_Marek/sken_3010` | Larch, 2 faces, 46 frames per face, 6000×4000; folders `1`, `2`, `kalibrace modrin` |
| `raw/board_64` | Frames of spruce board 64, delivered separately; its CSV is `SKEN_2810/64.csv` |

With both scan roots the index has 50 spruce and 35 larch boards matched `ok` (plus the 2 calibration
runs). 32 spruce CSVs and the CSV of larch 6 have no frames; larch 5 has only 56 frames
(`incomplete frames`); one aborted spruce run (session 62, 55 frames) has no CSV. Three matched boards
disagree with the CSV file time (spruce_014, spruce_035, larch_007; one frame short each) and are worth a
look in `board_ids.jpg`.

## Assumptions and limits

- Each face is planar and roughly parallel to the slider; `k` is one scale per face, about the image centre.
- The ESP log is not read; passes come from EXIF times, and one directory holds one board.
- Lens distortion is radial only (no tangential terms or decentring). On spruce_050 (50 mm lens) the four
  passes give k1 between -0.0014 and -0.0007, under 3 px at the frame corners, and the seams are unchanged.
- Rendering a ~35k px long face at native resolution peaks at ~7 GB RAM.
- The strip's black number band touches the marker borders, so only part of the visible markers are
  detected; wood tie points carry the frames in between. On larch most marker observations are dropped as
  outliers (182 of 263 on larch_020 face A, spruce 30–80 per face) while the seams stay under 1.5 px.
