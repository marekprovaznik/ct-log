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
   different height than the strip (`k` = strip distance / wood distance). Soft-L1 loss, then observations
   above `--outlier_px` are dropped and the problem is re-solved. Frames without usable markers are placed
   by their wood tie points alone.
5. **Board frame.** The strip axis becomes +x with marker IDs increasing along it, so faces from passes in
   opposite directions share the same board-end orientation; x = 0 is marker 0.
6. **Render.** Frames are warped at the median marker size (px per marker side, the same for all faces of
   a board) and feather-blended. The timber is segmented from the mosaic (wood colour mask, closing,
   largest component, holes filled), cropped, and the background zeroed.

## Outputs (`<out_dir>/<board_id>/`)

| File | Content |
|---|---|
| `face_<n>.jpg` | Face `n`, timber only |
| `face_<n>_mask.png` | Timber mask of the face |
| `unrolled.jpg` | All faces stacked, aligned on board x |
| `preview.jpg` | Downscaled `unrolled.jpg` |
| `report.json` | Registration quality, see below |

`report.json`, per face: `height_scale_k`, `seam_median_px` / `seam_max_px` (median wood tie-point
disagreement between overlapping frames, in output px), `gaps` and `n_missing_estimate` (steps longer than
1.5× the median step), `file_number_gaps`, `marker_ids_not_used`, `uncovered_columns` (board columns no
frame covers) and `flagged_frames`. Per frame: `placed_by` (`markers`, `features`, or null when unplaced),
marker reprojection error, `outlier_markers` and `seam_px`. A frame is flagged when it is unplaced or its
worst seam exceeds `--max_seam_px`.

## Assumptions and limits

- Each face is planar and roughly parallel to the slider; `k` is one scale per face, about the image centre.
- The ESP log is not read; passes come from EXIF times, and one directory holds one board.
- Lens distortion is not modelled.
- Rendering a ~35k px long face at native resolution peaks at ~7 GB RAM.
