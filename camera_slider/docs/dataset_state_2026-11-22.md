# CameraSlider dataset state (2026-11-22)

Source: Supervisely camera_slider_v2 (387077), read live; stitching checks from the CT-23 detector (`camera_slider.stitch_qa`) on the pieces plus the `wrong_stitch` tag; boards from the dataset_v2 manifest.

- **Annotated**: 🔵 annotated by a human (`bumaska`) · ⚪ not human-annotated (model pre-labels only) · ➖ piece not in Supervisely
- **Stitching OK**: ✅ no problem found · 🟧 tagged `wrong_stitch` by Matěj · 🟨 flagged by the CT-23 detector only · 🟥 not in Supervisely (see Issues)
- P1-P4 are the four ~1 m pieces of a board, in order along the board.
- Not listed: 32 spruce boards and larch 6 that have an ESP CSV but no delivered frames (CT-4).
- **Face length diff**: longest minus shortest face of the board in mm (`face_length_diff_mm`) · 🟥 above 20 mm · ➖ not measured. Skewed end cuts alone give several mm.
- 🟨 marks are candidates: the detector was checked on only 33 hand-labelled pieces (CT-23); for `face_shift` that is 3 examples (2 right, 1 false alarm, 1 missed), so some 🟨 may be fine and some ✅ may be wrong.

Pieces: 222 🔵, 118 ⚪, 0 ➖; 267 ✅, 0 🟧, 73 🟨, 0 🟥.

Boards by face length diff: 0 🟥, 0 ➖.

| Board | Annotated (P1 P2 P3 P4) | Stitching OK (P1 P2 P3 P4) | Face length diff (mm) | Issues |
|---|:---:|:---:|:---:|---|
| larch_003 | 🔵 🔵 ⚪ 🔵 | 🟨 ✅ ✅ 🟨 | 3.9 | P1: bad_seam; P4: bad_seam |
| larch_004 | 🔵 🔵 🔵 🔵 | 🟨 🟨 🟨 🟨 | 1.7 | P1: edge_step; P2: bad_seam; P3: bad_seam; P4: edge_step, bad_seam |
| larch_007 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ 🟨 | 4.2 | P1: edge_step; P4: edge_step, bad_seam |
| larch_008 | 🔵 🔵 🔵 🔵 | 🟨 ✅ 🟨 🟨 | 0.2 | P1: bad_seam; P3: bad_seam; P4: edge_step, bad_seam |
| larch_009 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 0.8 | P4: edge_step |
| larch_010 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ ✅ | 5.2 | P1: edge_step |
| larch_011 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 4.7 | P4: edge_step |
| larch_012 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ 🟨 | 1.2 | P1: edge_step; P4: edge_step |
| larch_013 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 3.3 |  |
| larch_014 | 🔵 🔵 🔵 ⚪ | 🟨 ✅ ✅ 🟨 | 7.5 | P1: bad_seam; P4: bad_seam |
| larch_015 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 1.8 | P4: edge_step |
| larch_016 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ 🟨 | 1.2 | P4: edge_step |
| larch_017 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 1.9 | P4: edge_step |
| larch_018 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 3.0 | P4: edge_step, bad_seam |
| larch_019 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ 🟨 | 5.5 | P4: edge_step |
| larch_020 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 1.6 | P4: edge_step |
| larch_021 | 🔵 🔵 🔵 🔵 | ✅ 🟨 ✅ 🟨 | 6.7 | P2: bad_seam; P4: edge_step |
| larch_022 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 0.7 | P4: edge_step |
| larch_023 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 6.0 | P4: bad_seam |
| larch_024 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 0.5 | P4: edge_step |
| larch_025 | 🔵 🔵 🔵 🔵 | ✅ ✅ 🟨 🟨 | 5.4 | P3: bad_seam; P4: bad_seam |
| larch_026 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 0.1 | P4: bad_seam |
| larch_027 | 🔵 🔵 🔵 🔵 | ✅ ✅ 🟨 ✅ | 10.4 | P3: bad_seam |
| larch_028 | 🔵 🔵 🔵 🔵 | 🟨 ✅ 🟨 ✅ | 4.2 | P1: bad_seam; P3: bad_seam |
| larch_029 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 8.1 | P4: bad_seam |
| larch_030 | 🔵 🔵 🔵 ⚪ | 🟨 ✅ ✅ 🟨 | 9.6 | P1: bad_seam; P4: bad_seam |
| larch_031 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 7.0 | P4: bad_seam |
| larch_032 | 🔵 ⚪ 🔵 ⚪ | ✅ 🟨 ✅ 🟨 | 7.2 | P2: stray_segment; P4: edge_step, bad_seam |
| larch_033 | ⚪ ⚪ 🔵 ⚪ | 🟨 🟨 🟨 🟨 | 12.6 | P1: edge_step, bad_seam; P2: bad_seam; P3: bad_seam; P4: edge_step, bad_seam |
| larch_034 | 🔵 ⚪ 🔵 ⚪ | 🟨 ✅ ✅ 🟨 | 0.6 | P1: edge_step; P4: edge_step, bad_seam |
| larch_035 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 5.4 | P4: edge_step, bad_seam |
| larch_036 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 6.6 | P4: edge_step |
| larch_037 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 5.3 | P4: bad_seam |
| larch_038 | ⚪ ⚪ ⚪ ⚪ | 🟨 🟨 ✅ ✅ | 1.4 | P1: bad_seam; P2: stray_segment, bad_seam |
| larch_039 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 0.8 | P4: edge_step |
| spruce_001 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 6.2 |  |
| spruce_002 | 🔵 ⚪ ⚪ ⚪ | ✅ ✅ ✅ 🟨 | 7.6 | P4: bad_seam |
| spruce_003 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ | 5.9 |  |
| spruce_004 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 7.9 | P4: edge_step, bad_seam |
| spruce_005 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 6.1 |  |
| spruce_006 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 10.4 |  |
| spruce_007 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 2.8 |  |
| spruce_009 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ 🟨 | 5.0 | P4: bad_seam |
| spruce_011 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 6.9 |  |
| spruce_012 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 4.5 |  |
| spruce_013 | 🔵 🔵 ⚪ 🔵 | ✅ ✅ ✅ ✅ | 7.3 |  |
| spruce_014 | ⚪ ⚪ ⚪ ⚪ | 🟨 ✅ ✅ 🟨 | 9.0 | P1: edge_step; P4: bad_seam |
| spruce_015 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ 🟨 | 13.3 | P4: edge_step |
| spruce_017 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 2.6 |  |
| spruce_018 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 7.8 |  |
| spruce_019 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 3.9 |  |
| spruce_021 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ 🟨 | 4.4 | P4: edge_step |
| spruce_022 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 6.0 |  |
| spruce_023 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ ✅ | 7.5 | P1: bad_seam |
| spruce_025 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ | 5.1 |  |
| spruce_026 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ ✅ | 8.2 | P1: bad_seam |
| spruce_028 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 6.0 |  |
| spruce_029 | 🔵 🔵 ⚪ 🔵 | ✅ ✅ ✅ ✅ | 4.3 |  |
| spruce_031 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ ✅ | 1.4 |  |
| spruce_032 | 🔵 ⚪ 🔵 ⚪ | 🟨 ✅ ✅ ✅ | 11.6 | P1: bad_seam |
| spruce_033 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ ✅ | 3.7 |  |
| spruce_034 | ⚪ 🔵 ⚪ 🔵 | ✅ ✅ ✅ ✅ | 5.4 |  |
| spruce_035 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 6.1 |  |
| spruce_036 | 🔵 🔵 ⚪ 🔵 | ✅ ✅ ✅ ✅ | 3.5 |  |
| spruce_037 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | 13.4 | P4: bad_seam |
| spruce_038 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 7.4 |  |
| spruce_039 | ⚪ 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟨 | 12.0 | P4: bad_seam |
| spruce_040 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ | 2.8 |  |
| spruce_042 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ 🟨 | 12.0 | P4: bad_seam |
| spruce_043 | ⚪ 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ | 7.0 |  |
| spruce_044 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 2.2 |  |
| spruce_045 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ ✅ | 2.7 |  |
| spruce_046 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ ✅ | 4.7 | P1: bad_seam |
| spruce_048 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 4.1 |  |
| spruce_049 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 14.4 |  |
| spruce_050 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ ✅ | 9.1 | P1: edge_step |
| spruce_051 | 🔵 🔵 ⚪ 🔵 | ✅ ✅ ✅ 🟨 | 9.9 | P4: bad_seam |
| spruce_053 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ ✅ | 4.2 |  |
| spruce_054 | ⚪ 🔵 ⚪ 🔵 | ✅ ✅ ✅ ✅ | 5.6 |  |
| spruce_055 | 🔵 ⚪ ⚪ 🔵 | ✅ 🟨 🟨 🟨 | 7.7 | P2: edge_step; P3: edge_step; P4: edge_step |
| spruce_061 | ⚪ ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ | 9.1 |  |
| spruce_062 | 🔵 🔵 🔵 ⚪ | 🟨 ✅ ✅ ✅ | 5.6 | P1: bad_seam |
| spruce_064 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ ✅ | 5.5 |  |
| spruce_100 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 11.0 |  |
| spruce_101 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ | 5.7 |  |
