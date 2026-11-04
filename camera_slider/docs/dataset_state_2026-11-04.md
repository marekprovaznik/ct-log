# CameraSlider dataset state (2026-11-04)

Source: Supervisely camera_slider_KAR (385745), read live; stitching checks from the CT-23 detector (`camera_slider.stitch_qa`) on KAR plus Matěj's `wrong_stitch` tag; boards from the dataset_v1 manifest.

- **Annotated**: 🔵 annotated by a human (`bumaska`) · ⚪ not human-annotated (model pre-labels only) · ➖ piece not in Supervisely
- **Stitching OK**: ✅ no problem found · 🟧 tagged `wrong_stitch` by Matěj · 🟨 flagged by the CT-23 detector only · 🟥 not in Supervisely (see Issues)
- P1-P4 are the four ~1 m pieces of a board, in order along the board.
- Not listed: 32 spruce boards and larch 6 that have an ESP CSV but no delivered frames (CT-4).
- 🟨 marks are candidates: the detector was checked on only 33 hand-labelled pieces (CT-23); for `face_shift` that is 3 examples (2 right, 1 false alarm, 1 missed), so some 🟨 may be fine and some ✅ may be wrong.

Pieces: 220 🔵, 60 ⚪, 64 ➖; 217 ✅, 11 🟧, 52 🟨, 64 🟥.

| Board | Annotated (P1 P2 P3 P4) | Stitching OK (P1 P2 P3 P4) | Issues |
|---|:---:|:---:|---|
| larch_003 | 🔵 🔵 ⚪ 🔵 | 🟨 ✅ ✅ ✅ | P1: edge_step |
| larch_004 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_005 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (rejected: incomplete frames) |
| larch_007 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_008 | 🔵 🔵 🔵 🔵 | 🟧 🟨 🟨 🟧 | P1: wrong_stitch tag, face_shift; P2: face_shift; P3: face_shift; P4: wrong_stitch tag, face_shift |
| larch_009 | 🔵 🔵 🔵 ⚪ | 🟨 🟨 🟨 🟧 | P1: face_shift; P2: face_shift; P3: face_shift; P4: wrong_stitch tag, face_shift, bad_seam |
| larch_010 | 🔵 🔵 🔵 🔵 | ✅ 🟨 🟨 ✅ | P2: bad_seam; P3: bad_seam |
| larch_011 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | P4: edge_step, bad_seam |
| larch_012 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_013 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_014 | 🔵 🔵 🔵 ⚪ | 🟧 ✅ 🟧 ✅ | P1: wrong_stitch tag; P3: wrong_stitch tag |
| larch_015 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_016 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (rejected: face 1: 11 uncovered columns) |
| larch_017 | 🔵 🔵 🔵 🔵 | 🟧 ✅ ✅ ✅ | P1: wrong_stitch tag |
| larch_018 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| larch_019 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_020 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| larch_021 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_022 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟧 | P4: wrong_stitch tag, edge_step |
| larch_023 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_024 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_025 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | P4: face_shift |
| larch_026 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| larch_027 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_028 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_029 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| larch_030 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| larch_031 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| larch_032 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| larch_033 | ⚪ ⚪ 🔵 ⚪ | 🟨 ✅ ✅ ✅ | P1: edge_step |
| larch_034 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| larch_035 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| larch_036 | 🔵 🔵 🔵 🔵 | 🟧 ✅ ✅ ✅ | P1: wrong_stitch tag |
| larch_037 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | P4: bad_seam |
| larch_038 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (rejected: face 1: 743 uncovered columns) |
| larch_039 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| spruce_001 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_002 | 🔵 ⚪ ⚪ ⚪ | ✅ ✅ ✅ ✅ |  |
| spruce_003 | ⚪ 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟧 | P4: wrong_stitch tag, stray_segment |
| spruce_004 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ ✅ | P1: face_shift |
| spruce_005 | 🔵 🔵 ⚪ 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_006 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_007 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_009 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_011 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_012 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_013 | 🔵 🔵 ⚪ 🔵 | ✅ ✅ ✅ 🟨 | P4: stray_segment |
| spruce_014 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_015 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_017 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No frame has at least 2 markers) |
| spruce_018 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_019 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No frame has at least 2 markers) |
| spruce_021 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (failed: No markers detected in any frame of the pass) |
| spruce_022 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | P4: stray_segment |
| spruce_023 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_025 | 🔵 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟨 | P4: stray_segment |
| spruce_026 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟧 | P4: wrong_stitch tag, stray_segment |
| spruce_028 | 🔵 🔵 🔵 🔵 | 🟧 ✅ ✅ ✅ | P1: wrong_stitch tag |
| spruce_029 | 🔵 🔵 ⚪ 🔵 | ✅ ✅ ✅ 🟨 | P4: stray_segment |
| spruce_031 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_032 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ 🟨 | P4: stray_segment |
| spruce_033 | 🔵 ⚪ ⚪ 🔵 | 🟨 🟨 🟨 🟨 | P1: face_shift; P2: face_shift; P3: face_shift; P4: face_shift |
| spruce_034 | ⚪ 🔵 ⚪ 🔵 | ✅ ✅ 🟨 🟨 | P3: face_shift; P4: stray_segment |
| spruce_035 | 🔵 🔵 🔵 🔵 | 🟨 🟨 🟨 🟨 | P1: face_shift; P2: face_shift; P3: face_shift; P4: face_shift |
| spruce_036 | 🔵 🔵 ⚪ 🔵 | 🟨 🟨 🟨 🟨 | P1: face_shift; P2: face_shift; P3: face_shift; P4: face_shift |
| spruce_037 | 🔵 🔵 🔵 🔵 | 🟨 ✅ ✅ 🟨 | P1: edge_step; P4: face_count |
| spruce_038 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_039 | ⚪ 🔵 🔵 ⚪ | ✅ ✅ ✅ 🟨 | P4: stray_segment |
| spruce_040 | 🔵 🔵 🔵 ⚪ | 🟨 ✅ ✅ ✅ | P1: stray_segment |
| spruce_042 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_043 | ⚪ 🔵 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| spruce_044 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ 🟨 | P4: stray_segment |
| spruce_045 | 🔵 ⚪ 🔵 ⚪ | 🟨 🟨 🟨 🟨 | P1: face_shift; P2: face_shift; P3: face_shift; P4: face_shift |
| spruce_046 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_048 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_049 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | built but not uploaded (one seam 0.32 mm, held back in CT-4) |
| spruce_050 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_051 | 🔵 🔵 ⚪ 🔵 | 🟨 ✅ ✅ ✅ | P1: stray_segment |
| spruce_053 | 🔵 ⚪ ⚪ 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_054 | ⚪ 🔵 ⚪ 🔵 | 🟨 🟨 🟨 🟨 | P1: stray_segment, face_shift; P2: face_shift; P3: face_shift; P4: edge_step, face_shift |
| spruce_055 | 🔵 ⚪ ⚪ 🔵 | 🟨 🟨 🟨 🟨 | P1: face_shift; P2: face_shift; P3: edge_step, face_shift; P4: edge_step, face_shift |
| spruce_061 | ➖ ➖ ➖ ➖ | 🟥 🟥 🟥 🟥 | not in Supervisely (rejected: face 2: 30791 uncovered columns) |
| spruce_062 | 🔵 🔵 🔵 ⚪ | 🟨 ✅ ✅ 🟨 | P1: stray_segment; P4: stray_segment |
| spruce_064 | 🔵 ⚪ 🔵 ⚪ | ✅ ✅ ✅ ✅ |  |
| spruce_100 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
| spruce_101 | 🔵 🔵 🔵 🔵 | ✅ ✅ ✅ ✅ |  |
