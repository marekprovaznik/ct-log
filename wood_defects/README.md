# Wood surface defects: label viewer and stitcher

Tools for looking at a line-scan wood surface defect dataset with its labels (see [Data](#data)). Run from the
repo root.

```bash
python -m wood_defects.visualise --images <images> [--segmentation <semantic_maps>] [--bboxes <bboxes>] \
    [--output <dir>] [--scale 0.5] [--ids 100000000 ...] [--limit N]
python -m wood_defects.stitch --images <images> [--segmentation <semantic_maps>] [--bboxes <bboxes>] \
    --output <dir>
```

The viewer fills each semantic-map segment in its spec colour and draws each box with its class name. With
both overlays only the boxes carry class names; with `--segmentation` alone each segment is labelled. With
`--output` it writes `<id>.png`; without it, each image opens in plotly.

The stitcher joins consecutive images along the board (y) and writes the same layout as the dataset
(`images/`, `semantic_maps/<run>_segm.png`, `bboxes/<run>_anno.txt`), so the viewer reads its output. Use
`--scale 0.25` to view stitched runs.

## Label format (checked on the local 100-image subset)

- Images are 2800×1024 (W×H); x is across the board, y along it. A few images are narrower (2766–2799 px)
  while their labels keep the 2800 px frame; both tools pad such images with black on the right.
- Bbox files: `<class>\t<x1>\t<y1>\t<x2>\t<y2>`, normalised to 2800×1024. The order was confirmed by
  drawing the boxes over the semantic maps.
- Semantic maps: RGB, black background, one spec colour per class.
- Class names differ between the spec and the bbox files; both map to one table in `labels.py`:
  `live_knot`, `dead_knot` (`Death_know`, `Dead_Knot`), `knot_missing`, `knot_with_crack`, `crack`,
  `quartzite` (`Quartzity`), `resin`, `marrow`, `blue_stain`, `overgrown`.

## Stitching

- Consecutive chunks follow each other with no missing or repeated lines, but each chunk is cropped from the
  line-scan stream at its own horizontal offset, so the board jumps sideways by a few to ~45 px at every seam.
  The stitcher finds that shift per seam (the x-shift minimising the mean absolute difference between the last
  row of one chunk and the first row of the next, within `--max_shift_px`) and places each chunk on a wider
  black canvas; masks and boxes get the same shift. The log reports the shifts and the remaining seam
  difference, which on the subset equals the difference between neighbouring rows inside an image.
- IDs are `<scan prefix><5-digit index>`. A run is a maximal sequence of consecutive indices within one
  prefix, named `<first>-<last>`; missing indices start a new run.
- A defect crossing a chunk boundary is labelled with one box per chunk. Boxes of the same class touching the
  seam from both sides with overlapping x ranges are merged into one; the log reports merges, seam boxes
  without a partner, and touching pairs with different classes (not merged).

## Data

The tools are built for the wood surface defect dataset by Pavel Kodytek, Alexandra Bodzas and Petr Bilik
(VSB - Technical University of Ostrava), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- P. Kodytek, A. Bodzas, P. Bilik. A large-scale image dataset of wood surface defects for automated
  vision-based quality control processes. F1000Research 10:581, 2022.
- Dataset: [doi:10.5281/zenodo.4694695](https://doi.org/10.5281/zenodo.4694695).

Outputs of these tools are modified versions of that data: images, semantic maps and boxes are stitched into
runs, boxes split by a seam are merged, and class names are normalised.
