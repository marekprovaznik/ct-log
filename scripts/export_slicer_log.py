import argparse
import base64
import gzip
import io
import json
import logging
import pathlib
from typing import Dict, List, Tuple
import zlib

import numpy as np
from PIL import Image

LABELS: Dict[str, Tuple[int, Tuple[float, float, float]]] = {
    "Wood": (1, (0.85, 0.65, 0.4)),
    "Knot": (2, (0.95, 0.85, 0.1)),
}
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one CT log and its Supervisely labels for 3D Slicer.")
    parser.add_argument("--log_dir", type=pathlib.Path, required=True, help="kwp_ds_v3/<log> with img/ and ann/.")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--name", default="log")
    parser.add_argument("--slice_mm", type=float, default=10.0)
    parser.add_argument("--pixel_mm", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pages = sorted(p.name for p in (args.log_dir / "img").glob("page_*.tiff"))
    ct, labels, pith, labelled = build_volumes(args.log_dir, pages)
    spacing = (args.pixel_mm, args.pixel_mm, args.slice_mm)
    args.out.mkdir(parents=True, exist_ok=True)
    write_nrrd(args.out / ("%s_ct.nrrd" % args.name), ct, spacing, {})
    write_nrrd(args.out / ("%s_labels.seg.nrrd" % args.name), labels, spacing, segment_header(labels))
    write_markups(args.out / ("%s_pith.mrk.json" % args.name), pith, spacing)
    meta = {
        "pages": pages,
        "labelled_pages": labelled,
        "spacing_mm_xyz": spacing,
        "labels": {k: v[0] for k, v in LABELS.items()},
        "note": "Unlabelled slices are 0 in the segmentation; pith = one control point per labelled slice.",
    }
    (args.out / ("%s_meta.json" % args.name)).write_text(json.dumps(meta, indent=1))
    logger.info("%d slices, %d labelled -> %s", len(pages), len(labelled), args.out)


def build_volumes(
    log_dir: pathlib.Path, pages: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, float, float, str]], List[str]]:
    first = read_ct(log_dir / "img" / pages[0])
    ct = np.zeros((len(pages), *first.shape), np.uint8)
    labels = np.zeros_like(ct)
    pith: List[Tuple[int, float, float, str]] = []
    labelled: List[str] = []
    for z, page in enumerate(pages):
        ct[z] = read_ct(log_dir / "img" / page)
        ann_path = log_dir / "ann" / (page + ".json")
        if not ann_path.exists():
            continue
        labelled.append(page)
        ann = json.loads(ann_path.read_text())
        for title in ("Wood", "Knot"):
            for obj in ann["objects"]:
                if obj["classTitle"] == title and obj["geometryType"] == "bitmap":
                    paste_bitmap(labels[z], obj["bitmap"], LABELS[title][0])
        for obj in ann["objects"]:
            if obj["classTitle"] == "Pith" and obj["geometryType"] == "point":
                x, y = obj["points"]["exterior"][0]
                pith.append((z, float(x), float(y), page))
    return ct, labels, pith, labelled


def read_ct(path: pathlib.Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    return arr[..., 0] if arr.ndim == 3 else arr


def paste_bitmap(target: np.ndarray, bitmap: Dict[str, object], value: int) -> None:
    raw = zlib.decompress(base64.b64decode(bitmap["data"]))
    mask = np.array(Image.open(io.BytesIO(raw)))
    if mask.ndim == 3:
        mask = mask[..., -1] if mask.shape[-1] == 4 else mask[..., 0]
    x0, y0 = bitmap["origin"]
    h, w = mask.shape
    target[y0 : y0 + h, x0 : x0 + w][mask > 0] = value


def segment_header(labels: np.ndarray) -> Dict[str, str]:
    header: Dict[str, str] = {}
    for i, (name, (value, color)) in enumerate(LABELS.items()):
        zs, ys, xs = np.nonzero(labels == value)
        extent = (
            "%d %d %d %d %d %d" % (xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max())
            if len(zs)
            else "0 -1 0 -1 0 -1"
        )
        header.update(
            {
                "Segment%d_ID" % i: name,
                "Segment%d_Name" % i: name,
                "Segment%d_LabelValue" % i: str(value),
                "Segment%d_Layer" % i: "0",
                "Segment%d_Color" % i: "%.3f %.3f %.3f" % color,
                "Segment%d_Extent" % i: extent,
                "Segment%d_Tags" % i: "",
            }
        )
    header["Segmentation_MasterRepresentation"] = "Binary labelmap"
    header["Segmentation_ContainedRepresentationNames"] = "Binary labelmap|"
    return header


def write_nrrd(
    path: pathlib.Path, volume: np.ndarray, spacing: Tuple[float, float, float], extra: Dict[str, str]
) -> None:
    nz, ny, nx = volume.shape
    lines = [
        "NRRD0004",
        "type: uint8",
        "dimension: 3",
        "space: left-posterior-superior",
        "sizes: %d %d %d" % (nx, ny, nz),
        "space directions: (%g,0,0) (0,%g,0) (0,0,%g)" % spacing,
        "kinds: domain domain domain",
        "encoding: gzip",
        "space origin: (0,0,0)",
    ]
    lines += ["%s:=%s" % (k, v) for k, v in extra.items()]
    with open(path, "wb") as fh:
        fh.write(("\n".join(lines) + "\n\n").encode())
        fh.write(gzip.compress(np.ascontiguousarray(volume).tobytes(), compresslevel=6))


def write_markups(
    path: pathlib.Path, pith: List[Tuple[int, float, float, str]], spacing: Tuple[float, float, float]
) -> None:
    points = [
        {
            "id": str(i + 1),
            "label": page.replace(".tiff", ""),
            "position": [x * spacing[0], y * spacing[1], z * spacing[2]],
            "locked": True,
        }
        for i, (z, x, y, page) in enumerate(pith)
    ]
    doc = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "name": "Pith (human)",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "controlPoints": points,
                "display": {"glyphScale": 1.0, "pointLabelsVisibility": False},
            }
        ],
    }
    path.write_text(json.dumps(doc))


if __name__ == "__main__":
    main()
