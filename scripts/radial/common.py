import base64
import hashlib
import io
import json
import pathlib
from typing import Dict, List, Optional, Tuple
import zlib

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

SIZE = 778
N_SLICES = 299
THIN_PX = 8.0
FOUND_IOU = 0.3
EIGHT = np.ones((3, 3), bool)
SRC_LOG = pathlib.Path("/mnt/D/datasets/ct_log/kwp_ds_v3/10")
SRC_PROB = pathlib.Path("/mnt/D/datasets/ct_log/knot_probs_ens_v4/10")
SRC_PRED = pathlib.Path("/mnt/D/datasets/ct_log/report_20261010/pred_m8")
OUT = pathlib.Path("/mnt/D/datasets/ct_log/radial/log10")


def page_name(z: int) -> str:
    return "page_%03d" % z


def read_png(path: pathlib.Path) -> np.ndarray:
    return np.array(Image.open(path))


def write_png(path: pathlib.Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, optimize=False, compress_level=6)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_bitmap(bitmap: Dict[str, object]) -> Tuple[np.ndarray, int, int]:
    raw = zlib.decompress(base64.b64decode(bitmap["data"]))
    mask = np.array(Image.open(io.BytesIO(raw)))
    if mask.ndim == 3:
        mask = mask[..., -1] if mask.shape[-1] == 4 else mask[..., 0]
    x0, y0 = bitmap["origin"]
    return mask > 0, int(x0), int(y0)


def read_annotation(path: pathlib.Path) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    ann = json.loads(path.read_text())
    label = np.zeros((SIZE, SIZE), np.uint8)
    pith = None
    for title, value in (("Wood", 1), ("Knot", 2)):
        for obj in ann["objects"]:
            if obj["classTitle"] == title and obj["geometryType"] == "bitmap":
                mask, x0, y0 = decode_bitmap(obj["bitmap"])
                h, w = mask.shape
                label[y0 : y0 + h, x0 : x0 + w][mask] = value
    for obj in ann["objects"]:
        if obj["classTitle"] == "Pith" and obj["geometryType"] == "point":
            x, y = obj["points"]["exterior"][0]
            pith = (float(x), float(y))
    return label, pith


def read_pred_npz(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    data = np.load(path)
    knot = np.unpackbits(data["knot"])[: SIZE * SIZE].reshape(SIZE, SIZE).astype(bool)
    wood = np.unpackbits(data["wood"])[: SIZE * SIZE].reshape(SIZE, SIZE).astype(bool)
    return knot, wood, (int(data["pith"][0]), int(data["pith"][1]))


def interpolate_pith(known: Dict[int, Tuple[float, float]], n: int) -> np.ndarray:
    zs = np.array(sorted(known))
    xy = np.array([known[z] for z in zs], np.float64)
    out = np.zeros((n, 2), np.float64)
    out[:, 0] = np.interp(np.arange(n), zs, xy[:, 0])
    out[:, 1] = np.interp(np.arange(n), zs, xy[:, 1])
    return out


def ray_coords(pith_xy: np.ndarray, theta_deg: float, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.deg2rad(theta_deg)
    r = np.arange(-radius, radius + 1, dtype=np.float64)
    cols = pith_xy[:, 0:1] + r[None, :] * np.cos(t)
    rows = pith_xy[:, 1:2] + r[None, :] * np.sin(t)
    return rows, cols


def sample_plane(volume: np.ndarray, rows: np.ndarray, cols: np.ndarray, order: int) -> np.ndarray:
    z = np.broadcast_to(np.arange(volume.shape[0], dtype=np.int64)[:, None], rows.shape)
    if order == 0:
        return gather(volume, z, np.rint(rows).astype(np.int64), np.rint(cols).astype(np.int64)).astype(volume.dtype)
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    wr = rows - r0
    wc = cols - c0
    vals = (
        (1 - wr) * (1 - wc) * gather(volume, z, r0, c0)
        + (1 - wr) * wc * gather(volume, z, r0, c0 + 1)
        + wr * (1 - wc) * gather(volume, z, r0 + 1, c0)
        + wr * wc * gather(volume, z, r0 + 1, c0 + 1)
    )
    return np.clip(np.rint(vals), 0, 255).astype(np.uint8)


def gather(volume: np.ndarray, z: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    inside = (rr >= 0) & (rr < volume.shape[1]) & (cc >= 0) & (cc < volume.shape[2])
    out = np.zeros(rr.shape, np.float64)
    out[inside] = volume[z[inside], rr[inside], cc[inside]]
    return out


def upsample_z(plane: np.ndarray, factor: int, linear: bool) -> np.ndarray:
    n = plane.shape[0]
    zq = np.arange((n - 1) * factor + 1, dtype=np.float64) / factor
    if not linear:
        return plane[np.rint(zq).astype(np.int64)]
    lo = np.floor(zq).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    w = (zq - lo)[:, None]
    vals = (1.0 - w) * plane[lo].astype(np.float32) + w * plane[hi].astype(np.float32)
    return np.clip(np.rint(vals), 0, 255).astype(plane.dtype)


def components(mask: np.ndarray) -> List[np.ndarray]:
    lab, n = ndi.label(mask, structure=EIGHT)
    return [lab == i for i in range(1, n + 1)]


def minor_axis(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(ys) < 2:
        return 0.0
    cov = np.cov(np.stack([xs, ys]).astype(np.float64), bias=True)
    return float(4.0 * np.sqrt(max(np.linalg.eigvalsh(cov)[0], 0.0)))


def match(gt: List[np.ndarray], pred: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    gt_iou = np.zeros(len(gt))
    gt_to_pred = np.full(len(gt), -1)
    if not gt or not pred:
        return gt_iou, gt_to_pred
    mat = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            inter = np.logical_and(g, p).sum()
            if inter:
                mat[i, j] = inter / np.logical_or(g, p).sum()
    ri, ci = linear_sum_assignment(-mat)
    for i, j in zip(ri, ci):
        if mat[i, j] > 0:
            gt_iou[i] = mat[i, j]
            gt_to_pred[i] = j
    return gt_iou, gt_to_pred
