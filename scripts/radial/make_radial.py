import argparse
import json
import logging
import math
import pathlib
from typing import Dict, List, Tuple

import numpy as np
from scripts.radial.common import (
    N_SLICES,
    OUT,
    page_name,
    ray_coords,
    read_png,
    sample_plane,
    upsample_z,
    write_png,
)

logger = logging.getLogger(__name__)
Z_FACTOR = 10
PAD_PX = 10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Straightened radial (longitudinal) planes through the pith of log 10."
    )
    parser.add_argument("--root", type=pathlib.Path, default=OUT)
    parser.add_argument("--deg_step", type=float, default=2.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    knots = json.loads((args.root / "knots.json").read_text())
    gt_pith = np.array(knots["gt_pith_xy"])
    m4_pith = np.array(knots["m4_pith_xy"])
    ct, gt, prob, mask = load_volumes(args.root)
    radius = int(math.ceil(max_wood_radius(gt, gt_pith))) + PAD_PX
    planes = plane_list(knots["knots"], args.deg_step)
    logger.info("R = %d px, %d planes (%d knot-aligned)", radius, len(planes), sum(p["knot_ids"] != [] for p in planes))

    out_dir = args.root / "radial"
    for i, plane in enumerate(planes):
        theta = plane["theta_deg"]
        rows, cols = ray_coords(gt_pith, theta, radius)
        layers = {
            "ct": (sample_plane(ct, rows, cols, 1), True),
            "gt": (sample_plane(gt, rows, cols, 0), False),
            "m4_prob": (sample_plane(prob, rows, cols, 1), True),
            "m4_mask": (sample_plane(mask, rows, cols, 0), False),
        }
        rows, cols = ray_coords(m4_pith, theta, radius)
        layers.update(
            {
                "m4axis/ct": (sample_plane(ct, rows, cols, 1), True),
                "m4axis/m4_prob": (sample_plane(prob, rows, cols, 1), True),
                "m4axis/m4_mask": (sample_plane(mask, rows, cols, 0), False),
            }
        )
        for name, (arr, linear) in layers.items():
            write_png(out_dir / plane["dir"] / ("%s.png" % name), arr)
            write_png(out_dir / plane["dir"] / ("%s_x10.png" % name), upsample_z(arr, Z_FACTOR, linear))
        if i % 50 == 0:
            logger.info("%d / %d", i, len(planes))

    failures = verify(out_dir, knots["knots"], planes, gt, gt_pith, radius)
    index = {
        "radius_px": radius,
        "width_px": 2 * radius + 1,
        "n_rows_native": N_SLICES,
        "n_rows_x10": (N_SLICES - 1) * Z_FACTOR + 1,
        "geometry": {
            "column j": "signed distance r = j - R px from the slice's pith along direction theta; r < 0 is theta + 180",
            "row z": "native: row z = slice z (10 mm); _x10: row k = slice k / 10, so row 10 z = slice z exactly",
            "sample point": "x = pith_x(z) + r cos(theta), y = pith_y(z) + r sin(theta); theta 0 = +x, 90 = +y (down in the image)",
            "axis": "gt-axis files (ct, gt, m4_*) use the human pith, linearly interpolated over unlabelled pages and held constant at the ends; m4axis/ uses M4's pith",
            "interpolation": "ct and m4_prob bilinear in-plane, linear in z for _x10; gt and m4_mask nearest in-plane and in z",
            "gt values": "0 background (also outside the image and on unlabelled pages 000-002, 100, 291, 298), 1 wood, 2 knot",
        },
        "verification": failures,
        "planes": planes,
    }
    (out_dir / "planes.json").write_text(json.dumps(index, indent=1))


def load_volumes(root: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ct = np.zeros((N_SLICES, 778, 778), np.uint8)
    gt = np.zeros_like(ct)
    prob = np.zeros_like(ct)
    mask = np.zeros_like(ct)
    for z in range(N_SLICES):
        page = page_name(z)
        ct[z] = read_png(root / "images" / ("%s.png" % page))
        ann = root / "annotations" / ("%s.png" % page)
        if ann.exists():
            gt[z] = read_png(ann)
        prob[z] = read_png(root / "predictions" / "m4" / ("%s_prob.png" % page))
        mask[z] = read_png(root / "predictions" / "m4" / ("%s_mask.png" % page))
    return ct, gt, prob, mask


def max_wood_radius(gt: np.ndarray, pith: np.ndarray) -> float:
    best = 0.0
    for z in range(gt.shape[0]):
        ys, xs = np.nonzero(gt[z])
        if len(ys):
            best = max(best, float(np.hypot(xs - pith[z, 0], ys - pith[z, 1]).max()))
    return best


def plane_list(knots: List[Dict[str, object]], deg_step: float) -> List[Dict[str, object]]:
    planes: Dict[str, Dict[str, object]] = {}
    for theta in np.arange(0.0, 180.0, deg_step):
        key = "theta_%05.1f" % theta
        planes[key] = {"dir": key, "theta_deg": round(float(theta), 1), "uniform": True, "knot_ids": []}
    for k in knots:
        theta = round(float(k["theta_deg"]), 1) % 360.0
        key = "theta_%05.1f" % theta
        planes.setdefault(key, {"dir": key, "theta_deg": theta, "uniform": False, "knot_ids": []})
        planes[key]["knot_ids"].append(k["id"])
        k["plane"] = key
    return [planes[k] for k in sorted(planes)]


def verify(
    out_dir: pathlib.Path,
    knots: List[Dict[str, object]],
    planes: List[Dict[str, object]],
    gt: np.ndarray,
    pith: np.ndarray,
    radius: int,
) -> Dict[str, object]:
    by_dir = {p["dir"]: p for p in planes}
    checked = 0
    ray_misses: List[str] = []
    for k in knots:
        if k["source"] != "gt":
            continue
        plane = by_dir["theta_%05.1f" % (round(float(k["theta_deg"]), 1) % 360.0)]
        stored = read_png(out_dir / plane["dir"] / "gt.png")
        z = int(k["z"])
        t = math.radians(plane["theta_deg"])
        r = np.arange(-radius, radius + 1)
        xs = np.rint(pith[z, 0] + r * math.cos(t)).astype(int)
        ys = np.rint(pith[z, 1] + r * math.sin(t)).astype(int)
        ok = (xs >= 0) & (xs < 778) & (ys >= 0) & (ys < 778)
        expected = np.zeros_like(stored[z])
        expected[ok] = gt[z, ys[ok], xs[ok]]
        if not np.array_equal(expected, stored[z]):
            msg = "radial row %d of %s differs from the axial label along the ray" % (z, plane["dir"])
            raise ValueError(msg)
        hit = (stored[z] == 2) & (r >= k["r_min_px"] - 1) & (r <= k["r_max_px"] + 1)
        if not hit.any():
            if ray_crosses_knot(gt[z] == 2, k, pith[z], t):
                msg = "knot %s crosses its ray but is absent from %s row %d" % (k["id"], plane["dir"], z)
                raise ValueError(msg)
            ray_misses.append(k["id"])
        checked += 1
    logger.info("verified %d GT knots; %d centroid rays miss a non-convex knot", checked, len(ray_misses))
    return {"n_gt_knots_checked": checked, "row_equals_axial_along_ray": True, "centroid_ray_misses": ray_misses}


def ray_crosses_knot(knot: np.ndarray, k: Dict[str, object], pith_xy: np.ndarray, t: float) -> bool:
    x0, y0, x1, y1 = k["bbox_xyxy"]
    ys, xs = np.nonzero(knot[y0 : y1 + 1, x0 : x1 + 1])
    dx = xs + x0 - pith_xy[0]
    dy = ys + y0 - pith_xy[1]
    along = dx * math.cos(t) + dy * math.sin(t)
    across = np.abs(-dx * math.sin(t) + dy * math.cos(t))
    return bool(((across < 0.5) & (along > 0)).any())


if __name__ == "__main__":
    main()
