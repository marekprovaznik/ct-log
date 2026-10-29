import argparse
import json
import math
import pathlib
from typing import Dict, List

import numpy as np
from scipy import ndimage as ndi
from scripts.radial.common import N_SLICES, OUT, page_name, read_png


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 3D Slicer reslices (from slicer_check.py) with the numpy radial-ray convention."
    )
    parser.add_argument("--work", type=pathlib.Path, required=True, help="Folder with slicer_check_* outputs.")
    parser.add_argument("--root", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    ct = np.stack([read_png(args.root / "images" / ("%s.png" % page_name(z))) for z in range(N_SLICES)])
    ct = ct.astype(np.float32)
    cases = json.loads((args.work / "slicer_check_cases.json").read_text())
    results: List[Dict[str, object]] = []
    for case in cases:
        results.append(compare(ct, args.work, case))
        print(json.dumps(results[-1]))
    out = {
        "method": "Slicer Red view resliced on the plane spanned by direction theta (LPS volume -> RAS: x = -R, y = -A, "
        "z = S / 10) and the S axis through the slice's human pith; every Slicer pixel is mapped back to (r, z) and "
        "the numpy ray x = pith_x + r cos(theta), y = pith_y + r sin(theta) is sampled trilinearly there. "
        "Control: the same with theta + 180.",
        "cases": results,
    }
    (args.root / "radial" / "slicer_check.json").write_text(json.dumps(out, indent=1))


def compare(ct: np.ndarray, work: pathlib.Path, case: Dict[str, object]) -> Dict[str, object]:
    sl = np.load(work / ("slicer_check_%s.npy" % case["name"])).astype(np.float64)
    xy_to_ras = np.load(work / ("slicer_check_%s_xytoras.npy" % case["name"]))
    ny, nx = sl.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    xy = np.stack([ii, jj, np.zeros_like(ii), np.ones_like(ii)]).astype(np.float64)
    ras = np.einsum("ab,bij->aij", xy_to_ras, xy)[:3]
    t = math.radians(case["theta_deg"])
    px, py = case["pith_xy"]
    r = -math.cos(t) * (ras[0] + px) - math.sin(t) * (ras[1] + py)
    z = ras[2] / 10.0
    x = px + r * math.cos(t)
    y = py + r * math.sin(t)
    ok = (z >= 0) & (z <= N_SLICES - 1) & (x >= 0) & (x <= 777) & (y >= 0) & (y <= 777) & (sl > 0)
    mine = ndi.map_coordinates(ct, [z, y, x], order=1)
    mirrored = ndi.map_coordinates(ct, [z, 2 * py - y, 2 * px - x], order=1)
    diff = np.abs(mine - sl)[ok]
    return {
        "name": case["name"],
        "theta_deg": case["theta_deg"],
        "n_pixels": int(ok.sum()),
        "mae": round(float(diff.mean()), 3),
        "p99_abs_diff": round(float(np.percentile(diff, 99)), 3),
        "corr": round(float(np.corrcoef(mine[ok], sl[ok])[0, 1]), 6),
        "mirrored_control_mae": round(float(np.abs(mirrored - sl)[ok].mean()), 3),
    }


if __name__ == "__main__":
    main()
