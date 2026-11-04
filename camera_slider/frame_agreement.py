import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_slider.registration import Distortion, undistort_maps
from camera_slider.wood import wood_mask

MIN_OVERLAP_PX = 64
ERODE_PX = 15


def piece_frame_agreement(job: Tuple[Path, Path, str, Path]) -> List[Dict]:
    board_dir, raw_root, raw_prefix, cache = job
    if cache.exists():
        return json.loads(cache.read_text())
    mapping = json.loads((board_dir / "mapping.json").read_text())
    pairs = []
    for face in mapping["faces"]:
        pairs += face_pairs(mapping, face, raw_root, raw_prefix)
    rows = []
    for piece in mapping["pieces"]:
        hits = [p for p in pairs if p["x1"] > piece["x0"] and p["x0"] < piece["x1"]]
        worst = max(hits, key=lambda p: p["p90_px"]) if hits else None
        rows.append(
            {
                "piece": piece["name"][:-4],
                "frame_p90_px": round(worst["p90_px"], 2) if worst else None,
                "frame_median_px": round(worst["median_px"], 2) if worst else None,
                "frame_pair": f"{worst['area']} {worst['frames']} x={int(worst['x0'] - piece['x0'])}" if worst else "",
            }
        )
    cache.write_text(json.dumps(rows))
    return rows


def face_pairs(mapping: Dict, face: Dict, raw_root: Path, raw_prefix: str) -> List[Dict]:
    fh, fw = face["frame_shape"]
    reduce = 4 if fw > 4000 else 2
    flag = cv2.IMREAD_REDUCED_COLOR_4 if reduce == 4 else cv2.IMREAD_REDUCED_COLOR_2
    d = face["distortion"]
    centre = (d["centre"][0] / reduce, d["centre"][1] / reduce)
    maps = undistort_maps(Distortion(d["k1"], d["k2"], centre, d["radius"] / reduce), (fh // reduce, fw // reduce))
    ppu = mapping["px_per_unit"]
    ox, oy = face["render_origin"]
    cx, cy = face["crop_box"][:2]
    to_work = np.diag([1 / reduce, 1 / reduce, 1.0]) @ np.array(
        [[ppu, 0, -ox * ppu - cx], [0, ppu, -oy * ppu - cy], [0, 0, 1.0]]
    )
    height = int(face["crop_shape"][0] / reduce)
    width = int(face["crop_shape"][1] / reduce)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    placement = face["placement"]
    out = []
    previous: Optional[Tuple[str, int, np.ndarray, np.ndarray]] = None
    for frame in face["frames"]:
        path = raw_root / frame["path"].removeprefix(raw_prefix)
        image = cv2.imread(str(path), flag | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)
        hom = to_work @ np.array(frame["homography"]) @ np.diag([reduce, reduce, 1.0])
        current = (frame["name"], *warp_frame(image, hom, width, height))
        if previous is not None:
            stats = pair_flow(previous, current, dis)
            if stats is not None:
                lo, hi, median, p90 = stats
                to_unrolled = [((v * reduce) + 0.5) * placement["scale_x"] - 0.5 + placement["x"] for v in (lo, hi)]
                out.append(
                    {
                        "area": face["area"],
                        "frames": f"{previous[0]}/{current[0]}",
                        "x0": to_unrolled[0],
                        "x1": to_unrolled[1],
                        "median_px": median * reduce,
                        "p90_px": p90 * reduce,
                    }
                )
        previous = current
    return out


def warp_frame(image: np.ndarray, hom: np.ndarray, width: int, height: int) -> Tuple[int, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    corners = cv2.perspectiveTransform(np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float32), hom)[0]
    x0 = int(np.clip(corners[:, 0].min(), 0, width))
    x1 = int(np.clip(corners[:, 0].max(), 0, width))
    if x1 - x0 < MIN_OVERLAP_PX:
        return x0, np.zeros((height, 0, 3), np.uint8), np.zeros((height, 0), bool)
    shift = np.array([[1, 0, -x0], [0, 1, 0], [0, 0, 1.0]]) @ hom
    warped = cv2.warpPerspective(image, shift, (x1 - x0, height), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.ones((h, w), np.uint8), shift, (x1 - x0, height), flags=cv2.INTER_NEAREST)
    return x0, warped, valid > 0


def pair_flow(
    a: Tuple[str, int, np.ndarray, np.ndarray], b: Tuple[str, int, np.ndarray, np.ndarray], dis: cv2.DISOpticalFlow
) -> Optional[Tuple[int, int, float, float]]:
    _, xa, ia, va = a
    _, xb, ib, vb = b
    lo, hi = max(xa, xb), min(xa + ia.shape[1], xb + ib.shape[1])
    if hi - lo < MIN_OVERLAP_PX:
        return None
    crop_a, crop_b = ia[:, lo - xa : hi - xa], ib[:, lo - xb : hi - xb]
    valid = va[:, lo - xa : hi - xa] & vb[:, lo - xb : hi - xb] & (wood_mask(crop_a) > 0) & (wood_mask(crop_b) > 0)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((ERODE_PX, ERODE_PX), np.uint8)) > 0
    if valid.sum() < MIN_OVERLAP_PX**2:
        return None
    flow = dis.calc(cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(crop_b, cv2.COLOR_BGR2GRAY), None)
    across_grain = np.abs(flow[..., 1])[valid]
    return lo, hi, float(np.median(across_grain)), float(np.percentile(across_grain, 90))
