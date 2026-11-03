import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_slider.compose import NO_OWNER, FaceCrop, FacePlacement, layout_placements
from camera_slider.registration import Distortion, PassRegistration, undistort_points

MAPPING_VERSION = 1
MAPPING_FILE = "mapping.json"


@dataclasses.dataclass
class FrameMap:
    index: int
    name: str
    path: str
    shape: Tuple[int, int]
    homography: np.ndarray


@dataclasses.dataclass
class FaceMap:
    area: str
    row: int
    origin: Tuple[float, float]
    crop_box: Tuple[int, int, int, int]
    placement: FacePlacement
    px_per_unit: float
    height_k: float
    distortion: Distortion
    frames: List[FrameMap]
    owner: np.ndarray

    @property
    def crop_shape(self) -> Tuple[int, int]:
        return self.owner.shape[0], self.owner.shape[1]


@dataclasses.dataclass
class BoardMap:
    board: str
    unrolled_shape: Tuple[int, int]
    faces: List[FaceMap]
    pieces: List[Dict]

    def piece(self, name: str) -> Dict:
        for piece in self.pieces:
            if piece["name"] == name:
                return piece
        msg = f"{self.board}: no piece {name}"
        raise ValueError(msg)

    def frame_index(self) -> Dict[str, Tuple[int, int]]:
        return {f.path: (k, f.index) for k, face in enumerate(self.faces) for f in face.frames}


def write_mapping(
    out_dir: Path,
    board: str,
    groups: List[List[Path]],
    geometries: List[PassRegistration],
    crops: List[FaceCrop],
    frame_shapes: List[Tuple[int, int]],
    areas: List[str],
    ordered: List[int],
    px_per_unit: float,
    unrolled_shape: Tuple[int, int],
    pieces: List[Dict],
    gap: int,
    stretch: bool,
    margin: int,
) -> int:
    placements = layout_placements([crops[k].image.shape[:2] for k in ordered], gap, stretch, margin)
    rows = {k: row for row, k in enumerate(ordered)}
    faces = []
    n_bytes = 0
    for k, (group, geometry, crop, shape, area) in enumerate(zip(groups, geometries, crops, frame_shapes, areas)):
        by_name = {p.name: p for p in group}
        frames = [
            {
                "index": index,
                "name": fit.name,
                "path": str(by_name[fit.name].resolve()),
                "homography": fit.transform.tolist(),
            }
            for index, fit in enumerate(f for f in geometry.frames if f.registered)
        ]
        owner_name = f"owner_{area}.png"
        if not cv2.imwrite(str(out_dir / owner_name), crop.owner, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            msg = f"Cannot write {out_dir / owner_name}"
            raise ValueError(msg)
        n_bytes += (out_dir / owner_name).stat().st_size
        d = geometry.distortion
        faces.append(
            {
                "area": area,
                "row": rows[k],
                "render_origin": list(crop.origin),
                "crop_box": list(crop.box),
                "crop_shape": list(crop.owner.shape),
                "placement": dataclasses.asdict(placements[rows[k]]),
                "height_k": geometry.height_k,
                "distortion": {"k1": d.k1, "k2": d.k2, "centre": list(d.centre), "radius": d.radius},
                "frame_shape": list(shape),
                "owner_map": owner_name,
                "frames": frames,
            }
        )
    mapping = {
        "version": MAPPING_VERSION,
        "board": board,
        "chain": (
            "frame px -> radial undistortion -> homography (board units, height k included) -> "
            "(x - render_origin) * px_per_unit -> minus crop_box[:2] -> placement (x scaled by scale_x "
            "with pixel-centre convention, then offset) -> unrolled px -> minus piece x0"
        ),
        "px_per_unit": px_per_unit,
        "unrolled_shape": list(unrolled_shape),
        "layout": {"gap_px": gap, "margin_px": margin, "stretch": stretch},
        "no_owner": NO_OWNER,
        "pieces": pieces,
        "faces": faces,
    }
    (out_dir / MAPPING_FILE).write_text(json.dumps(mapping, indent=1))
    return n_bytes + (out_dir / MAPPING_FILE).stat().st_size


def load_mapping(board_dir: Path) -> BoardMap:
    data = json.loads((board_dir / MAPPING_FILE).read_text())
    if data["version"] != MAPPING_VERSION:
        msg = f"{board_dir}: mapping version {data['version']}, expected {MAPPING_VERSION}"
        raise ValueError(msg)
    faces = []
    for face in data["faces"]:
        owner = cv2.imread(str(board_dir / face["owner_map"]), cv2.IMREAD_UNCHANGED)
        if owner is None or list(owner.shape) != face["crop_shape"]:
            msg = f"{board_dir}: bad owner map {face['owner_map']}"
            raise ValueError(msg)
        d = face["distortion"]
        frame_shape = tuple(face["frame_shape"])
        faces.append(
            FaceMap(
                face["area"],
                face["row"],
                tuple(face["render_origin"]),
                tuple(face["crop_box"]),
                FacePlacement(**face["placement"]),
                data["px_per_unit"],
                face["height_k"],
                Distortion(d["k1"], d["k2"], tuple(d["centre"]), d["radius"]),
                [
                    FrameMap(f["index"], f["name"], f["path"], frame_shape, np.array(f["homography"]))
                    for f in face["frames"]
                ],
                owner,
            )
        )
    faces.sort(key=lambda f: f.row)
    return BoardMap(data["board"], tuple(data["unrolled_shape"]), faces, data["pieces"])


def unrolled_to_face(
    board: BoardMap, pts: np.ndarray, rows: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    given = rows
    rows = np.full(len(pts), -1) if given is None else given.copy()
    crop = np.full((len(pts), 2), np.nan)
    for row, face in enumerate(board.faces):
        p = face.placement
        if given is None:
            inside = (pts[:, 1] >= p.y - 0.5) & (pts[:, 1] < p.y + p.height - 0.5)
            rows[inside] = row
        else:
            inside = rows == row
        crop[inside, 0] = (pts[inside, 0] - p.x + 0.5) / p.scale_x - 0.5
        crop[inside, 1] = pts[inside, 1] - p.y
    return rows, crop


def face_to_unrolled(face: FaceMap, crop: np.ndarray) -> np.ndarray:
    p = face.placement
    return np.stack([(crop[:, 0] + 0.5) * p.scale_x - 0.5 + p.x, crop[:, 1] + p.y], 1)


def owners(face: FaceMap, crop: np.ndarray) -> np.ndarray:
    h, w = face.owner.shape
    ix = np.round(crop[:, 0]).astype(int)
    iy = np.round(crop[:, 1]).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.full(len(crop), NO_OWNER, np.int64)
    out[valid] = face.owner[iy[valid], ix[valid]]
    return out


def crop_to_frame(face: FaceMap, crop: np.ndarray, frame: FrameMap) -> np.ndarray:
    board = (crop + np.array(face.crop_box[:2])) / face.px_per_unit + np.array(face.origin)
    undistorted = cv2.perspectiveTransform(board[None].astype(np.float64), np.linalg.inv(frame.homography))[0]
    return distort_points(undistorted, face.distortion)


def frame_to_crop(face: FaceMap, raw: np.ndarray, frame: FrameMap) -> np.ndarray:
    undistorted = undistort_points(raw.astype(np.float64), face.distortion)
    board = cv2.perspectiveTransform(undistorted[None], frame.homography)[0]
    return (board - np.array(face.origin)) * face.px_per_unit - np.array(face.crop_box[:2])


def distort_points(pts: np.ndarray, distortion: Distortion, iterations: int = 20) -> np.ndarray:
    centre = np.array(distortion.centre, np.float64)
    rel = pts - centre
    source = rel.copy()
    for _ in range(iterations):
        s = (source**2).sum(1) / distortion.radius**2
        source = rel / (1 + distortion.k1 * s + distortion.k2 * s**2)[:, None]
    return source + centre
