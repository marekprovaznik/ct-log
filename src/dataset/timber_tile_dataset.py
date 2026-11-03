from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

IGNORE_INDEX = 255
FLIP_PROB = 0.5
JITTER_PROB = 0.8
BRIGHTNESS_RANGE = (-0.08, 0.08)
CONTRAST_RANGE = (0.85, 1.15)
SATURATION_RANGE = (0.85, 1.15)
STRONG_AFFINE_PROB = 0.7
STRONG_SCALE_RANGE = (0.8, 1.25)
STRONG_ANGLE_RANGE = (-10.0, 10.0)
STRONG_BLUR_PROB = 0.3
STRONG_MOTION_LENGTH = (3, 9)
STRONG_GAUSS_SIGMA = (0.3, 1.2)
STRONG_GRADIENT_PROB = 0.5
STRONG_GRADIENT_AMPLITUDE = 0.2
STRONG_BRIGHTNESS_RANGE = (-0.15, 0.15)
STRONG_CONTRAST_RANGE = (0.75, 1.3)
STRONG_SATURATION_RANGE = (0.7, 1.3)
STRONG_NOISE_PROB = 0.3
STRONG_NOISE_SIGMA = 0.02


@dataclass(frozen=True)
class TimberClass:
    index: int
    name: str
    ignored: bool
    colour: List[int]


@dataclass(frozen=True)
class TimberClasses:
    """Class table of a timber tile dataset (classes.json), indexed by class index."""

    classes: List[TimberClass]

    @classmethod
    def from_json(cls, path: Path) -> "TimberClasses":
        entries = json.loads(Path(path).read_text())
        classes = [
            TimberClass(
                int(e["index"]), str(e["name"]), bool(e.get("ignored", False)), list(e.get("colour", [0, 0, 0]))
            )
            for e in entries
        ]
        return cls(sorted(classes, key=lambda c: c.index))

    @property
    def num_outputs(self) -> int:
        return max(c.index for c in self.classes if c.index != IGNORE_INDEX) + 1

    @property
    def ignored_indices(self) -> List[int]:
        return [c.index for c in self.classes if c.ignored]

    @property
    def trained_indices(self) -> List[int]:
        return [c.index for c in self.classes if not c.ignored and c.index != IGNORE_INDEX]

    @property
    def defect_indices(self) -> List[int]:
        return [i for i in self.trained_indices if i >= 2]

    def name(self, index: int) -> str:
        return {c.index: c.name for c in self.classes}[index]

    def output_names(self) -> List[str]:
        names = {c.index: c.name for c in self.classes}
        return [names.get(i, "unused_%d" % i) for i in range(self.num_outputs)]

    def lookup_table(self) -> np.ndarray:
        """Raw mask value -> training target (ignored classes and unknown values -> IGNORE_INDEX).

        Returns:
            np.ndarray: [256] uint8 table.
        """
        table = np.full(256, IGNORE_INDEX, dtype=np.uint8)
        for index in self.trained_indices:
            table[index] = index
        return table

    def palette(self) -> np.ndarray:
        palette = np.full((256, 3), 255, dtype=np.uint8)
        for c in self.classes:
            palette[c.index] = c.colour
        return palette


class TimberTileDataset(torch.utils.data.Dataset):
    """RGB tiles and class-index masks of one split of a CT-13 timber tile dataset.

    Masks are returned with ignored classes (classes.json "ignored": true) and padding mapped to
    IGNORE_INDEX, so losses and metrics skip them.
    """

    def __init__(
        self,
        root: str,
        split: str,
        augment: bool = False,
        augment_strength: str = "default",
        min_wood_frac: float = 0.0,
        max_tiles: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """Read tiles.csv and classes.json.

        Args:
            root: Dataset root with tiles.csv, classes.json, images/ and masks/.
            split: Value of the tiles.csv split column.
            augment: Random flips and colour jitter.
            augment_strength: "default" (flips, mild jitter) or "strong" (adds scale/rotation, motion or
                Gaussian blur, lighting gradient, stronger jitter and noise).
            min_wood_frac: Drop tiles with wood_frac below this.
            max_tiles: Keep a seeded random subset of at most this many tiles (smoke runs).
            seed: Seed of the subset.

        Raises:
            ValueError: If the split has no tiles.
        """
        self.root = Path(root)
        self.split = split
        self.augment = augment
        if augment_strength not in ("default", "strong"):
            msg = "augment_strength must be 'default' or 'strong', got %r" % augment_strength
            raise ValueError(msg)
        self.augment_strength = augment_strength
        self.classes = TimberClasses.from_json(self.root / "classes.json")
        self.table = self.classes.lookup_table()
        self.known_values = np.array(sorted({c.index for c in self.classes.classes}))
        tiles = pd.read_csv(self.root / "tiles.csv", dtype={"tile_id": str, "board": str, "split": str})
        tiles = tiles[(tiles["split"] == split) & (tiles["wood_frac"] >= min_wood_frac)]
        if max_tiles is not None and len(tiles) > max_tiles:
            tiles = tiles.sample(n=max_tiles, random_state=seed)
        self.tiles = tiles.sort_values("tile_id").reset_index(drop=True)
        if self.tiles.empty:
            msg = "No tiles for split %r in %s" % (split, self.root)
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.tiles.iloc[index]
        tile_id = row["tile_id"]
        image = np.asarray(Image.open(self.root / "images" / f"{tile_id}.png").convert("RGB"))
        raw = np.asarray(Image.open(self.root / "masks" / f"{tile_id}.png"))
        if raw.ndim != 2:
            msg = "Mask %s must be single-channel, got shape %s" % (tile_id, raw.shape)
            raise ValueError(msg)
        unknown = np.setdiff1d(np.unique(raw), self.known_values)
        if unknown.size:
            msg = "Mask %s has values %s not in classes.json" % (tile_id, unknown.tolist())
            raise ValueError(msg)
        image_t = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(self.table[raw].astype(np.int64))
        if self.augment and self.augment_strength == "strong":
            image_t, mask_t = augment_tile_strong(image_t, mask_t)
        elif self.augment:
            image_t, mask_t = augment_tile(image_t, mask_t)
        return {"image": image_t, "mask": mask_t, "tile_id": tile_id}


def augment_tile(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Horizontal/vertical flips and brightness/contrast/saturation jitter.

    Board surface tiles keep their grain direction along x, so no 90-degree rotations.

    Args:
        image: [3, H, W] float RGB in [0, 1].
        mask: [H, W] int64 targets.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: augmented image and mask.
    """
    for dim in (-1, -2):
        if torch.rand(1).item() < FLIP_PROB:
            image = image.flip(dim)
            mask = mask.flip(dim)
    if torch.rand(1).item() < JITTER_PROB:
        grey = image.mean(dim=0, keepdim=True)
        image = grey + (image - grey) * float(torch.empty(1).uniform_(*SATURATION_RANGE))
        mean = image.mean()
        image = (image - mean) * float(torch.empty(1).uniform_(*CONTRAST_RANGE)) + mean
        image = image + float(torch.empty(1).uniform_(*BRIGHTNESS_RANGE))
        image = image.clamp(0.0, 1.0)
    return image.contiguous(), mask.contiguous()


def augment_tile_strong(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flips, scale and small rotation, motion or Gaussian blur, lighting gradient, colour jitter and noise.

    Rotations stay within +-10 degrees so the grain keeps running along x. Pixels moved in from outside
    the tile are IGNORE_INDEX in the mask.

    Args:
        image: [3, H, W] float RGB in [0, 1].
        mask: [H, W] int64 targets.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: augmented image and mask.
    """
    for dim in (-1, -2):
        if torch.rand(1).item() < FLIP_PROB:
            image = image.flip(dim)
            mask = mask.flip(dim)
    if torch.rand(1).item() < STRONG_AFFINE_PROB:
        image, mask = random_affine(image, mask)
    if torch.rand(1).item() < STRONG_BLUR_PROB:
        image = random_blur(image)
    if torch.rand(1).item() < STRONG_GRADIENT_PROB:
        image = lighting_gradient(image)
    if torch.rand(1).item() < JITTER_PROB:
        grey = image.mean(dim=0, keepdim=True)
        image = grey + (image - grey) * float(torch.empty(1).uniform_(*STRONG_SATURATION_RANGE))
        mean = image.mean()
        image = (image - mean) * float(torch.empty(1).uniform_(*STRONG_CONTRAST_RANGE)) + mean
        image = image + float(torch.empty(1).uniform_(*STRONG_BRIGHTNESS_RANGE))
    if torch.rand(1).item() < STRONG_NOISE_PROB:
        image = image + torch.randn_like(image) * STRONG_NOISE_SIGMA * float(torch.rand(1))
    return image.clamp(0.0, 1.0).contiguous(), mask.contiguous()


def random_affine(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = float(np.exp(np.random.uniform(np.log(STRONG_SCALE_RANGE[0]), np.log(STRONG_SCALE_RANGE[1]))))
    angle = float(np.random.uniform(*STRONG_ANGLE_RANGE))
    params = {"angle": angle, "translate": [0, 0], "scale": scale, "shear": [0.0, 0.0]}
    image = TF.affine(image, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0, **params)
    shifted = (mask + 1).to(torch.uint8)[None]
    shifted = TF.affine(shifted, interpolation=TF.InterpolationMode.NEAREST, fill=0, **params)[0].long()
    mask = torch.where(shifted == 0, torch.full_like(shifted, IGNORE_INDEX), shifted - 1)
    return image, mask


def random_blur(image: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        length = int(np.random.randint(STRONG_MOTION_LENGTH[0], STRONG_MOTION_LENGTH[1] + 1))
        kernel = torch.full((3, 1, 1, length), 1.0 / length)
        return F.conv2d(F.pad(image[None], (length // 2, (length - 1) // 2, 0, 0), mode="replicate"), kernel, groups=3)[
            0
        ]
    sigma = float(np.random.uniform(*STRONG_GAUSS_SIGMA))
    return TF.gaussian_blur(image, kernel_size=2 * int(np.ceil(3 * sigma)) + 1, sigma=sigma)


def lighting_gradient(image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    ys = torch.linspace(-1.0, 1.0, height)[:, None]
    xs = torch.linspace(-1.0, 1.0, width)[None, :]
    theta = float(np.random.uniform(0, 2 * np.pi))
    ramp = np.cos(theta) * xs + np.sin(theta) * ys
    radial = 1.0 - (xs**2 + ys**2) / 2.0
    amplitude = float(np.random.uniform(-STRONG_GRADIENT_AMPLITUDE, STRONG_GRADIENT_AMPLITUDE))
    field = 1.0 + amplitude * (0.5 * ramp + 0.5 * radial)
    return image * field[None]
