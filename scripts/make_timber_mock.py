import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import ellipse

CLASSES = [
    ("background", [0, 0, 0], False),
    ("wood", [205, 170, 125], False),
    ("sound_knot", [230, 160, 40], False),
    ("dead_knot", [120, 50, 20], False),
    ("resin_pocket", [250, 230, 80], False),
    ("pith", [200, 40, 200], False),
    ("steel", [90, 90, 160], True),
]
DEFECT_TONES = {2: (150, 100, 50), 3: (60, 35, 20), 4: (220, 200, 120), 5: (110, 70, 60), 6: (80, 80, 90)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny synthetic dataset in the CT-13 timber tile format.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--tiles_per_board", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    classes = [{"index": i, "name": n, "colour": c, "ignored": ig} for i, (n, c, ig) in enumerate(CLASSES)]
    classes.append({"index": 255, "name": "ignore", "colour": [255, 255, 255], "ignored": True})
    (args.out / "classes.json").write_text(json.dumps(classes, indent=2))

    boards = {"train": ["spruce_001", "spruce_002", "larch_001"], "val": ["spruce_003"], "test": ["larch_002"]}
    rows: List[Dict[str, object]] = []
    for split, names in boards.items():
        for board in names:
            for k in range(args.tiles_per_board):
                rows.append(make_tile(args.out, board, split, k, args.size, rng, pad=(k == args.tiles_per_board - 1)))
    pd.DataFrame(rows).to_csv(args.out / "tiles.csv", index=False)
    (args.out / "DATASET.json").write_text(json.dumps({"name": "timber_mock", "tile_size": args.size}, indent=2))
    print("Wrote %d tiles to %s" % (len(rows), args.out))


def make_tile(
    root: Path, board: str, split: str, k: int, size: int, rng: np.random.Generator, pad: bool
) -> Dict[str, object]:
    strip = f"{board}_s0"
    tile_id = f"{strip}_x{k * size:05d}_y00000"
    mask = np.zeros((size, size), np.uint8)
    top, bottom = int(size * 0.1), int(size * 0.85)
    mask[top:bottom] = 1
    image = np.zeros((size, size, 3), np.float32)
    grain = 15 * np.sin(np.arange(size)[:, None] / 7.0 + rng.uniform(0, 6)) + rng.normal(0, 6, (size, size))
    image[mask == 1] = np.array([205, 170, 125], np.float32)
    image += (grain[..., None] * (mask == 1)[..., None]).astype(np.float32)
    counts = {name: 0 for name, _, _ in CLASSES[2:]}
    for _ in range(int(rng.integers(2, 6))):
        index = int(rng.choice(list(DEFECT_TONES)))
        radius_r, radius_c = int(rng.integers(12, 60)), int(rng.integers(12, 60))
        centre_r = int(rng.integers(top + radius_r, bottom - radius_r))
        centre_c = int(rng.integers(radius_c, size - radius_c))
        rr, cc = ellipse(centre_r, centre_c, radius_r, radius_c, shape=mask.shape)
        mask[rr, cc] = index
        image[rr, cc] = DEFECT_TONES[index]
        counts[CLASSES[index][0]] += 1
    pad_px = 0
    if pad:
        pad_px = size // 5
        mask[:, size - pad_px :] = 255
        image[:, size - pad_px :] = 0
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(root / "images" / f"{tile_id}.png")
    Image.fromarray(mask, mode="L").save(root / "masks" / f"{tile_id}.png")
    wood = float(((mask != 0) & (mask != 255)).mean())
    return {
        "tile_id": tile_id,
        "board": board,
        "species": board.split("_")[0],
        "split": split,
        "strip": strip,
        "x": k * size,
        "y": 0,
        "w": size,
        "h": size,
        "pad": pad_px,
        "wood_frac": round(wood, 4),
        **{f"n_{name}": n for name, n in counts.items()},
    }


if __name__ == "__main__":
    main()
