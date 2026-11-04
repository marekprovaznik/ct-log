import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch

from src.configs.timber_training_config import TimberTrainingConfig
from src.dataset.timber_tile_dataset import TimberClasses
from src.eval_timber import load_model, normalizer
from src.train_timber import predict

TILE = 800
STRIDE = 600

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict class maps for whole board pieces at native resolution.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--classes_json", type=str, required=True)
    parser.add_argument("--images", type=str, nargs="+", required=True)
    parser.add_argument("--px_per_mm", type=float, nargs="+", required=True, help="Native px/mm, one per image.")
    parser.add_argument("--target_px_per_mm", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    if len(args.images) != len(args.px_per_mm):
        msg = "Got %d images but %d px_per_mm values" % (len(args.images), len(args.px_per_mm))
        raise ValueError(msg)
    config = TimberTrainingConfig.from_yaml(args.config)
    classes = TimberClasses.from_json(Path(args.classes_json))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, classes, args.checkpoint, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for image_path, px_per_mm in zip(args.images, args.px_per_mm):
        image = Image.open(image_path).convert("RGB")
        scale = args.target_px_per_mm / px_per_mm
        small = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BOX)
        probs = predict_probs(model, np.asarray(small), classes, device, args.batch_size)
        labels = upsample_labels(probs, image.width, image.height)
        name = Path(image_path).name
        Image.fromarray(labels.astype(np.uint8)).save(out_dir / (Path(name).stem + ".png"))
        manifest.append({"image": str(image_path), "name": name, "scale": scale, "classes": classes.output_names()})
        print(
            "%s scale %.3f counts %s"
            % (name, scale, np.bincount(labels.ravel(), minlength=classes.num_outputs).tolist()),
            flush=True,
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def tile_starts(length: int) -> List[int]:
    if length <= TILE:
        return [0]
    starts = list(range(0, length - TILE, STRIDE))
    starts.append(length - TILE)
    return starts


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module, image: np.ndarray, classes: TimberClasses, device: torch.device, batch_size: int
) -> np.ndarray:
    """Class probabilities of a resampled piece from overlapping 800 px tiles, averaged in overlaps.

    Args:
        model: Segmentation network in eval mode.
        image: [H, W, 3] uint8 RGB at the training resolution.
        classes: Class table.
        device: Compute device.
        batch_size: Tiles per forward pass.

    Returns:
        np.ndarray: [C, H, W] float32 probabilities.
    """
    height, width = image.shape[:2]
    padded = np.zeros((max(height, TILE), max(width, TILE), 3), dtype=np.uint8)
    padded[:height, :width] = image
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    prob_sum = np.zeros((classes.num_outputs, *padded.shape[:2]), dtype=np.float32)
    count = np.zeros(padded.shape[:2], dtype=np.float32)
    boxes = [(y, x) for y in tile_starts(padded.shape[0]) for x in tile_starts(padded.shape[1])]
    normalize = normalizer()
    for start in range(0, len(boxes), batch_size):
        chunk = boxes[start : start + batch_size]
        batch = torch.stack([tensor[:, y : y + TILE, x : x + TILE] for y, x in chunk]).to(device)
        probs = predict(model, normalize(batch), classes).softmax(dim=1).cpu().numpy()
        for (y, x), tile_probs in zip(chunk, probs):
            prob_sum[:, y : y + TILE, x : x + TILE] += tile_probs
            count[y : y + TILE, x : x + TILE] += 1.0
    return (prob_sum / np.maximum(count, 1.0)[None])[:, :height, :width]


def upsample_labels(probs: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear upsampling of the probabilities to native size, then argmax.

    Args:
        probs: [C, h, w] probabilities at the training resolution.
        width: Native width.
        height: Native height.

    Returns:
        np.ndarray: [height, width] uint8 class map.
    """
    tensor = torch.from_numpy(probs)[None]
    upsampled = torch.nn.functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return upsampled[0].argmax(dim=0).numpy().astype(np.uint8)


if __name__ == "__main__":
    main()
