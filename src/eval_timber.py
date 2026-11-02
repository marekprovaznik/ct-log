import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision

from src.configs.timber_training_config import TimberTrainingConfig
from src.dataset.timber_tile_dataset import IGNORE_INDEX, TimberClasses, TimberTileDataset
from src.eval.timber_instances import TimberInstanceEvaluator
from src.train_timber import evaluate, predict
from src.unet_model import UnetSegPith

TILE = 800


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a timber checkpoint per tile or on reassembled strips.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--mode", choices=["tiles", "strips"], default="tiles")
    parser.add_argument("--max_tiles", type=int, default=None, help="Seeded subset (seed from config), tiles mode.")
    parser.add_argument("--strips", type=str, nargs="+", default=None, help="Strips mode: evaluate only these strips.")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--out", type=str, required=True, help="JSON with one entry per checkpoint.")
    parser.add_argument("--pred_dir", type=str, default=None, help="Strips mode: write predicted class PNGs here.")
    args = parser.parse_args()

    config = TimberTrainingConfig.from_yaml(args.config)
    if args.dataset_root:
        config.dataset_root = args.dataset_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TimberTileDataset(root=config.dataset_root, split=args.split, max_tiles=args.max_tiles, seed=config.seed)
    classes = dataset.classes
    results: Dict[str, Dict[str, float]] = {}
    for checkpoint in args.checkpoints:
        model = load_model(config, classes, checkpoint, device)
        if args.mode == "tiles":
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            loss, metrics = evaluate(model, loader, normalizer(), device, classes, config)
            metrics["loss"] = loss
        else:
            pred_dir = Path(args.pred_dir) / Path(checkpoint).stem if args.pred_dir else None
            metrics = evaluate_strips(model, dataset, classes, config, device, args.batch_size, pred_dir, args.strips)
        metrics.setdefault("n_tiles", float(len(dataset)))
        results[checkpoint] = metrics
        print("%s %s" % (checkpoint, json.dumps({k: round(v, 4) for k, v in metrics.items()})), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"split": args.split, "mode": args.mode, "results": results}, indent=2))


def load_model(
    config: TimberTrainingConfig, classes: TimberClasses, checkpoint: str, device: torch.device
) -> torch.nn.Module:
    model = UnetSegPith(
        arch=config.unet_arch,
        encoder=config.unet_encoder,
        num_classes=classes.num_outputs,
        pith=False,
        input_size=TILE,
        bf16=config.bf16 and device.type == "cuda",
        in_channels=3,
        decoder_channels=config.unet_decoder_channels,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


def normalizer() -> torch.nn.Module:
    return torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


@torch.no_grad()
def evaluate_strips(
    model: torch.nn.Module,
    dataset: TimberTileDataset,
    classes: TimberClasses,
    config: TimberTrainingConfig,
    device: torch.device,
    batch_size: int,
    pred_dir: Optional[Path],
    strips: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Instance metrics on strips reassembled from overlapping tiles (probabilities averaged).

    Args:
        model: Segmentation network in eval mode.
        dataset: Tiles of one split.
        classes: Class table.
        config: Training configuration (instance settings).
        device: Compute device.
        batch_size: Tiles per forward pass.
        pred_dir: Optional output directory for <strip>.png predicted class maps.
        strips: Evaluate only these strip ids (all when None).

    Returns:
        Dict[str, float]: TimberInstanceEvaluator.summary() over strips plus n_strips.
    """
    names = classes.output_names()
    instances = TimberInstanceEvaluator(
        classes.defect_indices,
        [names[i] for i in classes.defect_indices],
        IGNORE_INDEX,
        config.instance_score_reduce,
        config.instance_min_pred_area,
    )
    normalize = normalizer()
    tiles = dataset.tiles if strips is None else dataset.tiles[dataset.tiles["strip"].isin(strips)]
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    for strip, rows in tiles.groupby("strip", sort=True):
        target, probs = reassemble_strip(model, dataset, rows, classes, normalize, device, batch_size)
        prediction = probs.argmax(axis=0)
        instances.add(target, prediction, probs)
        if pred_dir:
            Image.fromarray(prediction.astype(np.uint8)).save(pred_dir / f"{strip}.png")
    metrics = instances.summary()
    metrics["n_strips"] = float(tiles["strip"].nunique())
    metrics["n_tiles"] = float(len(tiles))
    return metrics


def reassemble_strip(
    model: torch.nn.Module,
    dataset: TimberTileDataset,
    rows: pd.DataFrame,
    classes: TimberClasses,
    normalize: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict every tile of one strip and average class probabilities where tiles overlap.

    Args:
        model: Segmentation network.
        dataset: Dataset holding the tiles (its rows are indexed by position in dataset.tiles).
        rows: Rows of dataset.tiles belonging to one strip.
        classes: Class table.
        normalize: ImageNet normalization.
        device: Compute device.
        batch_size: Tiles per forward pass.

    Returns:
        Tuple[np.ndarray, np.ndarray]: [H, W] int target (IGNORE_INDEX where unsupervised or not
            covered) and [C, H, W] float32 averaged probabilities.
    """
    height = int((rows["y"] + rows["h"]).max())
    width = int((rows["x"] + rows["w"]).max())
    num = classes.num_outputs
    prob_sum = np.zeros((num, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    target = np.full((height, width), IGNORE_INDEX, dtype=np.int64)
    indices: List[int] = rows.index.tolist()
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        items = [dataset[i] for i in chunk]
        images = torch.stack([item["image"] for item in items]).to(device)
        probs = predict(model, normalize(images), classes).softmax(dim=1).cpu().numpy()
        for item, index, tile_probs in zip(items, chunk, probs):
            row = dataset.tiles.loc[index]
            y, x, h, w = int(row["y"]), int(row["x"]), int(row["h"]), int(row["w"])
            prob_sum[:, y : y + h, x : x + w] += tile_probs
            count[y : y + h, x : x + w] += 1.0
            target[y : y + h, x : x + w] = item["mask"].numpy()
    covered = count > 0
    probs_avg = np.where(covered[None], prob_sum / np.maximum(count, 1.0)[None], 0.0).astype(np.float32)
    return target, probs_avg


if __name__ == "__main__":
    main()
