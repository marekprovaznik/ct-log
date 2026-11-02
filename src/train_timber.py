import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision

from src.configs.timber_training_config import TimberTrainingConfig
from src.dataset.timber_tile_dataset import IGNORE_INDEX, TimberClasses, TimberTileDataset
from src.eval.timber_instances import TimberInstanceEvaluator
from src.loggers import CombinedLogger, ILogger, LocalLogger, MlflowLogger
from src.loss.functional.masked_seg_loss import masked_focal_loss, masked_soft_dice_loss
from src.unet_model import UnetSegPith
from src.utils.checkpoints import save_best_copy
from src.utils.ema import WeightEma
from src.utils.metrics import MetricsTracker
from src.utils.per_class_iou import PerClassIoU
from src.utils.provenance import git_provenance
from src.utils.train_info import TrainInfo

COMPLETION_MARKER = "Best foreground mean IoU"
SEPARATOR_WIDTH = 4
IGNORE_COLOUR = (128, 128, 128)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None, help="Cap train batches per epoch (smoke runs).")
    parser.add_argument("--max_tiles_per_split", type=int, default=None, help="Seeded tile subset per split.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_log_dir", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default=None)
    parser.add_argument("--no_mlflow", action="store_true")
    parser.add_argument("--no_log_model", action="store_true", help="Skip MLflow model uploads.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = TimberTrainingConfig.from_yaml(args.config)
    overrides = {
        "mlflow_run_name": args.run_name,
        "dataset_root": args.dataset_root,
        "num_epochs": args.num_epochs,
        "max_train_batches": args.max_train_batches,
        "max_tiles_per_split": args.max_tiles_per_split,
        "checkpoint_path": Path(args.checkpoint_path) if args.checkpoint_path else None,
        "local_log_dir": Path(args.local_log_dir) if args.local_log_dir else None,
        "mlflow_experiment_name": args.mlflow_experiment_name,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    if args.no_mlflow:
        config.use_mlflow = False
    if args.no_log_model:
        config.log_model = False
    if args.resume:
        config.resume = True
    train(config)


def train(config: TimberTrainingConfig) -> Dict[str, float]:
    """Train, validate every epoch and test the best (smoothed) checkpoint at the end.

    Args:
        config: Training configuration.

    Returns:
        Dict[str, float]: Test metrics (empty when the dataset has no test split).
    """
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, classes = build_dataloaders(config)
    print(
        "Tiles: %s; outputs %d (%s); trained %s; defects %s; ignored %s"
        % (
            {split: len(loader.dataset) for split, loader in loaders.items()},
            classes.num_outputs,
            ",".join(classes.output_names()),
            classes.trained_indices,
            classes.defect_indices,
            classes.ignored_indices,
        )
    )

    logger = build_logger(config)
    run_params = config.model_dump(mode="json") | git_provenance()
    logger.log_params(run_params)
    logger.log_tags(
        {
            "dataset": config.dataset_version,
            "task": "timber-defect-segmentation",
            "classes": ",".join(classes.output_names()),
        }
    )
    logger.log_dict(run_params, "config.yaml")

    model = UnetSegPith(
        arch=config.unet_arch,
        encoder=config.unet_encoder,
        num_classes=classes.num_outputs,
        pith=False,
        input_size=loaders["train"].dataset[0]["image"].shape[-1],
        bf16=config.bf16 and device.type == "cuda",
        in_channels=3,
        decoder_channels=config.unet_decoder_channels,
    ).to(device)
    normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.num_epochs - config.warmup_epochs, 1))
        if config.lr_schedule == "cosine"
        else None
    )
    ema = WeightEma([model], None, config.ema_decay) if config.ema_decay else None

    run_name = config.mlflow_run_name or config.checkpoint_path.stem
    best_copy_dir = config.local_checkpoint_dir / run_name
    info = TrainInfo(
        run_name=run_name,
        num_epochs=config.num_epochs,
        checkpoint_path=str(config.checkpoint_path),
        local_best_copies=str(best_copy_dir),
    )
    state = {
        "start_epoch": 0,
        "history": [],
        "best_smoothed": -math.inf,
        "best_epoch": None,
        "best_fg_iou": 0.0,
        "last_uploaded": -math.inf,
    }
    resume_path = config.checkpoint_path.with_suffix(".resume.pth")
    if config.resume and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        if ema is not None:
            ema.load_state_dict(saved["ema"])
        state = saved["state"]
        info = TrainInfo(**saved["train_info"])
        info.status = "running"
        info.resumed_from_epoch = state["start_epoch"]
        print("Resumed from %s at epoch %d" % (resume_path, state["start_epoch"]))

    tracker = MetricsTracker()
    num_batches = len(loaders["train"])
    if config.max_train_batches is not None:
        num_batches = min(num_batches, config.max_train_batches)

    for epoch in range(state["start_epoch"], config.num_epochs):
        epoch_start = time.monotonic()
        model.train()
        losses: List[float] = []
        for batch_idx, batch in enumerate(loaders["train"]):
            if batch_idx >= num_batches:
                break
            if epoch < config.warmup_epochs:
                fraction = (epoch * num_batches + batch_idx + 1) / (config.warmup_epochs * num_batches)
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * min(fraction, 1.0)
            optimizer.zero_grad()
            logits = model(normalize(batch["image"].to(device)))
            loss = compute_loss(logits, batch["mask"].to(device), classes, config)
            if not torch.isfinite(loss):
                msg = "Non-finite loss %s at epoch %d batch %d" % (loss.item(), epoch, batch_idx)
                raise ValueError(msg)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update()
            losses.append(loss.item())
            if config.log_interval and batch_idx % config.log_interval == 0:
                print("Epoch %d, Batch %d/%d, Loss: %.4f" % (epoch, batch_idx, num_batches, loss.item()))
        train_loss = float(np.mean(losses))

        if ema is not None:
            ema.swap()
        val_loss, val_metrics = evaluate(model, loaders["val"], normalize, device, classes, config)
        tracker.add(epoch, "train", train_loss, float("nan"), lr=optimizer.param_groups[0]["lr"])
        logger.log_metrics(tracker.get(epoch, "train")[-1])
        tracker.add(epoch, "val", val_loss, val_metrics["mean_iou_fg"], **val_metrics)
        logger.log_metrics(tracker.get(epoch, "val")[-1])
        print(
            "Epoch %d, TrainLoss %.4f, ValLoss %.4f, %s"
            % (epoch, train_loss, val_loss, " ".join("%s=%.3f" % item for item in val_metrics.items()))
        )

        if config.train_eval_interval and epoch % config.train_eval_interval == 0:
            train_eval_loss, train_eval_metrics = evaluate(
                model, loaders["train_eval"], normalize, device, classes, config
            )
            tracker.add(epoch, "train_eval", train_eval_loss, train_eval_metrics["mean_iou_fg"], **train_eval_metrics)
            logger.log_metrics(tracker.get(epoch, "train_eval")[-1])

        if config.viz_interval and (epoch % config.viz_interval == 0 or epoch == config.num_epochs - 1):
            for split in ("train_eval", "val"):
                log_panels(model, loaders[split].dataset, split, epoch, normalize, device, classes, config, logger)

        value = val_metrics.get(config.best_metric, float("nan"))
        state["history"].append(value if math.isfinite(value) else 0.0)
        state["best_fg_iou"] = max(state["best_fg_iou"], val_metrics["mean_iou_fg"])
        window = state["history"][-config.smoothing_window :]
        smoothed = float(np.mean(window))
        info.last_val = {"loss": val_loss, **val_metrics, "smoothed_" + config.best_metric: smoothed}
        if smoothed > state["best_smoothed"]:
            state["best_smoothed"] = smoothed
            state["best_epoch"] = epoch
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.checkpoint_path)
            copy_path = save_best_copy(
                {"model": model.state_dict()}, best_copy_dir, epoch, config.local_checkpoint_keep
            )
            info.best_epoch = epoch
            info.best_smoothed_fg = smoothed
            info.best_epoch_val_fg = value
            print(
                "Epoch %d, new best smoothed %s=%.4f, local copy %s" % (epoch, config.best_metric, smoothed, copy_path)
            )
            if config.log_model and smoothed >= state["last_uploaded"] + config.mlflow_model_min_improvement:
                logger.log_model(model, "timber_unet_epoch_%d" % epoch, model.example_input(), step=epoch)
                state["last_uploaded"] = smoothed
                info.uploaded_models.append({"epoch": epoch, "smoothed": round(smoothed, 4)})
        if config.save_every_epochs and (epoch + 1) % config.save_every_epochs == 0:
            periodic_dir = best_copy_dir.parent / (best_copy_dir.name + "_periodic")
            periodic_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, periodic_dir / ("seg_head_epoch_%d.pth" % epoch))
        if ema is not None:
            ema.swap()

        if scheduler is not None and epoch >= config.warmup_epochs:
            scheduler.step()
        info.current_epoch = epoch
        info.last_epoch_minutes = round((time.monotonic() - epoch_start) / 60, 2)
        state["start_epoch"] = epoch + 1
        if config.resume:
            save_resume(resume_path, model, optimizer, scheduler, ema, state, info)
        logger.log_dict(info.to_dict(), "train_info.yaml")


    test_metrics: Dict[str, float] = {}
    if "test" in loaders:
        if state["best_epoch"] is not None:
            model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
        test_loss, test_metrics = evaluate(model, loaders["test"], normalize, device, classes, config)
        print("TEST loss %.4f %s" % (test_loss, " ".join("%s=%.3f" % item for item in test_metrics.items())))
        tracker.add(config.num_epochs, "test", test_loss, test_metrics["mean_iou_fg"], **test_metrics)
        logger.log_metrics(tracker.get(config.num_epochs, "test")[-1])
        info.test = {"loss": test_loss, **test_metrics}
        if config.viz_interval:
            log_panels(
                model, loaders["test"].dataset, "test", config.num_epochs, normalize, device, classes, config, logger
            )
        if config.log_model and state["best_epoch"] is not None and state["best_smoothed"] > state["last_uploaded"]:
            logger.log_model(
                model, "timber_unet_epoch_%d" % state["best_epoch"], model.example_input(), step=state["best_epoch"]
            )
    info.status = "finished"
    logger.log_dict(info.to_dict(), "train_info.yaml")
    logger.end()
    print("Best smoothed %s: %.4f (epoch %s)" % (config.best_metric, state["best_smoothed"], state["best_epoch"]))
    print("%s: %.4f" % (COMPLETION_MARKER, state["best_fg_iou"]))
    return test_metrics


def build_dataloaders(config: TimberTrainingConfig) -> Tuple[Dict[str, torch.utils.data.DataLoader], TimberClasses]:
    """Train (augmented, shuffled), train_eval (fixed subset of train), val and optional test loaders.

    Args:
        config: Training configuration.

    Returns:
        Tuple[Dict[str, torch.utils.data.DataLoader], TimberClasses]: loaders keyed by split, and
            the class table.
    """
    common = {"root": config.dataset_root, "min_wood_frac": config.min_wood_frac, "seed": config.seed}
    limit = config.max_tiles_per_split
    train_ds = TimberTileDataset(split=config.train_split, augment=config.augment, max_tiles=limit, **common)
    eval_limit = min(config.train_eval_tiles, limit) if limit else config.train_eval_tiles
    datasets = {
        "train": train_ds,
        "train_eval": TimberTileDataset(split=config.train_split, max_tiles=eval_limit, **common),
        "val": TimberTileDataset(split=config.val_split, max_tiles=limit, **common),
    }
    try:
        datasets["test"] = TimberTileDataset(split=config.test_split, max_tiles=limit, **common)
    except ValueError:
        print("No test tiles; skipping the test pass")
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            drop_last=split == "train" and len(dataset) > config.batch_size,
            num_workers=config.num_workers,
        )
        for split, dataset in datasets.items()
    }
    return loaders, train_ds.classes


def build_logger(config: TimberTrainingConfig) -> ILogger:
    loggers: List[ILogger] = []
    if config.use_local_logger:
        loggers.append(LocalLogger(log_dir=config.local_log_dir))
    if config.use_mlflow:
        loggers.append(
            MlflowLogger(
                experiment_name=config.mlflow_experiment_name,
                run_name=config.mlflow_run_name,
                tracking_uri=config.mlflow_tracking_uri,
            )
        )
    logger = CombinedLogger(loggers)
    logger.start()
    return logger


def compute_loss(
    logits: torch.Tensor, target: torch.Tensor, classes: TimberClasses, config: TimberTrainingConfig
) -> torch.Tensor:
    """Focal + soft Dice; ignored classes and padding (IGNORE_INDEX) are excluded from both.

    Args:
        logits: [B, C, H, W] class logits.
        target: [B, H, W] int64 targets.
        classes: Class table.
        config: Training configuration (loss weights, focal alpha/gamma).

    Returns:
        torch.Tensor: Scalar loss.
    """
    focal = masked_focal_loss(logits, target, IGNORE_INDEX, config.focal_alpha, config.focal_gamma)
    dice = masked_soft_dice_loss(logits, target, IGNORE_INDEX, classes.trained_indices)
    return config.distribution_loss_weight * focal + config.district_loss_weight * dice


def predict(model: torch.nn.Module, images: torch.Tensor, classes: TimberClasses) -> torch.Tensor:
    """Class logits with ignored and unused output channels suppressed.

    Args:
        model: Segmentation network.
        images: [B, 3, H, W] normalized input.
        classes: Class table.

    Returns:
        torch.Tensor: [B, C, H, W] float logits; channels of ignored classes are -inf.
    """
    logits = model(images).float()
    blocked = [i for i in range(classes.num_outputs) if i not in classes.trained_indices]
    if blocked:
        logits[:, blocked] = -math.inf
    return logits


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    normalize: torch.nn.Module,
    device: torch.device,
    classes: TimberClasses,
    config: TimberTrainingConfig,
) -> Tuple[float, Dict[str, float]]:
    """Loss, pixel IoU per trained class and per-class instance metrics of the defect classes.

    Args:
        model: Segmentation network.
        loader: Loader of the evaluated split.
        normalize: ImageNet normalization.
        device: Compute device.
        classes: Class table.
        config: Training configuration.

    Returns:
        Tuple[float, Dict[str, float]]: mean loss and metrics (see TimberInstanceEvaluator.summary
            plus iou_<class> and mean_iou_fg over trained non-background classes).
    """
    model.eval()
    names = classes.output_names()
    pixel = PerClassIoU(num_classes=classes.num_outputs, class_names=names)
    instances = TimberInstanceEvaluator(
        classes.defect_indices,
        [names[i] for i in classes.defect_indices],
        IGNORE_INDEX,
        config.instance_score_reduce,
        config.instance_min_pred_area,
    )
    losses: List[float] = []
    for batch in loader:
        target = batch["mask"].to(device)
        logits = predict(model, normalize(batch["image"].to(device)), classes)
        losses.append(compute_loss(logits.masked_fill(torch.isinf(logits), -1e4), target, classes, config).item())
        probs = logits.softmax(dim=1)
        prediction = probs.argmax(dim=1)
        pixel.update(torch.where(target == IGNORE_INDEX, IGNORE_INDEX, prediction).cpu(), target.cpu())
        probs_np, prediction_np, target_np = probs.cpu().numpy(), prediction.cpu().numpy(), target.cpu().numpy()
        for i in range(len(target_np)):
            instances.add(target_np[i], prediction_np[i], probs_np[i])
    iou = pixel.compute()
    metrics = {"iou_%s" % names[i]: iou["iou_%s" % names[i]] for i in classes.trained_indices}
    metrics["mean_iou_fg"] = float(np.mean([metrics["iou_%s" % names[i]] for i in classes.trained_indices if i != 0]))
    metrics["mean_iou_defects_px"] = float(np.mean([metrics["iou_%s" % names[i]] for i in classes.defect_indices]))
    metrics.update(instances.summary())
    return float(np.mean(losses)), metrics


@torch.no_grad()
def log_panels(
    model: torch.nn.Module,
    dataset: TimberTileDataset,
    split: str,
    step: int,
    normalize: torch.nn.Module,
    device: torch.device,
    classes: TimberClasses,
    config: TimberTrainingConfig,
    logger: ILogger,
) -> None:
    """Log input | ground truth | prediction panels for fixed, evenly spaced tiles of a split.

    Args:
        model: Segmentation network.
        dataset: Dataset of the split.
        split: Artifact folder.
        step: Epoch.
        normalize: ImageNet normalization.
        device: Compute device.
        classes: Class table (colours from classes.json).
        config: Training configuration.
        logger: Logger receiving the images.
    """
    if not config.viz_num_tiles:
        return
    model.eval()
    count = min(config.viz_num_tiles, len(dataset))
    indices = sorted({int((k + 0.5) * len(dataset) / count) for k in range(count)})
    samples = [dataset[i] for i in indices]
    images = torch.stack([s["image"] for s in samples])
    predictions = predict(model, normalize(images.to(device)), classes).argmax(dim=1).cpu().numpy()
    palette = classes.palette()
    for sample, prediction in zip(samples, predictions):
        rgb = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        target = sample["mask"].numpy()
        gt_rgb = palette[target]
        gt_rgb[target == IGNORE_INDEX] = IGNORE_COLOUR
        pred_rgb = palette[prediction]
        separator = np.full((rgb.shape[0], SEPARATOR_WIDTH, 3), 255, dtype=np.uint8)
        panel = np.concatenate([rgb, separator, gt_rgb, separator, pred_rgb], axis=1)
        logger.log_image(panel, "%s/%s" % (split, sample["tile_id"]), step)


def save_resume(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    ema: Optional[WeightEma],
    state: Dict[str, object],
    info: TrainInfo,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "ema": ema.state_dict() if ema is not None else None,
            "state": state,
            "train_info": asdict(info),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


if __name__ == "__main__":
    main()
