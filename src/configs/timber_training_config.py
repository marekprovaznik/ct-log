from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
import yaml


class TimberTrainingConfig(BaseModel):
    """Configuration for 2D RGB defect segmentation on timber tile datasets (CT-13 format)."""

    dataset_root: str
    dataset_version: str = "timber_v1"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    min_wood_frac: float = Field(0.0, ge=0, le=1)
    max_tiles_per_split: Optional[int] = Field(None, gt=0)

    unet_arch: str = "Unet"
    unet_encoder: str = "tu-tf_efficientnetv2_s"
    unet_decoder_channels: Optional[List[int]] = None
    bf16: bool = True

    lr: float = Field(1e-4, gt=0)
    num_epochs: int = Field(..., gt=0)
    max_train_batches: Optional[int] = Field(None, gt=0)
    batch_size: int = Field(6, gt=0)
    num_workers: int = Field(0, ge=0)
    lr_schedule: Literal["none", "cosine"] = "cosine"
    warmup_epochs: int = Field(0, ge=0)
    ema_decay: Optional[float] = 0.9995
    augment: bool = True
    seed: int = 0

    distribution_loss_weight: float = Field(0.4, ge=0)
    district_loss_weight: float = Field(0.6, ge=0)
    focal_alpha: float = Field(2.0, gt=0)
    focal_gamma: float = Field(2.0, ge=0)

    instance_score_reduce: Literal["mean", "max"] = "mean"
    instance_min_pred_area: int = Field(16, ge=0)
    best_metric: str = "inst_iou_defects"
    smoothing_window: int = Field(5, gt=0)
    train_eval_interval: int = Field(0, ge=0)
    train_eval_tiles: int = Field(300, gt=0)

    checkpoint_path: Path
    resume: bool = True
    local_checkpoint_dir: Path = Path("/tmp/ctlog-checkpoints")
    local_checkpoint_keep: int = Field(3, gt=0)
    save_every_epochs: int = Field(0, ge=0)

    log_interval: int = Field(50, ge=0)
    viz_interval: int = Field(5, ge=0)
    viz_num_tiles: int = Field(4, ge=0)
    use_local_logger: bool = True
    local_log_dir: Path = Path("logs/timber")
    use_mlflow: bool = True
    mlflow_experiment_name: str = "ct-timber-v1"
    mlflow_run_name: Optional[str] = None
    mlflow_tracking_uri: str = "https://dagshub.com/AddictionLord/ct-log.mlflow"
    mlflow_model_min_improvement: float = Field(0.005, ge=0)
    log_model: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "TimberTrainingConfig":
        with open(path) as handle:
            return cls(**yaml.safe_load(handle))
