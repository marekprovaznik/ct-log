import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from src.dataset.timber_tile_dataset import IGNORE_INDEX, TimberClasses, TimberTileDataset
from src.eval.timber_instances import TimberInstanceEvaluator
from src.loss.functional.masked_seg_loss import masked_focal_loss, masked_soft_dice_loss
import torch


@pytest.fixture
def mock_root(tmp_path: Path) -> Path:
    """Two-tile dataset with one ignored defect class (index 3) and 255 padding."""
    classes = [
        {"index": 0, "name": "background", "ignored": False},
        {"index": 1, "name": "wood", "ignored": False},
        {"index": 2, "name": "knot", "ignored": False},
        {"index": 3, "name": "steel", "ignored": True},
        {"index": 255, "name": "ignore", "ignored": True},
    ]
    (tmp_path / "classes.json").write_text(json.dumps(classes))
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    mask = np.ones((64, 64), np.uint8)
    mask[10:20, 10:20] = 2
    mask[40:50, 40:50] = 3
    mask[:, 60:] = 255
    for tile_id in ("a", "b"):
        Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(tmp_path / "images" / f"{tile_id}.png")
        Image.fromarray(mask, mode="L").save(tmp_path / "masks" / f"{tile_id}.png")
    (tmp_path / "tiles.csv").write_text(
        "tile_id,board,species,split,wood_frac\na,b1,spruce,val,0.9\nb,b2,larch,test,0.9\n"
    )
    return tmp_path


def test_dataset_maps_ignored_classes_to_ignore_index(mock_root: Path) -> None:
    """Ignored classes and padding become IGNORE_INDEX; trained classes keep their index."""
    sample = TimberTileDataset(str(mock_root), "val")[0]
    mask = sample["mask"]
    assert mask[15, 15] == 2
    assert mask[45, 45] == IGNORE_INDEX
    assert mask[0, 62] == IGNORE_INDEX
    assert sample["image"].shape == (3, 64, 64)


def test_classes_table(mock_root: Path) -> None:
    """Output count excludes 255; defect indices skip ignored classes."""
    classes = TimberClasses.from_json(mock_root / "classes.json")
    assert classes.num_outputs == 4
    assert classes.trained_indices == [0, 1, 2]
    assert classes.defect_indices == [2]


def test_dataset_rejects_unknown_mask_values(mock_root: Path) -> None:
    """A mask value missing from classes.json raises."""
    bad = np.full((64, 64), 7, np.uint8)
    Image.fromarray(bad, mode="L").save(mock_root / "masks" / "a.png")
    with pytest.raises(ValueError, match="not in classes.json"):
        TimberTileDataset(str(mock_root), "val")[0]


def test_instance_metrics_perfect_prediction() -> None:
    """Prediction equal to GT gives per-instance IoU, mAP50 and mAP50-95 of 1.0."""
    target = np.ones((64, 64), np.int64)
    target[5:15, 5:15] = 2
    target[30:40, 30:45] = 2
    target[50:60, 5:20] = 3
    target[:, 62:] = IGNORE_INDEX
    probs = np.zeros((4, 64, 64), np.float32)
    for index in range(4):
        probs[index][target == index] = 1.0
    evaluator = TimberInstanceEvaluator([2, 3], ["knot", "pith"], IGNORE_INDEX)
    evaluator.add(target, target.copy(), probs)
    metrics = evaluator.summary()
    for key in ("inst_iou_knot", "map50_knot", "map50_95_knot", "inst_iou_pith", "map50_pith", "inst_iou_defects"):
        assert metrics[key] == pytest.approx(1.0)
    assert metrics["n_gt_knot"] == 2


def test_instance_metrics_ignore_region_does_not_count() -> None:
    """Predictions inside IGNORE_INDEX pixels are not false positives."""
    target = np.ones((32, 32), np.int64)
    target[2:8, 2:8] = 2
    target[:, 20:] = IGNORE_INDEX
    prediction = target.copy()
    prediction[prediction == IGNORE_INDEX] = 2
    probs = np.ones((3, 32, 32), np.float32) / 3
    evaluator = TimberInstanceEvaluator([2], ["knot"], IGNORE_INDEX)
    evaluator.add(target, prediction, probs)
    metrics = evaluator.summary()
    assert metrics["n_pred_knot"] == 1
    assert metrics["map50_knot"] == pytest.approx(1.0)


def test_losses_skip_ignored_pixels() -> None:
    """Changing logits only at ignored pixels leaves both losses unchanged."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16)
    target = torch.randint(0, 3, (2, 16, 16))
    target[:, :, 8:] = IGNORE_INDEX
    perturbed = logits.clone()
    perturbed[:, :, :, 8:] += 10.0
    for fn in (
        lambda x: masked_focal_loss(x, target, IGNORE_INDEX, 2.0, 2.0),
        lambda x: masked_soft_dice_loss(x, target, IGNORE_INDEX, [0, 1, 2]),
    ):
        assert torch.isfinite(fn(logits))
        assert fn(logits) == pytest.approx(fn(perturbed).item())
