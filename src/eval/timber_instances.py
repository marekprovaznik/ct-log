from typing import Dict, List

import numpy as np
from scipy import ndimage as ndi

from src.eval.knot_instances import S8, KnotInstanceEvaluator


class TimberInstanceEvaluator:
    """Per-class instance metrics for multi-class defect masks, one KnotInstanceEvaluator per class.

    Instances are 8-connected components of (target == c) and (argmax == c) per tile; the
    prediction is clipped to the supervised pixels (target != ignore). Each component is scored by
    its mean class probability. Predicted components smaller than min_pred_area px are dropped
    (GT is kept as is), so speckle from early epochs does not blow up the AP matching. Classes without GT instances in the split get NaN and are left out
    of the macro means.
    """

    def __init__(
        self,
        class_indices: List[int],
        class_names: List[str],
        ignore_index: int,
        score_reduce: str = "mean",
        min_pred_area: int = 0,
    ):
        """Create one evaluator per defect class.

        Args:
            class_indices: Output indices of the evaluated classes.
            class_names: Names of those classes, used in the metric keys.
            ignore_index: Target value of unsupervised pixels.
            score_reduce: "mean" or "max" of the class probability over a component.
            min_pred_area: Drop predicted components with fewer pixels (0 keeps all).
        """
        self.class_indices = class_indices
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.min_pred_area = min_pred_area
        self.evaluators = {index: KnotInstanceEvaluator(score_reduce=score_reduce) for index in class_indices}

    def add(self, target: np.ndarray, prediction: np.ndarray, probs: np.ndarray) -> None:
        """Add one tile.

        Args:
            target: [H, W] int class ids, ignore_index where unsupervised.
            prediction: [H, W] int predicted class ids.
            probs: [C, H, W] float class probabilities.
        """
        valid = target != self.ignore_index
        for index, evaluator in self.evaluators.items():
            pred = (prediction == index) & valid
            if self.min_pred_area > 1:
                labels, count = ndi.label(pred, structure=S8)
                if count:
                    areas = np.bincount(labels.ravel(), minlength=count + 1)
                    keep = areas >= self.min_pred_area
                    keep[0] = False
                    pred = keep[labels]
            evaluator.add(target == index, pred, score_map=probs[index])

    def summary(self) -> Dict[str, float]:
        """Flat metrics per class and macro means over classes with GT.

        Returns:
            Dict[str, float]: per class <name>: inst_iou_<name>, map50_<name>, map50_95_<name>,
                inst_f1_50_<name>, n_gt_<name>, n_pred_<name>; macro: inst_iou_defects,
                map50_defects, map50_95_defects.
        """
        out: Dict[str, float] = {}
        macro: Dict[str, List[float]] = {"inst_iou": [], "map50": [], "map50_95": []}
        for index, name in zip(self.class_indices, self.class_names):
            result = self.evaluators[index].result()
            n_gt = int(result["n_gt"])
            out[f"n_gt_{name}"] = float(n_gt)
            out[f"n_pred_{name}"] = float(result["n_pred"])
            if n_gt == 0:
                values = {"inst_iou": float("nan"), "map50": float("nan"), "map50_95": float("nan")}
                out[f"inst_f1_50_{name}"] = float("nan")
            else:
                values = {
                    "inst_iou": float(result["iou_0.3"]["mean_per_gt_iou"] or 0.0),
                    "map50": float(result["map50"]),
                    "map50_95": float(result["map50_95"]),
                }
                out[f"inst_f1_50_{name}"] = float(result["iou_0.5"]["f1"])
                for key, value in values.items():
                    macro[key].append(value)
            for key, value in values.items():
                out[f"{key}_{name}"] = value
        for key, values in macro.items():
            out[f"{key}_defects"] = float(np.mean(values)) if values else float("nan")
        return out
