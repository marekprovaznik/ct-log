from typing import List

import torch
import torch.nn.functional as F


def masked_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    alpha: float = 1.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Multi-class focal loss averaged over pixels whose target is not ignore_index.

    Args:
        logits: [B, C, H, W] class logits.
        target: [B, H, W] int64 class ids, ignore_index where unsupervised.
        ignore_index: Target value excluded from the loss.
        alpha: Global scale.
        gamma: Focusing parameter.

    Returns:
        torch.Tensor: Scalar loss; zero (with grad) when no pixel is valid.
    """
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    log_probs = F.log_softmax(logits.float(), dim=1)
    log_p_t = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    focal = alpha * (1.0 - log_p_t.exp()) ** gamma * (-log_p_t)
    return focal[valid].mean()


def masked_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    classes: List[int],
    smooth: float = 1.0,
    present_only: bool = False,
) -> torch.Tensor:
    """Soft Dice loss over the listed classes, ignoring pixels whose target is ignore_index.

    Per-sample Dice with additive smoothing, averaged over samples and then classes, as
    multiclass_tversky_loss with alpha = beta = 0.5.

    Args:
        logits: [B, C, H, W] class logits.
        target: [B, H, W] int64 class ids, ignore_index where unsupervised.
        ignore_index: Target value excluded from the loss.
        classes: Output channels that enter the loss (ignored classes left out).
        smooth: Additive smoothing.
        present_only: Average only over (sample, class) pairs with ground-truth pixels of the
            class, so samples without a class do not reward predicting it nowhere; false positives
            are then left to the other loss terms.

    Returns:
        torch.Tensor: Scalar loss.
    """
    valid = (target != ignore_index).unsqueeze(1).float()
    probs = F.softmax(logits.float(), dim=1) * valid
    safe_target = torch.where(target != ignore_index, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float() * valid
    index = torch.tensor(classes, device=logits.device)
    probs = probs.index_select(1, index)
    one_hot = one_hot.index_select(1, index)
    inter = (probs * one_hot).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    dice = (2.0 * inter + smooth) / (denom + smooth)
    if present_only:
        present = one_hot.sum(dim=(2, 3)) > 0
        if not present.any():
            return logits.sum() * 0.0
        return 1.0 - dice[present].mean()
    return 1.0 - dice.mean()
