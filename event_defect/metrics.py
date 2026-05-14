from __future__ import annotations

import torch


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    """Compute defect-vs-normal metrics from multiclass logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    pred_class = logits.argmax(dim=1)
    pred_defect = pred_class != 0
    true_defect = labels != 0

    tp = int((pred_defect & true_defect).sum().item())
    fp = int((pred_defect & ~true_defect).sum().item())
    tn = int((~pred_defect & ~true_defect).sum().item())
    fn = int((~pred_defect & true_defect).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
