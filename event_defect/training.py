from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from .metrics import binary_metrics


def collate_detection_batch(batch):
    xs, targets = zip(*batch)
    return torch.stack(xs, dim=0), list(targets)


def boxes_to_heatmap(
    targets: Iterable[dict],
    height: int,
    width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    heatmaps = []
    for target in targets:
        heatmap = torch.zeros((1, height, width), dtype=torch.float32, device=device)
        boxes = target.get("boxes")
        if boxes is None:
            heatmaps.append(heatmap)
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)
        for box in boxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 >= x1 and y2 >= y1:
                heatmap[:, y1 : y2 + 1, x1 : x2 + 1] = 1.0
        heatmaps.append(heatmap)
    return torch.stack(heatmaps, dim=0)


def compute_detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict],
    heatmap_weight: float = 1.0,
    classification_weight: float = 0.5,
    anomaly_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap_logits = outputs["heatmap"]
    _, _, height, width = heatmap_logits.shape
    heatmap_target = boxes_to_heatmap(
        targets, height=height, width=width, device=heatmap_logits.device
    )
    heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_logits, heatmap_target)

    class_labels = torch.tensor(
        [int(target.get("class_id", target.get("is_defect", 0))) for target in targets],
        dtype=torch.long,
        device=outputs["class_logits"].device,
    )
    defect_labels = torch.tensor(
        [int(target.get("is_defect", 0)) for target in targets],
        dtype=torch.long,
        device=outputs["class_logits"].device,
    )
    class_loss = F.cross_entropy(outputs["class_logits"], class_labels)
    anomaly_loss = F.binary_cross_entropy_with_logits(
        outputs["anomaly_score"].flatten(), defect_labels.float()
    )
    loss = (
        heatmap_weight * heatmap_loss
        + classification_weight * class_loss
        + anomaly_weight * anomaly_loss
    )
    parts = {
        "heatmap": float(heatmap_loss.detach().cpu()),
        "classification": float(class_loss.detach().cpu()),
        "anomaly": float(anomaly_loss.detach().cpu()),
    }
    return loss, parts


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    moved = []
    for target in targets:
        moved_target = {}
        for key, value in target.items():
            moved_target[key] = value.to(device) if hasattr(value, "to") else value
        moved.append(moved_target)
    return moved


def train_one_epoch(model, loader, optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "heatmap": 0.0, "classification": 0.0, "anomaly": 0.0}
    count = 0
    for x, targets in loader:
        x = x.to(device)
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss, parts = compute_detection_loss(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = x.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "heatmap": 0.0, "classification": 0.0, "anomaly": 0.0}
    all_logits = []
    all_labels = []
    count = 0
    for x, targets in loader:
        x = x.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = model(x)
        loss, parts = compute_detection_loss(outputs, targets)
        labels = torch.tensor(
            [int(target.get("class_id", target.get("is_defect", 0))) for target in targets],
            dtype=torch.long,
            device=device,
        )
        pred = outputs["class_logits"].argmax(dim=1)
        batch_size = x.shape[0]
        all_logits.append(outputs["class_logits"].detach().cpu())
        all_labels.append(labels.detach().cpu())
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        count += batch_size

    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    if all_logits:
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        cls_pred = logits.argmax(dim=1)
        metrics["class_accuracy"] = float((cls_pred == labels).float().mean().item())
        metrics.update(binary_metrics(logits, labels))
    else:
        metrics.update(
            {
                "class_accuracy": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            }
        )
    return metrics
