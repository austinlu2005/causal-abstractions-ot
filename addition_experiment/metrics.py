"""Metrics for exact-match and shared-digit counterfactual evaluation."""

from __future__ import annotations

import torch


def labels_to_digits(labels: torch.Tensor | list[int]) -> torch.Tensor:
    """Convert class labels into zero-padded hundreds, tens, and ones digits."""
    flat = torch.as_tensor(labels, dtype=torch.long).view(-1)
    hundreds = flat // 100
    tens = (flat // 10) % 10
    ones = flat % 10
    return torch.stack([hundreds, tens, ones], dim=1)


def shared_digit_counts(predictions: torch.Tensor | list[int], targets: torch.Tensor | list[int]) -> torch.Tensor:
    """Count how many output digits match for each prediction-target pair."""
    pred_digits = labels_to_digits(predictions)
    target_digits = labels_to_digits(targets)
    return (pred_digits == target_digits).to(torch.float32).sum(dim=1)


def exact_match_accuracy(predictions: torch.Tensor | list[int], targets: torch.Tensor | list[int]) -> float:
    """Compute exact class-match accuracy."""
    preds = torch.as_tensor(predictions, dtype=torch.long).view(-1)
    gold = torch.as_tensor(targets, dtype=torch.long).view(-1)
    return float((preds == gold).to(torch.float32).mean().item())


def mean_shared_digits(predictions: torch.Tensor | list[int], targets: torch.Tensor | list[int]) -> float:
    """Compute the mean number of matching output digits."""
    counts = shared_digit_counts(predictions, targets)
    return float(counts.mean().item())


def metrics_from_predictions(predictions: torch.Tensor | list[int], targets: torch.Tensor | list[int]) -> dict[str, float]:
    """Bundle the standard prediction-based evaluation metrics."""
    return {
        "exact_acc": exact_match_accuracy(predictions, targets),
        "mean_shared_digits": mean_shared_digits(predictions, targets),
    }


def metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor | list[int]) -> dict[str, float]:
    """Convert logits to predictions and compute the standard metrics."""
    predictions = torch.argmax(logits, dim=1)
    return metrics_from_predictions(predictions, targets)
