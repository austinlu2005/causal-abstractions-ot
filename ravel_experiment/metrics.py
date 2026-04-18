"""RAVEL-specific label extraction and reporting metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import RAVELPairBank


def _gather_variant_logits(logits: torch.Tensor, variant_token_ids: torch.Tensor) -> torch.Tensor:
    if variant_token_ids.ndim == 2:
        variant_token_ids = variant_token_ids.unsqueeze(0).expand(logits.shape[0], -1, -1)
    batch_size, num_classes, num_variants = variant_token_ids.shape
    gathered = torch.gather(
        logits,
        dim=1,
        index=variant_token_ids.to(logits.device).reshape(batch_size, num_classes * num_variants),
    )
    gathered = gathered.reshape(batch_size, num_classes, num_variants)
    return gathered.max(dim=-1).values


def gather_variable_logits(logits: torch.Tensor, bank: RAVELPairBank) -> torch.Tensor:
    target_logits = _gather_variant_logits(logits, bank.candidate_variant_token_ids)
    mask = bank.candidate_mask.to(logits.device)
    return target_logits.masked_fill(~mask, float("-inf"))


def gather_global_candidate_logits(logits: torch.Tensor, bank: RAVELPairBank) -> torch.Tensor:
    return _gather_variant_logits(logits, bank.global_candidate_variant_token_ids)


def cross_entropy_for_bank(logits: torch.Tensor, bank: RAVELPairBank) -> torch.Tensor:
    target_logits = gather_variable_logits(logits, bank)
    return F.cross_entropy(target_logits, bank.labels.to(target_logits.device))


def cross_entropy_for_das(logits: torch.Tensor, answer_token_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, answer_token_ids.to(logits.device))


def _masked_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    if mask.numel() == 0 or not bool(mask.any()):
        return None
    return float((predictions[mask] == labels[mask]).float().mean().item())


def metrics_from_logits(logits: torch.Tensor, bank: RAVELPairBank, tokenizer=None) -> dict[str, float]:
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    labels = bank.labels.to(predictions.device)
    changed_mask = bank.changed_mask.to(predictions.device)
    invariant_mask = ~changed_mask
    exact_acc = float((predictions == labels).float().mean().item())
    metrics: dict[str, float] = {
        "exact_acc": exact_acc,
        "changed_count": float(changed_mask.sum().item()),
        "invariant_count": float(invariant_mask.sum().item()),
    }
    changed_acc = _masked_accuracy(predictions, labels, changed_mask)
    invariant_acc = _masked_accuracy(predictions, labels, invariant_mask)
    if changed_acc is not None:
        metrics["changed_exact_acc"] = float(changed_acc)
    if invariant_acc is not None:
        metrics["invariant_exact_acc"] = float(invariant_acc)
    if tokenizer is not None:
        decoded_acc = 0.0
        if bank.size:
            predicted_texts = [
                bank.candidate_texts[index][int(prediction)]
                for index, prediction in enumerate(predictions.detach().cpu().tolist())
            ]
            decoded_acc = sum(
                int(str(expected).strip() == str(predicted).strip())
                for expected, predicted in zip(bank.expected_answer_texts, predicted_texts)
            ) / len(predicted_texts)
        metrics["decoded_answer_acc"] = float(decoded_acc)
    return metrics


def das_metrics_from_logits(logits: torch.Tensor, bank: RAVELPairBank, tokenizer=None) -> dict[str, float]:
    return metrics_from_logits(logits, bank, tokenizer=tokenizer)


def prediction_details_from_logits(logits: torch.Tensor, bank: RAVELPairBank, tokenizer=None) -> dict[str, object]:
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    labels = bank.labels.to(predictions.device)
    predicted_token_ids = torch.gather(
        bank.candidate_token_ids.to(logits.device),
        dim=1,
        index=predictions.view(-1, 1),
    ).view(-1)
    predicted_texts = [
        bank.candidate_texts[index][int(prediction)]
        for index, prediction in enumerate(predictions.detach().cpu().tolist())
    ]
    details: dict[str, object] = {
        "labels": labels.detach().cpu().tolist(),
        "predictions": predictions.detach().cpu().tolist(),
        "correct": (predictions == labels).detach().cpu().to(torch.int64).tolist(),
        "target_logits": target_logits.detach().cpu().tolist(),
        "base_prompts": [str(item["prompt"]) for item in bank.base_inputs],
        "source_prompts": [str(item["prompt"]) for item in bank.source_inputs],
        "query_attributes": list(bank.query_attributes),
        "changed_mask": bank.changed_mask.detach().cpu().to(torch.int64).tolist(),
        "base_answer_texts": list(bank.base_answer_texts),
        "expected_answer_texts": list(bank.expected_answer_texts),
        "predicted_texts": predicted_texts,
        "predicted_token_ids": predicted_token_ids.detach().cpu().tolist(),
        "target_token_ids": bank.answer_token_ids.detach().cpu().tolist(),
    }
    if tokenizer is not None:
        details["predicted_first_token_text"] = [
            tokenizer.decode([int(token_id)]) for token_id in predicted_token_ids.detach().cpu().tolist()
        ]
    return details


def das_prediction_details_from_logits(logits: torch.Tensor, bank: RAVELPairBank, tokenizer=None) -> dict[str, object]:
    return prediction_details_from_logits(logits, bank, tokenizer=tokenizer)


def build_variable_signature(bank: RAVELPairBank, signature_mode: str) -> torch.Tensor:
    if signature_mode == "changed_mask":
        return bank.changed_mask.to(torch.float32)
    if signature_mode == "answer_token_delta":
        num_classes = int(bank.global_candidate_variant_token_ids.shape[0])
        base_onehot = F.one_hot(bank.base_global_labels.to(torch.long), num_classes=num_classes).to(torch.float32)
        source_onehot = F.one_hot(bank.expected_global_labels.to(torch.long), num_classes=num_classes).to(torch.float32)
        return (source_onehot - base_onehot).reshape(-1)
    raise ValueError(f"Unsupported signature_mode={signature_mode}")
