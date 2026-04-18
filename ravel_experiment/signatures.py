"""RAVEL residual-site effect signatures."""

from __future__ import annotations

import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from mcqa_experiment.intervention import forward_factual_logits, run_soft_residual_intervention

from .data import RAVELPairBank
from .metrics import gather_global_candidate_logits
from mcqa_experiment.sites import ResidualSite


def _per_example_kl(counterfactual_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    counterfactual_log_probs = torch.log_softmax(counterfactual_logits, dim=-1)
    counterfactual_probs = counterfactual_log_probs.exp()
    return torch.sum(counterfactual_probs * (counterfactual_log_probs - base_log_probs), dim=-1)


def signature_from_logits(
    *,
    counterfactual_logits: torch.Tensor,
    base_logits: torch.Tensor,
    bank: RAVELPairBank,
    signature_mode: str,
) -> torch.Tensor:
    if signature_mode == "whole_vocab_kl_t1":
        return _per_example_kl(counterfactual_logits, base_logits).reshape(-1)
    if signature_mode == "answer_token_delta":
        counterfactual_target_logits = gather_global_candidate_logits(counterfactual_logits, bank)
        base_target_logits = gather_global_candidate_logits(base_logits, bank)
        return (counterfactual_target_logits - base_target_logits).reshape(-1)
    if signature_mode == "changed_mask":
        return bank.changed_mask.to(torch.float32).reshape(-1)
    raise ValueError(f"Unsupported signature_mode={signature_mode}")


def collect_base_logits(
    *,
    model,
    bank: RAVELPairBank,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, bank.size, batch_size):
            end = min(start + batch_size, bank.size)
            logits = forward_factual_logits(
                model=model,
                input_ids=bank.base_input_ids[start:end].to(device),
                attention_mask=bank.base_attention_mask[start:end].to(device),
            )
            outputs.append(logits.detach().cpu())
    return torch.cat(outputs, dim=0)


def collect_site_signatures(
    *,
    model,
    bank: RAVELPairBank,
    sites: list[ResidualSite],
    base_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
    signature_mode: str,
    show_progress: bool = False,
) -> torch.Tensor:
    signatures = []
    site_iterator = sites
    if show_progress and tqdm is not None:
        site_iterator = tqdm(sites, desc=f"Building {signature_mode} signatures", leave=False)
    with torch.no_grad():
        for site in site_iterator:
            site_logits_chunks = []
            for start in range(0, bank.size, batch_size):
                end = min(start + batch_size, bank.size)
                logits = run_soft_residual_intervention(
                    model=model,
                    base_input_ids=bank.base_input_ids[start:end].to(device),
                    base_attention_mask=bank.base_attention_mask[start:end].to(device),
                    source_input_ids=bank.source_input_ids[start:end].to(device),
                    source_attention_mask=bank.source_attention_mask[start:end].to(device),
                    site_weights={site: 1.0},
                    strength=1.0,
                    base_position_by_id={
                        key: value[start:end] for key, value in bank.base_position_by_id.items()
                    },
                    source_position_by_id={
                        key: value[start:end] for key, value in bank.source_position_by_id.items()
                    },
                )
                site_logits_chunks.append(logits.detach().cpu())
            site_logits = torch.cat(site_logits_chunks, dim=0)
            signatures.append(
                signature_from_logits(
                    counterfactual_logits=site_logits,
                    base_logits=base_logits,
                    bank=bank,
                    signature_mode=signature_mode,
                )
            )
    return torch.stack(signatures, dim=0)
