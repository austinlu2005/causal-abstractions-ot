"""Run OT and UOT sweeps on the fixed binary-addition C1 benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from pyvene import VanillaIntervention
from scipy.spatial.distance import cdist

from addition_experiment.pyvene_utils import CanonicalSite, build_intervenable, enumerate_canonical_sites, run_intervenable_logits
from addition_experiment.runtime import resolve_device, write_json
from binary_addition_common import (
    FACTUAL_CHECKPOINT,
    PairBank,
    build_default_pair_banks,
    default_config,
    ensure_factual_model,
    metrics_from_binary_logits,
)
from variable_width_mlp import VariableWidthMLPForClassification, logits_from_output


TAU_VALUES = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0)
UOT_REG_M_VALUES = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
RESOLUTION_VALUES = (1, 2, 5)
LAMBDA_VALUES = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0)
SELECTION_SCHEMES = (
    ("positive_only", 1.0, 0.0),
    ("invariant_only", 0.0, 1.0),
    ("w999", 0.999, 0.001),
    ("w995", 0.995, 0.005),
    ("w99", 0.99, 0.01),
    ("w975", 0.975, 0.025),
    ("w95", 0.95, 0.05),
    ("w90", 0.90, 0.10),
    ("w80", 0.80, 0.20),
    ("w70", 0.70, 0.30),
    ("w50", 0.50, 0.50),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument("--checkpoint", type=Path, default=FACTUAL_CHECKPOINT, help="Checkpoint path.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain the factual checkpoint if needed.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for JSON summaries.")
    return parser.parse_args()


def collect_base_logits(
    model: VariableWidthMLPForClassification,
    base_inputs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, base_inputs.shape[0], batch_size):
            batch = base_inputs[start : start + batch_size].to(device)
            output = model(inputs_embeds=batch.unsqueeze(1))
            outputs.append(logits_from_output(output).detach().cpu())
    return torch.cat(outputs, dim=0)


def build_variable_signature(bank: PairBank, num_classes: int) -> torch.Tensor:
    base_onehot = F.one_hot(bank.base_labels, num_classes=num_classes).to(torch.float32)
    cf_onehot = F.one_hot(bank.cf_labels, num_classes=num_classes).to(torch.float32)
    effect = (cf_onehot - base_onehot).permute(1, 0).contiguous()
    return effect.reshape(1, -1)


def collect_site_signatures(
    model: VariableWidthMLPForClassification,
    bank: PairBank,
    sites: list[CanonicalSite],
    base_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    signatures = []
    base_prob = torch.softmax(base_logits, dim=-1)
    for site in sites:
        intervenable = build_intervenable(
            model=model,
            layer=site.layer,
            component=site.component,
            intervention=VanillaIntervention(),
            device=device,
            unit=site.unit,
            max_units=site.max_units,
            freeze_model=True,
            freeze_intervention=True,
            use_fast=False,
        )
        site_logits = run_intervenable_logits(
            intervenable=intervenable,
            base_inputs=bank.base_inputs,
            source_inputs=bank.source_inputs,
            subspace_dims=site.subspace_dims,
            position=site.position,
            batch_size=batch_size,
            device=device,
        )
        effect = (torch.softmax(site_logits, dim=-1) - base_prob).permute(1, 0).contiguous()
        signatures.append(effect.reshape(-1))
    return torch.stack(signatures, dim=0)


def build_cross_cost(variable_signature: torch.Tensor, site_signatures: torch.Tensor) -> np.ndarray:
    cost = cdist(variable_signature.cpu().numpy(), site_signatures.cpu().numpy(), metric="cosine")
    cost = np.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)
    if float(cost.max()) > 0.0:
        cost = cost / float(cost.max())
    return cost


def sinkhorn_balanced(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    tau: float,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    kernel = np.exp(-cost / max(tau, 1e-12))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max_iter):
        previous_u = u.copy()
        u = a / (kernel @ v + 1e-12)
        v = b / (kernel.T @ u + 1e-12)
        if np.max(np.abs(u - previous_u)) < tol:
            break
    return np.diag(u) @ kernel @ np.diag(v)


def sinkhorn_unbalanced(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    tau: float,
    reg_m: float,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    kernel = np.exp(-cost / max(tau, 1e-12))
    u = np.ones_like(a)
    v = np.ones_like(b)
    power = reg_m / (reg_m + tau)
    for _ in range(max_iter):
        previous_u = u.copy()
        u = np.power(a / (kernel @ v + 1e-12), power)
        v = np.power(b / (kernel.T @ u + 1e-12), power)
        if np.max(np.abs(u - previous_u)) < tol:
            break
    return np.diag(u) @ kernel @ np.diag(v)


def normalize_transport_rows(transport: np.ndarray) -> np.ndarray:
    row_sums = transport.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0.0, row_sums, 1.0)
    return transport / safe


def truncate_transport_rows(normalized_transport: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    truncated = np.zeros_like(normalized_transport)
    k = max(1, min(int(top_k), normalized_transport.shape[1]))
    order = np.argsort(-normalized_transport[0], kind="stable")[:k]
    truncated[0, order] = normalized_transport[0, order]
    row_sum = float(truncated[0].sum())
    if row_sum > 0.0:
        truncated[0] = truncated[0] / row_sum
    return truncated, order


def build_layer_masks(
    model: VariableWidthMLPForClassification,
    sites: list[CanonicalSite],
    transport_weights: np.ndarray,
) -> dict[int, torch.Tensor]:
    masks = {layer: torch.zeros(int(model.config.hidden_dims[layer]), dtype=torch.float32) for layer in range(model.config.n_layer)}
    for site_index, site in enumerate(sites):
        weight = float(transport_weights[0, site_index])
        if weight <= 0.0:
            continue
        per_dim_weight = weight / float(len(site.dims))
        for dim in site.dims:
            masks[site.layer][dim] += per_dim_weight
    return {layer: mask for layer, mask in masks.items() if float(mask.sum().item()) > 0.0}


def run_soft_transport_intervention_logits(
    model: VariableWidthMLPForClassification,
    base_inputs: torch.Tensor,
    source_inputs: torch.Tensor,
    layer_masks: dict[int, torch.Tensor],
    strength: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    device_masks = {layer: mask.to(device=device, dtype=torch.float32).view(1, 1, -1) for layer, mask in layer_masks.items()}
    model.eval()
    with torch.no_grad():
        for start in range(0, base_inputs.shape[0], batch_size):
            end = min(start + batch_size, base_inputs.shape[0])
            base_hidden = base_inputs[start:end].to(device).unsqueeze(1)
            source_hidden = source_inputs[start:end].to(device).unsqueeze(1)
            for layer, block in enumerate(model.h):
                source_hidden = block(source_hidden)
                base_hidden = block(base_hidden)
                layer_mask = device_masks.get(layer)
                if layer_mask is not None:
                    base_hidden = base_hidden + float(strength) * layer_mask * (source_hidden - base_hidden)
            logits = model.score(base_hidden)
            if model.config.squeeze_output:
                logits = logits.squeeze(1)
            outputs.append(logits.detach().cpu())
    return torch.cat(outputs, dim=0)


def evaluate_transport(
    model: VariableWidthMLPForClassification,
    bank: PairBank,
    sites: list[CanonicalSite],
    transport_weights: np.ndarray,
    strength: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    layer_masks = build_layer_masks(model, sites, transport_weights)
    logits = run_soft_transport_intervention_logits(
        model,
        bank.base_inputs,
        bank.source_inputs,
        layer_masks,
        strength,
        batch_size,
        device,
    )
    metrics = metrics_from_binary_logits(logits, bank.cf_labels)
    scaled_layer_mass = {f"L{layer}": float(strength) * float(mask.sum().item()) for layer, mask in layer_masks.items()}
    total_scaled_mass = sum(scaled_layer_mass.values())
    if total_scaled_mass > 0.0:
        normalized_layer_mass = {key: value / total_scaled_mass for key, value in scaled_layer_mass.items()}
    else:
        normalized_layer_mass = {key: 0.0 for key in scaled_layer_mass}
    return {
        **metrics,
        "scaled_layer_mass_by_layer": scaled_layer_mass,
        "normalized_layer_mass_by_layer": normalized_layer_mass,
    }


def summarize_selected_sites(
    sites: list[CanonicalSite],
    truncated_transport: np.ndarray,
    selected_order: np.ndarray,
) -> list[str]:
    labels = []
    for index in selected_order.tolist():
        if float(truncated_transport[0, index]) > 0.0:
            labels.append(sites[index].label)
    return labels


def calibrate_topk_and_lambda(
    model: VariableWidthMLPForClassification,
    calibration_positive_bank: PairBank,
    calibration_invariant_bank: PairBank,
    sites: list[CanonicalSite],
    normalized_transport: np.ndarray,
    top_k_values: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    best_by_scheme: dict[str, dict[str, object]] = {}
    for top_k in top_k_values:
        truncated, selected_order = truncate_transport_rows(normalized_transport, top_k)
        for strength in LAMBDA_VALUES:
            positive_metrics = evaluate_transport(
                model, calibration_positive_bank, sites, truncated, float(strength), batch_size, device
            )
            invariant_metrics = evaluate_transport(
                model, calibration_invariant_bank, sites, truncated, float(strength), batch_size, device
            )
            candidate = {
                "selection_top_k": int(top_k),
                "selection_lambda": float(strength),
                "selected_site_labels": summarize_selected_sites(sites, truncated, selected_order),
                "selected_scaled_layer_mass_by_layer": positive_metrics["scaled_layer_mass_by_layer"],
                "selected_normalized_layer_mass_by_layer": positive_metrics["normalized_layer_mass_by_layer"],
                "calibration_positive_exact_acc": float(positive_metrics["exact_acc"]),
                "calibration_invariant_exact_acc": float(invariant_metrics["exact_acc"]),
                "calibration_positive_mean_shared_bits": float(positive_metrics["mean_shared_bits"]),
                "calibration_invariant_mean_shared_bits": float(invariant_metrics["mean_shared_bits"]),
            }
            for scheme_name, positive_weight, invariant_weight in SELECTION_SCHEMES:
                score = (
                    positive_weight * candidate["calibration_positive_exact_acc"]
                    + invariant_weight * candidate["calibration_invariant_exact_acc"]
                )
                enriched = {
                    **candidate,
                    "selection_scheme": scheme_name,
                    "selection_positive_weight": float(positive_weight),
                    "selection_invariant_weight": float(invariant_weight),
                    "selection_score": float(score),
                }
                incumbent = best_by_scheme.get(scheme_name)
                if incumbent is None or (
                    enriched["selection_score"],
                    enriched["calibration_positive_exact_acc"],
                    enriched["calibration_invariant_exact_acc"],
                    enriched["calibration_positive_mean_shared_bits"],
                    enriched["calibration_invariant_mean_shared_bits"],
                ) > (
                    incumbent["selection_score"],
                    incumbent["calibration_positive_exact_acc"],
                    incumbent["calibration_invariant_exact_acc"],
                    incumbent["calibration_positive_mean_shared_bits"],
                    incumbent["calibration_invariant_mean_shared_bits"],
                ):
                    best_by_scheme[scheme_name] = enriched
    return best_by_scheme


def summarize_record(
    method: str,
    tau: float,
    reg_m: float | None,
    resolution: int,
    selected: dict[str, object],
    test_positive_metrics: dict[str, object],
    test_invariant_metrics: dict[str, object],
    runtime_sec: float,
) -> dict[str, object]:
    return {
        "method": method,
        "tau": float(tau),
        "uot_reg_m": None if reg_m is None else float(reg_m),
        "site_resolution": int(resolution),
        "selection_scheme": selected["selection_scheme"],
        "selection_positive_weight": float(selected["selection_positive_weight"]),
        "selection_invariant_weight": float(selected["selection_invariant_weight"]),
        "selection_score": float(selected["selection_score"]),
        "selection_top_k": int(selected["selection_top_k"]),
        "selection_lambda": float(selected["selection_lambda"]),
        "selected_site_labels": selected["selected_site_labels"],
        "selected_scaled_layer_mass_by_layer": selected["selected_scaled_layer_mass_by_layer"],
        "selected_normalized_layer_mass_by_layer": selected["selected_normalized_layer_mass_by_layer"],
        "calibration_positive_exact_acc": float(selected["calibration_positive_exact_acc"]),
        "calibration_invariant_exact_acc": float(selected["calibration_invariant_exact_acc"]),
        "calibration_positive_mean_shared_bits": float(selected["calibration_positive_mean_shared_bits"]),
        "calibration_invariant_mean_shared_bits": float(selected["calibration_invariant_mean_shared_bits"]),
        "test_positive_exact_acc": float(test_positive_metrics["exact_acc"]),
        "test_invariant_exact_acc": float(test_invariant_metrics["exact_acc"]),
        "test_combined_exact_acc": 0.5 * (
            float(test_positive_metrics["exact_acc"]) + float(test_invariant_metrics["exact_acc"])
        ),
        "test_positive_mean_shared_bits": float(test_positive_metrics["mean_shared_bits"]),
        "test_invariant_mean_shared_bits": float(test_invariant_metrics["mean_shared_bits"]),
        "test_positive_layer_mass_by_layer": selected["selected_scaled_layer_mass_by_layer"],
        "test_invariant_layer_mass_by_layer": selected["selected_scaled_layer_mass_by_layer"],
        "runtime_sec": float(runtime_sec),
    }


def best_record_by_test_positive(records: list[dict[str, object]]) -> dict[str, object]:
    return max(
        records,
        key=lambda record: (
            float(record["test_positive_exact_acc"]),
            float(record["test_invariant_exact_acc"]),
            float(record["test_combined_exact_acc"]),
        ),
    )


def main() -> None:
    args = parse_args()
    config = default_config()
    device = resolve_device(args.device)
    model, model_payload, _ = ensure_factual_model(
        device=device,
        checkpoint_path=args.checkpoint,
        force_retrain=args.force_retrain,
        config=config,
    )
    banks = build_default_pair_banks(config)

    base_logits = collect_base_logits(model, banks["fit"].base_inputs, batch_size=128, device=device)
    variable_signature = build_variable_signature(banks["fit"], num_classes=int(model.config.num_classes))
    p = np.ones(variable_signature.shape[0], dtype=np.float64) / float(variable_signature.shape[0])

    sweep_records = []
    total_outer_runs = len(RESOLUTION_VALUES) * len(TAU_VALUES) * (1 + len(UOT_REG_M_VALUES))
    outer_index = 0
    total_runtime_start = perf_counter()
    for resolution in RESOLUTION_VALUES:
        sites = enumerate_canonical_sites(model, resolution=resolution)
        site_signatures = collect_site_signatures(model, banks["fit"], sites, base_logits, batch_size=128, device=device)
        cross_cost = build_cross_cost(variable_signature, site_signatures)
        q = np.ones(site_signatures.shape[0], dtype=np.float64) / float(site_signatures.shape[0])
        top_k_values = tuple(range(1, len(sites) + 1))
        for tau in TAU_VALUES:
            for method, reg_ms in (("ot", (None,)), ("uot", UOT_REG_M_VALUES)):
                for reg_m in reg_ms:
                    outer_index += 1
                    print(f"[{outer_index}/{total_outer_runs}] res={resolution} method={method} tau={tau} reg_m={reg_m}")
                    outer_start = perf_counter()
                    if method == "ot":
                        transport = sinkhorn_balanced(p, q, cross_cost, tau=float(tau))
                    else:
                        transport = sinkhorn_unbalanced(p, q, cross_cost, tau=float(tau), reg_m=float(reg_m))
                    normalized_transport = normalize_transport_rows(transport)
                    selected_by_scheme = calibrate_topk_and_lambda(
                        model=model,
                        calibration_positive_bank=banks["calibration_positive"],
                        calibration_invariant_bank=banks["calibration_invariant"],
                        sites=sites,
                        normalized_transport=normalized_transport,
                        top_k_values=top_k_values,
                        batch_size=128,
                        device=device,
                    )
                    runtime_sec = perf_counter() - outer_start
                    for scheme_name, selected in selected_by_scheme.items():
                        truncated, _ = truncate_transport_rows(normalized_transport, int(selected["selection_top_k"]))
                        test_positive_metrics = evaluate_transport(
                            model,
                            banks["test_positive"],
                            sites,
                            truncated,
                            float(selected["selection_lambda"]),
                            128,
                            device,
                        )
                        test_invariant_metrics = evaluate_transport(
                            model,
                            banks["test_invariant"],
                            sites,
                            truncated,
                            float(selected["selection_lambda"]),
                            128,
                            device,
                        )
                        sweep_records.append(
                            summarize_record(
                                method=method,
                                tau=float(tau),
                                reg_m=None if reg_m is None else float(reg_m),
                                resolution=resolution,
                                selected=selected,
                                test_positive_metrics=test_positive_metrics,
                                test_invariant_metrics=test_invariant_metrics,
                                runtime_sec=runtime_sec,
                            )
                        )

    best_by_method_and_scheme_test_positive: dict[str, dict[str, dict[str, object]]] = {}
    for method in ("ot", "uot"):
        bucket = {}
        for scheme_name, _positive_weight, _invariant_weight in SELECTION_SCHEMES:
            subset = [
                record
                for record in sweep_records
                if record["method"] == method and record["selection_scheme"] == scheme_name
            ]
            bucket[scheme_name] = best_record_by_test_positive(subset)
        best_by_method_and_scheme_test_positive[method] = bucket

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / f"{timestamp}_binary_addition_c1_ot_uot"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "ot_uot_results.json"
    payload = {
        "model": {
            "checkpoint_path": str(args.checkpoint),
            "hidden_dims": list(model.config.hidden_dims),
            "num_parameters": int(model_payload["num_parameters"]),
            "factual_exact_acc": float(model_payload["factual_exact_acc"]),
        },
        "benchmark": {
            "target_variable": "C1",
            "input_dim": int(model.config.input_dim),
            "num_classes": int(model.config.num_classes),
        },
        "banks": {name: bank.metadata() for name, bank in banks.items()},
        "sweep_hyperparameters": {
            "tau_values": list(TAU_VALUES),
            "uot_reg_m_values": list(UOT_REG_M_VALUES),
            "resolution_values": list(RESOLUTION_VALUES),
            "lambda_values": list(LAMBDA_VALUES),
            "selection_schemes": [
                {
                    "name": scheme_name,
                    "positive_weight": positive_weight,
                    "invariant_weight": invariant_weight,
                }
                for scheme_name, positive_weight, invariant_weight in SELECTION_SCHEMES
            ],
        },
        "sweep_records": sweep_records,
        "best_by_method_and_scheme_positive": best_by_method_and_scheme_test_positive,
        "best_by_method_and_scheme_test_positive": best_by_method_and_scheme_test_positive,
        "total_runtime_sec": float(perf_counter() - total_runtime_start),
    }
    write_json(out_path, payload)
    print(
        json.dumps(
            {"json": str(out_path.resolve()), "best_by_method_and_scheme_test_positive": best_by_method_and_scheme_test_positive},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
