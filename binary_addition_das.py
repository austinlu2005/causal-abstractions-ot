"""Run DAS sweeps on the fixed binary-addition C1 benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from pyvene import RotatedSpaceIntervention
from torch.utils.data import DataLoader

from addition_experiment.pyvene_utils import build_intervenable
from addition_experiment.runtime import resolve_device, write_json
from binary_addition_common import (
    FACTUAL_CHECKPOINT,
    PairDataset,
    build_default_pair_banks,
    default_config,
    ensure_factual_model,
    iter_das_specs,
    metrics_from_binary_logits,
)
from variable_width_mlp import logits_from_output


DAS_LR_VALUES = (3e-4, 1e-3, 3e-3, 1e-2)
SELECTION_SCHEMES = (
    ("positive_only", 1.0, 0.0),
    ("invariant_only", 0.0, 1.0),
    ("weighted_0.50", 0.50, 0.50),
    ("weighted_0.70", 0.70, 0.30),
    ("weighted_0.80", 0.80, 0.20),
    ("weighted_0.90", 0.90, 0.10),
    ("weighted_0.95", 0.95, 0.05),
    ("weighted_0.99", 0.99, 0.01),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument("--checkpoint", type=Path, default=FACTUAL_CHECKPOINT, help="Checkpoint path.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain the factual checkpoint if needed.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for JSON summaries.")
    return parser.parse_args()


def train_das(intervenable, dataset: PairDataset, spec, config, device: torch.device, learning_rate: float) -> list[float]:
    optimizer_parameters = []
    for intervention in intervenable.interventions.values():
        if hasattr(intervention, "rotate_layer"):
            optimizer_parameters.append({"params": intervention.rotate_layer.parameters()})
    optimizer = torch.optim.Adam(optimizer_parameters, lr=float(learning_rate))
    loader = DataLoader(dataset, batch_size=config.das_batch_size, shuffle=True)
    losses = []
    best_loss = None
    plateau = 0
    for _epoch in range(config.das_max_epochs):
        epoch_losses = []
        for batch in loader:
            base = batch["input_ids"].to(device)
            source = batch["source_input_ids"].to(device)
            labels = batch["labels"].to(device).view(-1)
            positions = [[spec.position]] * base.shape[0]
            subspaces = [spec.subspace_dims] * base.shape[0]
            _, cf_output = intervenable(
                {"inputs_embeds": base.unsqueeze(1)},
                [{"inputs_embeds": source.unsqueeze(1)}],
                {"sources->base": ([positions], [positions])},
                subspaces=[subspaces],
            )
            logits = logits_from_output(cf_output)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(epoch_loss)
        improved = best_loss is None or epoch_loss < best_loss * (1.0 - config.das_plateau_rel_delta)
        if improved:
            best_loss = epoch_loss
            plateau = 0
        else:
            plateau += 1
        if len(losses) >= config.das_min_epochs and plateau >= config.das_plateau_patience:
            break
    return losses


def evaluate_das(intervenable, dataset: PairDataset, spec, config, device: torch.device) -> dict[str, float]:
    logits_all = []
    labels_all = []
    loader = DataLoader(dataset, batch_size=config.das_batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            base = batch["input_ids"].to(device)
            source = batch["source_input_ids"].to(device)
            positions = [[spec.position]] * base.shape[0]
            subspaces = [spec.subspace_dims] * base.shape[0]
            _, cf_output = intervenable(
                {"inputs_embeds": base.unsqueeze(1)},
                [{"inputs_embeds": source.unsqueeze(1)}],
                {"sources->base": ([positions], [positions])},
                subspaces=[subspaces],
            )
            logits_all.append(logits_from_output(cf_output).cpu())
            labels_all.append(batch["labels"].view(-1).cpu())
    return metrics_from_binary_logits(torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0))


def summarize_selection(
    records: list[dict[str, object]],
    scheme_name: str,
    positive_weight: float,
    invariant_weight: float,
) -> dict[str, object]:
    score_fn = lambda row: (
        positive_weight * float(row["calibration_positive_exact_acc"])
        + invariant_weight * float(row["calibration_invariant_exact_acc"])
    )
    best = max(
        records,
        key=lambda row: (
            score_fn(row),
            float(row["calibration_positive_exact_acc"]),
            float(row["calibration_invariant_exact_acc"]),
            float(row["calibration_positive_mean_shared_bits"]),
            float(row["calibration_invariant_mean_shared_bits"]),
        ),
    )
    return {
        "selection_scheme": scheme_name,
        "selection_positive_weight": float(positive_weight),
        "selection_invariant_weight": float(invariant_weight),
        "selection_score": float(score_fn(best)),
        "selected_site": best["site_label"],
        "layer": int(best["layer"]),
        "subspace_dim": int(best["subspace_dim"]),
        "learning_rate": float(best["learning_rate"]),
        "runtime_sec": float(best["runtime_sec"]),
        "train_epochs_ran": int(best["train_epochs_ran"]),
        "calibration_positive_exact_acc": float(best["calibration_positive_exact_acc"]),
        "calibration_invariant_exact_acc": float(best["calibration_invariant_exact_acc"]),
        "calibration_positive_mean_shared_bits": float(best["calibration_positive_mean_shared_bits"]),
        "calibration_invariant_mean_shared_bits": float(best["calibration_invariant_mean_shared_bits"]),
        "test_positive_exact_acc": float(best["test_positive_exact_acc"]),
        "test_invariant_exact_acc": float(best["test_invariant_exact_acc"]),
        "test_combined_exact_acc": 0.5 * (float(best["test_positive_exact_acc"]) + float(best["test_invariant_exact_acc"])),
        "test_positive_mean_shared_bits": float(best["test_positive_mean_shared_bits"]),
        "test_invariant_mean_shared_bits": float(best["test_invariant_mean_shared_bits"]),
        "selected_layer_mass_by_layer": {f"L{int(best['layer'])}": 1.0},
    }


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
    train_dataset = PairDataset(banks["fit"].base_inputs, banks["fit"].source_inputs, banks["fit"].cf_labels)
    calib_positive = PairDataset(
        banks["calibration_positive"].base_inputs,
        banks["calibration_positive"].source_inputs,
        banks["calibration_positive"].cf_labels,
    )
    calib_invariant = PairDataset(
        banks["calibration_invariant"].base_inputs,
        banks["calibration_invariant"].source_inputs,
        banks["calibration_invariant"].cf_labels,
    )
    test_positive = PairDataset(
        banks["test_positive"].base_inputs,
        banks["test_positive"].source_inputs,
        banks["test_positive"].cf_labels,
    )
    test_invariant = PairDataset(
        banks["test_invariant"].base_inputs,
        banks["test_invariant"].source_inputs,
        banks["test_invariant"].cf_labels,
    )

    specs = iter_das_specs(model)
    total_runtime_start = perf_counter()
    candidate_records = []
    total_candidates = len(specs) * len(DAS_LR_VALUES)
    for spec in specs:
        for learning_rate in DAS_LR_VALUES:
            candidate_start = perf_counter()
            intervention = RotatedSpaceIntervention(embed_dim=int(model.config.hidden_dims[spec.layer]))
            intervenable = build_intervenable(
                model=model,
                layer=spec.layer,
                component=spec.component,
                intervention=intervention,
                device=device,
                unit=spec.unit,
                max_units=spec.max_units,
                freeze_model=True,
                freeze_intervention=False,
                use_fast=False,
            )
            loss_history = train_das(intervenable, train_dataset, spec, config, device, learning_rate=learning_rate)
            calibration_positive_metrics = evaluate_das(intervenable, calib_positive, spec, config, device)
            calibration_invariant_metrics = evaluate_das(intervenable, calib_invariant, spec, config, device)
            test_positive_metrics = evaluate_das(intervenable, test_positive, spec, config, device)
            test_invariant_metrics = evaluate_das(intervenable, test_invariant, spec, config, device)
            record = {
                "site_label": spec.label,
                "layer": int(spec.layer),
                "subspace_dim": int(spec.subspace_dim),
                "learning_rate": float(learning_rate),
                "train_epochs_ran": int(len(loss_history)),
                "runtime_sec": float(perf_counter() - candidate_start),
                "calibration_positive_exact_acc": float(calibration_positive_metrics["exact_acc"]),
                "calibration_invariant_exact_acc": float(calibration_invariant_metrics["exact_acc"]),
                "calibration_positive_mean_shared_bits": float(calibration_positive_metrics["mean_shared_bits"]),
                "calibration_invariant_mean_shared_bits": float(calibration_invariant_metrics["mean_shared_bits"]),
                "test_positive_exact_acc": float(test_positive_metrics["exact_acc"]),
                "test_invariant_exact_acc": float(test_invariant_metrics["exact_acc"]),
                "test_positive_mean_shared_bits": float(test_positive_metrics["mean_shared_bits"]),
                "test_invariant_mean_shared_bits": float(test_invariant_metrics["mean_shared_bits"]),
            }
            candidate_records.append(record)
            print(
                f"[{len(candidate_records)}/{total_candidates}] {spec.label} lr={learning_rate:g} | "
                f"calib_pos={record['calibration_positive_exact_acc']:.4f} "
                f"calib_inv={record['calibration_invariant_exact_acc']:.4f} "
                f"test_pos={record['test_positive_exact_acc']:.4f} "
                f"test_inv={record['test_invariant_exact_acc']:.4f}"
            )

    selection_summaries = {
        scheme_name: summarize_selection(candidate_records, scheme_name, positive_weight, invariant_weight)
        for scheme_name, positive_weight, invariant_weight in SELECTION_SCHEMES
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / f"{timestamp}_binary_addition_c1_das"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "das_results.json"
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
            "learning_rates": list(DAS_LR_VALUES),
            "selection_schemes": [
                {
                    "name": scheme_name,
                    "positive_weight": positive_weight,
                    "invariant_weight": invariant_weight,
                }
                for scheme_name, positive_weight, invariant_weight in SELECTION_SCHEMES
            ],
        },
        "candidate_records": candidate_records,
        "selection_summaries": selection_summaries,
        "total_runtime_sec": float(perf_counter() - total_runtime_start),
    }
    write_json(out_path, payload)
    print(json.dumps({"json": str(out_path.resolve()), "selection_summaries": selection_summaries}, indent=2))


if __name__ == "__main__":
    main()
