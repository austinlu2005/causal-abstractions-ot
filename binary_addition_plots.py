"""Generate the binary-addition paper plots from DAS and OT/UOT results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from addition_experiment.runtime import resolve_device, write_json
from binary_addition_common import FACTUAL_CHECKPOINT, as_float_dict, compute_layer_probes, default_config, ensure_factual_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--das-results", type=Path, required=True, help="Path to das_results.json.")
    parser.add_argument("--ot-results", type=Path, required=True, help="Path to ot_uot_results.json.")
    parser.add_argument("--checkpoint", type=Path, default=FACTUAL_CHECKPOINT, help="Checkpoint path used for probe analysis.")
    parser.add_argument("--device", default="cpu", help="Torch device to use for probe analysis.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for generated plots.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def one_hot_layer_mass(layer: int, num_layers: int) -> dict[str, float]:
    return {f"L{idx}": 1.0 if idx == layer else 0.0 for idx in range(num_layers)}


def normalize_layer_mass(raw: dict[str, float], num_layers: int) -> dict[str, float]:
    dense = as_float_dict(raw, num_layers)
    total = sum(dense.values())
    if total > 0.0:
        return {key: value / total for key, value in dense.items()}
    return dense


def get_ot_bucket(ot_uot: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    if "best_by_method_and_scheme_test_positive" in ot_uot:
        return ot_uot["best_by_method_and_scheme_test_positive"]
    return ot_uot["best_by_method_and_scheme_positive"]


def get_record_layer_mass(record: dict[str, object], num_layers: int) -> dict[str, float]:
    if "selected_normalized_layer_mass_by_layer" in record:
        return as_float_dict(record["selected_normalized_layer_mass_by_layer"], num_layers)
    if "test_positive_layer_mass_by_layer" in record:
        return normalize_layer_mass(record["test_positive_layer_mass_by_layer"], num_layers)
    return {f"L{layer}": 0.0 for layer in range(num_layers)}


def plot_accuracy(summary_rows: list[tuple[str, float, float]], out_path: Path) -> None:
    labels = [label for label, _sens, _inv in summary_rows]
    sensitivity = [sens for _label, sens, _inv in summary_rows]
    invariance = [inv for _label, _sens, inv in summary_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(x - width / 2, sensitivity, width=width, label="Sensitivity", color="#e15759")
    ax.bar(x + width / 2, invariance, width=width, label="Invariance", color="#4e79a7")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Exact Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_runtime(runtime_rows: list[tuple[str, float]], out_path: Path) -> None:
    labels = [label for label, _value in runtime_rows]
    runtimes = [value for _label, value in runtime_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(x, runtimes, color="#59a14f")
    ax.set_ylabel("Runtime (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_cross_layer(
    layer_probes: list[float],
    ot_positive: dict[str, float],
    uot_positive: dict[str, float],
    das_positive: dict[str, float],
    das_balanced: dict[str, float],
    out_path: Path,
) -> None:
    labels = ["Layer Probes", "OT pos.", "UOT pos.", "DAS pos.", "DAS bal."]
    num_layers = len(layer_probes)
    series = [
        {f"L{layer}": float(layer_probes[layer]) for layer in range(num_layers)},
        ot_positive,
        uot_positive,
        das_positive,
        das_balanced,
    ]
    x = np.arange(len(labels))
    width = 0.30
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for layer in range(num_layers):
        values = [entry[f"L{layer}"] for entry in series]
        ax.bar(x + (layer - (num_layers - 1) / 2) * width, values, width=width, label=f"L{layer}")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probe / Handle Mass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    das = load_json(args.das_results)
    ot_uot = load_json(args.ot_results)

    ot_bucket = get_ot_bucket(ot_uot)
    ot_positive = ot_bucket["ot"]["positive_only"]
    uot_positive = ot_bucket["uot"]["positive_only"]
    ot_balanced = ot_bucket["ot"]["w70"]
    uot_balanced = ot_bucket["uot"]["w70"]
    das_positive = das["selection_summaries"]["positive_only"]
    das_balanced = das["selection_summaries"]["weighted_0.50"]

    summary_rows = [
        ("OT pos.", float(ot_positive["test_positive_exact_acc"]), float(ot_positive["test_invariant_exact_acc"])),
        ("UOT pos.", float(uot_positive["test_positive_exact_acc"]), float(uot_positive["test_invariant_exact_acc"])),
        ("DAS pos.", float(das_positive["test_positive_exact_acc"]), float(das_positive["test_invariant_exact_acc"])),
        ("OT bal.", float(ot_balanced["test_positive_exact_acc"]), float(ot_balanced["test_invariant_exact_acc"])),
        ("UOT bal.", float(uot_balanced["test_positive_exact_acc"]), float(uot_balanced["test_invariant_exact_acc"])),
        ("DAS bal.", float(das_balanced["test_positive_exact_acc"]), float(das_balanced["test_invariant_exact_acc"])),
    ]
    runtime_rows = [
        ("OT pos.", float(ot_positive["runtime_sec"])),
        ("UOT pos.", float(uot_positive["runtime_sec"])),
        ("DAS pos.", float(das_positive["runtime_sec"])),
        ("OT bal.", float(ot_balanced["runtime_sec"])),
        ("UOT bal.", float(uot_balanced["runtime_sec"])),
        ("DAS bal.", float(das_balanced["runtime_sec"])),
    ]

    device = resolve_device(args.device)
    model, _payload, _trained_now = ensure_factual_model(
        device=device,
        checkpoint_path=args.checkpoint,
        force_retrain=False,
        config=default_config(),
    )
    if "model" in das and "hidden_dims" in das["model"]:
        num_layers = len(das["model"]["hidden_dims"])
    else:
        num_layers = len(model.config.hidden_dims)
    layer_probe_results = compute_layer_probes(model, default_config(), device)
    layer_probe_values = [float(row["test_acc"]) for row in layer_probe_results]

    ot_positive_mass = get_record_layer_mass(ot_positive, num_layers)
    uot_positive_mass = get_record_layer_mass(uot_positive, num_layers)
    das_positive_mass = one_hot_layer_mass(int(das_positive["layer"]), num_layers)
    das_balanced_mass = one_hot_layer_mass(int(das_balanced["layer"]), num_layers)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / f"{timestamp}_binary_addition_c1_plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    accuracy_path = run_dir / "binary_addition_accuracy.png"
    runtime_path = run_dir / "binary_addition_runtime.png"
    cross_layer_path = run_dir / "binary_addition_cross_layer.png"

    plot_accuracy(summary_rows, accuracy_path)
    plot_runtime(runtime_rows, runtime_path)
    plot_cross_layer(
        layer_probe_values,
        ot_positive_mass,
        uot_positive_mass,
        das_positive_mass,
        das_balanced_mass,
        cross_layer_path,
    )

    summary_path = run_dir / "binary_addition_plot_summary.json"
    summary_payload = {
        "accuracy_rows": [
            {
                "label": label,
                "sensitivity": sensitivity,
                "invariance": invariance,
            }
            for label, sensitivity, invariance in summary_rows
        ],
        "runtime_rows": [
            {
                "label": label,
                "runtime_sec": runtime_sec,
            }
            for label, runtime_sec in runtime_rows
        ],
        "layer_probes": layer_probe_results,
        "cross_layer_series": {
            "layer_probes": {f"L{layer}": value for layer, value in enumerate(layer_probe_values)},
            "ot_positive": ot_positive_mass,
            "uot_positive": uot_positive_mass,
            "das_positive": das_positive_mass,
            "das_balanced": das_balanced_mass,
        },
        "selected_supports": {
            "ot_positive": ot_positive.get("selected_site_labels", []),
            "uot_positive": uot_positive.get("selected_site_labels", []),
            "das_positive": [das_positive["selected_site"]],
            "das_balanced": [das_balanced["selected_site"]],
        },
        "output_paths": {
            "accuracy": str(accuracy_path),
            "runtime": str(runtime_path),
            "cross_layer": str(cross_layer_path),
        },
    }
    write_json(summary_path, summary_payload)
    print(json.dumps({"json": str(summary_path.resolve()), "output_paths": summary_payload["output_paths"]}, indent=2))


if __name__ == "__main__":
    main()
