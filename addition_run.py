import os
from datetime import datetime
from pathlib import Path

import numpy as np

from addition_experiment.backbone import AdditionTrainConfig, load_backbone, train_backbone
from addition_experiment.compare_runner import CompareExperimentConfig, run_comparison_with_model
from addition_experiment.reporting import write_text_report
from addition_experiment.runtime import resolve_device, write_json
from addition_experiment.scm import load_addition_problem
from addition_experiment.seed_sweep import (
    build_seed_sweep_payload,
    format_seed_sweep_summary,
    save_seed_sweep_plots,
)


SEEDS = (42,)  # (41, 42, 43, 44, 45)
DEVICE = "mps"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_addition"
CHECKPOINT_PATH_TEMPLATE = "models/addition_mlp_seed{seed}.pt"
OUTPUT_PATH = RUN_DIR / "addition_seed_sweep_results.json"
SUMMARY_PATH = RUN_DIR / "addition_run_summary.txt"
RETRAIN_BACKBONES = False

# Methods included in the comparison run.
METHODS = ("ot", "gw", "fgw", "das")

# Abstract variables to align
TARGET_VARS = ("C1", "C2")

# Backbone training hyperparameters.
FACTUAL_TRAIN_SIZE = 30000
FACTUAL_VALIDATION_SIZE = 4000
HIDDEN_DIMS = (192, 192, 192, 192)
LEARNING_RATE = 1e-3
EPOCHS = 200
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256

# Pair-bank split sizes.
TRAIN_PAIR_SIZE = 200
CALIBRATION_PAIR_SIZE = 1000
TEST_PAIR_SIZE = 5000

# Training pair-bank construction hyperparameters.
# `TRAIN_PAIR_POLICY`:
#   - "mixed": target the positive carry fraction specified below
#   - "unfiltered": sample ordered pairs without balancing
# `*_PAIR_POLICY_TARGET` can be one of:
#   - "any", "C1", "C2", "both", "C1_only", "C2_only"
TRAIN_PAIR_POLICY_TARGET = "any"
TRAIN_MIXED_POSITIVE_FRACTION = 0.5
TRAIN_PAIR_POLICY = "unfiltered"  # "mixed" # "unfiltered"
TRAIN_PAIR_POOL_SIZE = 256

# Calibration pair-bank construction hyperparameters.
CALIBRATION_PAIR_POLICY_TARGET = "C1"
CALIBRATION_MIXED_POSITIVE_FRACTION = 1.0
CALIBRATION_PAIR_POLICY = "mixed"  # "mixed" # "unfiltered"
CALIBRATION_PAIR_POOL_SIZE = 256

# Test pair-bank construction hyperparameters.
TEST_PAIR_POLICY_TARGET = "C1"
TEST_MIXED_POSITIVE_FRACTION = 1.0
TEST_PAIR_POLICY = "mixed"  # "mixed" # "unfiltered"
TEST_PAIR_POOL_SIZE = 256

# Shared evaluation batch size.
BATCH_SIZE = 128

# OT / GW / FGW hyperparameters.
RESOLUTION = 1
FGW_ALPHA = 0.5
OT_EPSILONS = (0.05,) # (0.01, 0.03, 0.05, 0.1)
OT_TOP_K_VALUES = tuple(range(1, 10))
OT_LAMBDAS = tuple(np.arange(0.1, 2.0 + 1e-9, 0.1))

# DAS hyperparameters.
DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = (1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192)
DAS_LAYERS = None


def build_train_config(seed: int) -> AdditionTrainConfig:
    """Build the backbone training config shared across all seeds in the sweep."""
    return AdditionTrainConfig(
        seed=seed,
        n_train=FACTUAL_TRAIN_SIZE,
        n_validation=FACTUAL_VALIDATION_SIZE,
        hidden_dims=tuple(HIDDEN_DIMS),
        abstract_variables=tuple(TARGET_VARS),
        learning_rate=LEARNING_RATE,
        train_epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )


def _format_epsilon_tag(epsilon: float) -> str:
    """Build a stable directory/file tag for one epsilon sweep value."""
    return f"{float(epsilon):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def build_compare_config(
    seed: int,
    checkpoint_path: Path,
    run_dir: Path,
    ot_epsilon: float,
) -> CompareExperimentConfig:
    """Build the per-seed comparison config shared across all seeds in the sweep."""
    return CompareExperimentConfig(
        seed=seed,
        checkpoint_path=checkpoint_path,
        output_path=run_dir / "addition_run_results.json",
        summary_path=run_dir / "addition_run_summary.txt",
        methods=tuple(METHODS),
        factual_validation_size=FACTUAL_VALIDATION_SIZE,
        train_pair_size=TRAIN_PAIR_SIZE,
        calibration_pair_size=CALIBRATION_PAIR_SIZE,
        test_pair_size=TEST_PAIR_SIZE,
        target_vars=tuple(TARGET_VARS),
        train_pair_policy=TRAIN_PAIR_POLICY,
        train_pair_policy_target=TRAIN_PAIR_POLICY_TARGET,
        train_mixed_positive_fraction=TRAIN_MIXED_POSITIVE_FRACTION,
        train_pair_pool_size=TRAIN_PAIR_POOL_SIZE,
        calibration_pair_policy=CALIBRATION_PAIR_POLICY,
        calibration_pair_policy_target=CALIBRATION_PAIR_POLICY_TARGET,
        calibration_mixed_positive_fraction=CALIBRATION_MIXED_POSITIVE_FRACTION,
        calibration_pair_pool_size=CALIBRATION_PAIR_POOL_SIZE,
        test_pair_policy=TEST_PAIR_POLICY,
        test_pair_policy_target=TEST_PAIR_POLICY_TARGET,
        test_mixed_positive_fraction=TEST_MIXED_POSITIVE_FRACTION,
        test_pair_pool_size=TEST_PAIR_POOL_SIZE,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        fgw_alpha=FGW_ALPHA,
        ot_epsilon=ot_epsilon,
        ot_top_k_values=OT_TOP_K_VALUES,
        ot_lambdas=tuple(OT_LAMBDAS),
        das_max_epochs=DAS_MAX_EPOCHS,
        das_min_epochs=DAS_MIN_EPOCHS,
        das_plateau_patience=DAS_PLATEAU_PATIENCE,
        das_plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
        das_learning_rate=DAS_LEARNING_RATE,
        das_subspace_dims=DAS_SUBSPACE_DIMS,
        das_layers=DAS_LAYERS,
    )


def load_or_train_backbone(problem, device, seed: int, checkpoint_path: Path):
    """Train or reuse one backbone checkpoint for the requested seed."""
    train_config = build_train_config(seed)
    if RETRAIN_BACKBONES or not checkpoint_path.exists():
        model, _, backbone_meta = train_backbone(
            problem=problem,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        return model, backbone_meta, "trained"

    model, _, backbone_meta = load_backbone(
        problem=problem,
        checkpoint_path=checkpoint_path,
        device=device,
        train_config=train_config,
    )
    return model, backbone_meta, "loaded"


def print_loaded_backbone_validation(backbone_meta: dict[str, object]) -> None:
    """Print the factual validation accuracy for a reused checkpoint."""
    factual_metrics = dict(backbone_meta.get("factual_validation_metrics", {}))
    exact_acc = float(factual_metrics.get("exact_acc", 0.0))
    num_examples = int(factual_metrics.get("num_examples", 0))
    print(
        "Loaded backbone factual validation "
        f"| exact_acc={exact_acc:.4f} "
        f"| num_examples={num_examples}"
    )


def main() -> None:
    problem = load_addition_problem(run_checks=True)
    device = resolve_device(DEVICE)
    epsilon_sweeps = []
    epsilon_summary_sections = []
    main_summary_lines = [
        "Addition Run Summary",
        f"device: {device}",
        f"methods: {', '.join(METHODS)}",
        f"target_vars: {', '.join(TARGET_VARS)}",
        (
            "pair_sizes: "
            f"train={TRAIN_PAIR_SIZE}, "
            f"calibration={CALIBRATION_PAIR_SIZE}, "
            f"test={TEST_PAIR_SIZE}"
        ),
        f"train_pair_policy: {TRAIN_PAIR_POLICY}",
        f"train_pair_policy_target: {TRAIN_PAIR_POLICY_TARGET}",
        f"train_mixed_positive_fraction: {TRAIN_MIXED_POSITIVE_FRACTION}",
        f"train_pair_pool_size: {TRAIN_PAIR_POOL_SIZE}",
        f"calibration_pair_policy: {CALIBRATION_PAIR_POLICY}",
        f"calibration_pair_policy_target: {CALIBRATION_PAIR_POLICY_TARGET}",
        f"calibration_mixed_positive_fraction: {CALIBRATION_MIXED_POSITIVE_FRACTION}",
        f"calibration_pair_pool_size: {CALIBRATION_PAIR_POOL_SIZE}",
        f"test_pair_policy: {TEST_PAIR_POLICY}",
        f"test_pair_policy_target: {TEST_PAIR_POLICY_TARGET}",
        f"test_mixed_positive_fraction: {TEST_MIXED_POSITIVE_FRACTION}",
        f"test_pair_pool_size: {TEST_PAIR_POOL_SIZE}",
        f"batch_size: {BATCH_SIZE}",
        f"resolution: {RESOLUTION}",
        f"fgw_alpha: {FGW_ALPHA}",
        "ot_epsilons: " + ", ".join(f"{float(value):.6f}" for value in OT_EPSILONS),
        "ot_top_k_values: " + ", ".join(str(int(value)) for value in OT_TOP_K_VALUES),
        "ot_lambdas: " + ", ".join(f"{float(value):.6f}" for value in OT_LAMBDAS),
        f"das_max_epochs: {DAS_MAX_EPOCHS}",
        f"das_min_epochs: {DAS_MIN_EPOCHS}",
        f"das_plateau_patience: {DAS_PLATEAU_PATIENCE}",
        f"das_plateau_rel_delta: {DAS_PLATEAU_REL_DELTA}",
        f"das_learning_rate: {DAS_LEARNING_RATE}",
        (
            "das_subspace_dims: "
            + ("None" if DAS_SUBSPACE_DIMS is None else ", ".join(str(int(value)) for value in DAS_SUBSPACE_DIMS))
        ),
        f"das_layers: {DAS_LAYERS}",
        "",
    ]

    for epsilon_index, ot_epsilon in enumerate(OT_EPSILONS, start=1):
        epsilon_tag = _format_epsilon_tag(ot_epsilon)
        epsilon_run_dir = RUN_DIR / f"epsilon_{epsilon_tag}"
        epsilon_output_path = epsilon_run_dir / "addition_seed_sweep_results.json"
        epsilon_summary_path = epsilon_run_dir / "addition_run_summary.txt"
        seed_runs = []
        print(
            f"[epsilon {epsilon_index}/{len(OT_EPSILONS)}] "
            f"ot_epsilon={float(ot_epsilon):.6f}"
        )

        for seed_index, seed in enumerate(SEEDS, start=1):
            checkpoint_path = Path(CHECKPOINT_PATH_TEMPLATE.format(seed=seed))
            seed_run_dir = epsilon_run_dir / f"seed_{seed}"
            print(f"[{seed_index}/{len(SEEDS)}] seed={seed} | checkpoint={checkpoint_path}")
            model, backbone_meta, backbone_source = load_or_train_backbone(
                problem=problem,
                device=device,
                seed=seed,
                checkpoint_path=checkpoint_path,
            )
            if backbone_source == "loaded":
                print_loaded_backbone_validation(backbone_meta)
            comparison = run_comparison_with_model(
                problem=problem,
                model=model,
                backbone_meta=backbone_meta,
                device=device,
                config=build_compare_config(seed, checkpoint_path, seed_run_dir, ot_epsilon),
            )
            seed_runs.append(
                {
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_path),
                    "backbone_source": backbone_source,
                    "comparison": comparison,
                }
            )
            print()

        epsilon_payload = build_seed_sweep_payload(seed_runs)
        epsilon_payload["device"] = str(device)
        epsilon_payload["checkpoint_path_template"] = CHECKPOINT_PATH_TEMPLATE
        epsilon_payload["retrain_backbones"] = RETRAIN_BACKBONES
        epsilon_payload["ot_epsilon"] = float(ot_epsilon)
        epsilon_payload["training_config"] = {
            "train_size": FACTUAL_TRAIN_SIZE,
            "validation_size": FACTUAL_VALIDATION_SIZE,
            "hidden_dims": list(HIDDEN_DIMS),
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
        }
        epsilon_payload["comparison_config"] = {
            "methods": list(METHODS),
            "train_pair_size": TRAIN_PAIR_SIZE,
            "calibration_pair_size": CALIBRATION_PAIR_SIZE,
            "test_pair_size": TEST_PAIR_SIZE,
            "train_pair_policy": TRAIN_PAIR_POLICY,
            "train_pair_policy_target": TRAIN_PAIR_POLICY_TARGET,
            "train_mixed_positive_fraction": TRAIN_MIXED_POSITIVE_FRACTION,
            "train_pair_pool_size": TRAIN_PAIR_POOL_SIZE,
            "calibration_pair_policy": CALIBRATION_PAIR_POLICY,
            "calibration_pair_policy_target": CALIBRATION_PAIR_POLICY_TARGET,
            "calibration_mixed_positive_fraction": CALIBRATION_MIXED_POSITIVE_FRACTION,
            "calibration_pair_pool_size": CALIBRATION_PAIR_POOL_SIZE,
            "test_pair_policy": TEST_PAIR_POLICY,
            "test_pair_policy_target": TEST_PAIR_POLICY_TARGET,
            "test_mixed_positive_fraction": TEST_MIXED_POSITIVE_FRACTION,
            "test_pair_pool_size": TEST_PAIR_POOL_SIZE,
            "batch_size": BATCH_SIZE,
            "resolution": RESOLUTION,
            "fgw_alpha": FGW_ALPHA,
            "ot_epsilon": float(ot_epsilon),
            "ot_top_k_values": OT_TOP_K_VALUES,
            "ot_lambdas": list(OT_LAMBDAS),
            "das_max_epochs": DAS_MAX_EPOCHS,
            "das_min_epochs": DAS_MIN_EPOCHS,
            "das_plateau_patience": DAS_PLATEAU_PATIENCE,
            "das_plateau_rel_delta": DAS_PLATEAU_REL_DELTA,
            "das_learning_rate": DAS_LEARNING_RATE,
            "das_subspace_dims": None if DAS_SUBSPACE_DIMS is None else list(DAS_SUBSPACE_DIMS),
            "das_layers": DAS_LAYERS,
        }
        epsilon_payload["summary_path"] = str(epsilon_summary_path)
        epsilon_payload["plots"] = save_seed_sweep_plots(epsilon_payload, epsilon_output_path)
        write_json(epsilon_output_path, epsilon_payload)
        epsilon_summary_text = format_seed_sweep_summary(epsilon_payload)
        write_text_report(epsilon_summary_path, epsilon_summary_text)

        epsilon_sweeps.append(
            {
                "ot_epsilon": float(ot_epsilon),
                "output_path": str(epsilon_output_path),
                "summary_path": str(epsilon_summary_path),
                "payload": epsilon_payload,
            }
        )
        epsilon_summary_sections.append(
            "\n".join(
                [
                    f"OT epsilon sweep: {float(ot_epsilon):.6f}",
                    "",
                    epsilon_summary_text,
                ]
            )
        )

    payload = {
        "device": str(device),
        "checkpoint_path_template": CHECKPOINT_PATH_TEMPLATE,
        "retrain_backbones": RETRAIN_BACKBONES,
        "ot_epsilons": [float(value) for value in OT_EPSILONS],
        "target_vars": list(TARGET_VARS),
        "training_config": {
            "train_size": FACTUAL_TRAIN_SIZE,
            "validation_size": FACTUAL_VALIDATION_SIZE,
            "hidden_dims": list(HIDDEN_DIMS),
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
        },
        "comparison_config": {
            "methods": list(METHODS),
            "train_pair_size": TRAIN_PAIR_SIZE,
            "calibration_pair_size": CALIBRATION_PAIR_SIZE,
            "test_pair_size": TEST_PAIR_SIZE,
            "train_pair_policy": TRAIN_PAIR_POLICY,
            "train_pair_policy_target": TRAIN_PAIR_POLICY_TARGET,
            "train_mixed_positive_fraction": TRAIN_MIXED_POSITIVE_FRACTION,
            "train_pair_pool_size": TRAIN_PAIR_POOL_SIZE,
            "calibration_pair_policy": CALIBRATION_PAIR_POLICY,
            "calibration_pair_policy_target": CALIBRATION_PAIR_POLICY_TARGET,
            "calibration_mixed_positive_fraction": CALIBRATION_MIXED_POSITIVE_FRACTION,
            "calibration_pair_pool_size": CALIBRATION_PAIR_POOL_SIZE,
            "test_pair_policy": TEST_PAIR_POLICY,
            "test_pair_policy_target": TEST_PAIR_POLICY_TARGET,
            "test_mixed_positive_fraction": TEST_MIXED_POSITIVE_FRACTION,
            "test_pair_pool_size": TEST_PAIR_POOL_SIZE,
            "batch_size": BATCH_SIZE,
            "resolution": RESOLUTION,
            "fgw_alpha": FGW_ALPHA,
            "ot_epsilons": [float(value) for value in OT_EPSILONS],
            "ot_top_k_values": OT_TOP_K_VALUES,
            "ot_lambdas": list(OT_LAMBDAS),
            "das_max_epochs": DAS_MAX_EPOCHS,
            "das_min_epochs": DAS_MIN_EPOCHS,
            "das_plateau_patience": DAS_PLATEAU_PATIENCE,
            "das_plateau_rel_delta": DAS_PLATEAU_REL_DELTA,
            "das_learning_rate": DAS_LEARNING_RATE,
            "das_subspace_dims": None if DAS_SUBSPACE_DIMS is None else list(DAS_SUBSPACE_DIMS),
            "das_layers": DAS_LAYERS,
        },
        "epsilon_sweeps": epsilon_sweeps,
        "summary_path": str(SUMMARY_PATH),
    }
    write_json(OUTPUT_PATH, payload)
    write_text_report(
        SUMMARY_PATH,
        "\n".join(main_summary_lines)
        + "\n\n"
        + ("\n\n" + ("=" * 80) + "\n\n").join(epsilon_summary_sections),
    )

    print(f"Wrote run results to {Path(OUTPUT_PATH).resolve()}")
    print(f"Wrote run summary to {Path(SUMMARY_PATH).resolve()}")


if __name__ == "__main__":
    main()
