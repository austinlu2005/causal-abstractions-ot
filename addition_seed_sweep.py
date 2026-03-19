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


SEEDS = (41, 42, 43)
DEVICE = "cuda"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_seed_sweep"
CHECKPOINT_PATH_TEMPLATE = "models/addition_mlp_seed{seed}.pt"
OUTPUT_PATH = RUN_DIR / "addition_seed_sweep_results.json"
SUMMARY_PATH = RUN_DIR / "addition_seed_sweep_summary.txt"
RETRAIN_BACKBONES = False

METHODS = ("gw", "ot", "fgw", "das")
FACTUAL_TRAIN_SIZE = 30000
FACTUAL_VALIDATION_SIZE = 4000
HIDDEN_DIMS = (192, 192, 192, 192)
TARGET_VARS = ("S1", "C1", "S2", "C2")
LEARNING_RATE = 2e-3
EPOCHS = 100
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256

TRAIN_PAIR_SIZE = 1000
CALIBRATION_PAIR_SIZE = 1000
TEST_PAIR_SIZE = 5000

BATCH_SIZE = 128
RESOLUTION = 1
FGW_ALPHA = 0.5
OT_TOP_K_VALUES = None
OT_LAMBDAS = tuple(np.linspace(0.25, 4.0, 16))

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 10
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


def build_compare_config(seed: int, checkpoint_path: Path, run_dir: Path) -> CompareExperimentConfig:
    """Build the per-seed comparison config shared across all seeds in the sweep."""
    return CompareExperimentConfig(
        seed=seed,
        checkpoint_path=checkpoint_path,
        output_path=run_dir / "addition_compare_results.json",
        summary_path=run_dir / "addition_compare_summary.txt",
        methods=tuple(METHODS),
        factual_validation_size=FACTUAL_VALIDATION_SIZE,
        train_pair_size=TRAIN_PAIR_SIZE,
        calibration_pair_size=CALIBRATION_PAIR_SIZE,
        test_pair_size=TEST_PAIR_SIZE,
        target_vars=tuple(TARGET_VARS),
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        fgw_alpha=FGW_ALPHA,
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


def main() -> None:
    problem = load_addition_problem(run_checks=True)
    device = resolve_device(DEVICE)
    seed_runs = []

    for index, seed in enumerate(SEEDS, start=1):
        checkpoint_path = Path(CHECKPOINT_PATH_TEMPLATE.format(seed=seed))
        seed_run_dir = RUN_DIR / f"seed_{seed}"
        print(f"[{index}/{len(SEEDS)}] seed={seed} | checkpoint={checkpoint_path}")
        model, backbone_meta, backbone_source = load_or_train_backbone(
            problem=problem,
            device=device,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        comparison = run_comparison_with_model(
            problem=problem,
            model=model,
            backbone_meta=backbone_meta,
            device=device,
            config=build_compare_config(seed, checkpoint_path, seed_run_dir),
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

    payload = build_seed_sweep_payload(seed_runs)
    payload["device"] = str(device)
    payload["checkpoint_path_template"] = CHECKPOINT_PATH_TEMPLATE
    payload["retrain_backbones"] = RETRAIN_BACKBONES
    payload["training_config"] = {
        "train_size": FACTUAL_TRAIN_SIZE,
        "validation_size": FACTUAL_VALIDATION_SIZE,
        "hidden_dims": list(HIDDEN_DIMS),
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
    }
    payload["comparison_config"] = {
        "methods": list(METHODS),
        "train_pair_size": TRAIN_PAIR_SIZE,
        "calibration_pair_size": CALIBRATION_PAIR_SIZE,
        "test_pair_size": TEST_PAIR_SIZE,
        "batch_size": BATCH_SIZE,
        "resolution": RESOLUTION,
        "fgw_alpha": FGW_ALPHA,
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
    payload["summary_path"] = str(SUMMARY_PATH)
    payload["plots"] = save_seed_sweep_plots(payload, OUTPUT_PATH)
    write_json(OUTPUT_PATH, payload)
    write_text_report(SUMMARY_PATH, format_seed_sweep_summary(payload))

    print(f"Wrote seed sweep results to {Path(OUTPUT_PATH).resolve()}")
    print(f"Wrote seed sweep summary to {Path(SUMMARY_PATH).resolve()}")


if __name__ == "__main__":
    main()
