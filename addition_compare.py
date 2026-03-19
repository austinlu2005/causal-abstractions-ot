import os
from datetime import datetime
from pathlib import Path

from addition_experiment.backbone import AdditionTrainConfig
from addition_experiment.compare_runner import (
    CompareExperimentConfig,
    run_comparison_from_checkpoint,
)
from addition_experiment.runtime import resolve_device
from addition_experiment.scm import load_addition_problem


SEED = 42
DEVICE = "cuda"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / RUN_TIMESTAMP
CHECKPOINT_PATH = Path("models/addition_mlp.pt")
OUTPUT_PATH = RUN_DIR / "addition_compare_results.json"
SUMMARY_PATH = RUN_DIR / "addition_compare_summary.txt"

METHODS = ("gw", "ot", "fgw", "das")  # Any subset of: "gw", "ot", "fgw", "das"
FACTUAL_VALIDATION_SIZE = 4000
TRAIN_PAIR_SIZE = 1000
CALIBRATION_PAIR_SIZE = 1000
TEST_PAIR_SIZE = 5000
TARGET_VARS = ("S1", "C1", "S2", "C2")

BATCH_SIZE = 128
RESOLUTION = 1
FGW_ALPHA = 0.5
OT_TOP_K_VALUES = None # None sweeps all top-k values
import numpy as np
OT_LAMBDAS = tuple(np.linspace(0.25, 4.0, 16))

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 10
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = (1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192) # None means to sweep all subspace dims
DAS_LAYERS = None # None means to sweep all layers


def main() -> None:
    problem = load_addition_problem(run_checks=True)
    device = resolve_device(DEVICE)
    backbone_train_config = AdditionTrainConfig(
        seed=SEED,
        n_validation=FACTUAL_VALIDATION_SIZE,
        abstract_variables=tuple(TARGET_VARS),
    )
    compare_config = CompareExperimentConfig(
        seed=SEED,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=OUTPUT_PATH,
        summary_path=SUMMARY_PATH,
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
    run_comparison_from_checkpoint(
        problem=problem,
        device=device,
        backbone_train_config=backbone_train_config,
        config=compare_config,
    )


if __name__ == "__main__":
    main()
