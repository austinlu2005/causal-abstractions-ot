import os
from datetime import datetime
from pathlib import Path

from addition_experiment.backbone import AdditionTrainConfig, train_backbone
from addition_experiment.runtime import resolve_device, write_json
from addition_experiment.scm import load_addition_problem


SEED = 42
DEVICE = "cuda"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / RUN_TIMESTAMP
CHECKPOINT_PATH = Path("models/addition_mlp.pt")
OUTPUT_PATH = RUN_DIR / "addition_train_results.json"

TRAIN_SIZE = 30000
VALIDATION_SIZE = 4000
HIDDEN_DIMS = (192, 192, 192, 192)
TARGET_VARS = ("S1", "C1", "S2", "C2")
LEARNING_RATE = 2e-3
EPOCHS = 100
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256


def main() -> None:
    problem = load_addition_problem(run_checks=True)
    device = resolve_device(DEVICE)
    train_config = AdditionTrainConfig(
        seed=SEED,
        n_train=TRAIN_SIZE,
        n_validation=VALIDATION_SIZE,
        hidden_dims=tuple(HIDDEN_DIMS),
        abstract_variables=tuple(TARGET_VARS),
        learning_rate=LEARNING_RATE,
        train_epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )
    _, _, payload = train_backbone(
        problem=problem,
        train_config=train_config,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )
    write_json(OUTPUT_PATH, payload)
    print(f"Wrote training results to {Path(OUTPUT_PATH).resolve()}")


if __name__ == "__main__":
    main()
