from pathlib import Path

from addition_experiment.backbone import AdditionTrainConfig, train_backbone
from addition_experiment.runtime import resolve_device
from addition_experiment.scm import load_addition_problem


SEED = 2
DEVICE = "cuda"
CHECKPOINT_PATH = Path(f"models/addition_mlp_seed{SEED}.pt")

TRAIN_SIZE = 30000
VALIDATION_SIZE = 4000
HIDDEN_DIMS = [16, 16, 16]
TARGET_VARS = ["S1", "C1", "S2", "C2"]
LEARNING_RATE = 1e-3
EPOCHS = 500
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
    train_backbone(
        problem=problem,
        train_config=train_config,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )
    print(f"Wrote checkpoint to {CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()
