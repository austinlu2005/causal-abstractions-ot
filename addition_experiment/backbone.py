"""Backbone model training, loading, checkpointing, and factual evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from variable_width_mlp import (
    VariableWidthMLPConfig,
    VariableWidthMLPForClassification,
    load_variable_width_mlp_checkpoint,
)

from .constants import (
    DEFAULT_ACTIVATION,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_FACTUAL_TRAIN_SIZE,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_TARGET_VARS,
    INPUT_DIM,
    NUM_CLASSES,
)
from .runtime import ensure_parent_dir, set_seed
from .scm import AdditionProblem, compute_states_for_digits, digits_to_inputs_embeds, sample_digit_rows


@dataclass(frozen=True)
class AdditionTrainConfig:
    """Configuration for supervised training and loading of the addition backbone."""

    seed: int = 42
    n_train: int = DEFAULT_FACTUAL_TRAIN_SIZE
    n_validation: int = DEFAULT_FACTUAL_VALIDATION_SIZE
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    input_dim: int = INPUT_DIM
    num_classes: int = NUM_CLASSES
    dropout: float = DEFAULT_DROPOUT
    activation: str = DEFAULT_ACTIVATION
    abstract_variables: tuple[str, ...] = DEFAULT_TARGET_VARS
    learning_rate: float = 2e-3
    train_epochs: int = 20
    train_batch_size: int = 256
    eval_batch_size: int = 256
    verbose: bool = True


def build_factual_tensors(
    problem: AdditionProblem,
    size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample supervised addition examples and convert them to tensors."""
    digits = sample_digit_rows(size, seed)
    states = compute_states_for_digits(digits)
    inputs = digits_to_inputs_embeds(digits, problem.input_var_order)
    labels = torch.tensor(states["O"], dtype=torch.long)
    return inputs, labels


def evaluate_factual_model(
    model: VariableWidthMLPForClassification,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute factual exact-match accuracy for a trained model."""
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            logits = model(inputs_embeds=inputs[start:end].to(device).unsqueeze(1))[0]
            predictions.append(torch.argmax(logits, dim=1).detach().cpu())
    preds = torch.cat(predictions, dim=0)
    acc = float((preds == labels.view(-1)).to(torch.float32).mean().item())
    return {
        "exact_acc": acc,
        "num_examples": int(labels.numel()),
    }


def save_backbone_checkpoint(
    model: VariableWidthMLPForClassification,
    config: VariableWidthMLPConfig,
    checkpoint_path: str | Path,
    seed: int,
    abstract_variables: tuple[str, ...],
) -> None:
    """Write the trained MLP checkpoint and minimal metadata to disk."""
    ensure_parent_dir(checkpoint_path)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": config.to_dict(),
        "metadata": {
            "seed": seed,
            "task": "two_digit_base10_addition_onehot",
            "abstract_variables": list(abstract_variables),
            "scm_variables": ["A1", "B1", "A2", "B2", "S1", "C1", "S2", "C2"],
            "target": "O",
            "output_classes": list(range(NUM_CLASSES)),
        },
    }
    torch.save(payload, checkpoint_path)


def checkpoint_matches_train_config(
    checkpoint: dict[str, object],
    train_config: AdditionTrainConfig,
) -> bool:
    """Check whether an existing checkpoint matches the requested model spec."""
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict):
        return False
    return (
        int(model_config.get("input_dim", -1)) == int(train_config.input_dim)
        and tuple(int(dim) for dim in model_config.get("hidden_dims", [])) == tuple(train_config.hidden_dims)
        and int(model_config.get("num_classes", -1)) == int(train_config.num_classes)
        and str(model_config.get("activation", "")) == str(train_config.activation)
    )


def train_backbone(
    problem: AdditionProblem,
    train_config: AdditionTrainConfig,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
) -> tuple[VariableWidthMLPForClassification, VariableWidthMLPConfig, dict[str, object]]:
    """Train the backbone MLP and return the model, config, and metrics."""
    device = torch.device(device)
    set_seed(train_config.seed)

    x_train, y_train = build_factual_tensors(problem, train_config.n_train, train_config.seed + 1)
    x_validation, y_validation = build_factual_tensors(
        problem,
        train_config.n_validation,
        train_config.seed + 2,
    )

    config = VariableWidthMLPConfig(
        input_dim=train_config.input_dim,
        hidden_dims=list(train_config.hidden_dims),
        num_classes=train_config.num_classes,
        dropout=train_config.dropout,
        activation=train_config.activation,
    )
    model = VariableWidthMLPForClassification(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=train_config.train_batch_size,
        shuffle=True,
    )

    loss_history = []
    validation_history = []
    if train_config.verbose:
        print(
            "Backbone training "
            f"| device={device} "
            f"| train_examples={train_config.n_train} "
            f"| validation_examples={train_config.n_validation} "
            f"| hidden_dims={tuple(train_config.hidden_dims)} "
            f"| epochs={train_config.train_epochs}"
        )
    model.train()
    for epoch in range(train_config.train_epochs):
        running_loss = 0.0
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs_embeds=batch_inputs.unsqueeze(1))[0]
            loss = F.cross_entropy(logits, batch_labels.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * batch_inputs.shape[0]
        epoch_loss = running_loss / len(loader.dataset)
        loss_history.append(epoch_loss)
        epoch_validation_metrics = evaluate_factual_model(
            model=model,
            inputs=x_validation,
            labels=y_validation,
            batch_size=train_config.eval_batch_size,
            device=device,
        )
        validation_history.append(epoch_validation_metrics)
        if train_config.verbose:
            print(
                f"Epoch {epoch + 1}/{train_config.train_epochs} "
                f"| train_loss={epoch_loss:.4f} "
                f"| val_exact_acc={epoch_validation_metrics['exact_acc']:.4f}"
            )

    factual_metrics = validation_history[-1] if validation_history else evaluate_factual_model(
        model=model,
        inputs=x_validation,
        labels=y_validation,
        batch_size=train_config.eval_batch_size,
        device=device,
    )
    save_backbone_checkpoint(
        model,
        config,
        checkpoint_path,
        train_config.seed,
        train_config.abstract_variables,
    )
    return model, config, {
        "train_loss_history": loss_history,
        "validation_history": validation_history,
        "factual_validation_metrics": factual_metrics,
        "checkpoint_path": str(checkpoint_path),
    }


def load_backbone(
    problem: AdditionProblem,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
    train_config: AdditionTrainConfig | None = None,
) -> tuple[VariableWidthMLPForClassification, VariableWidthMLPConfig, dict[str, object]]:
    """Load an existing checkpoint and fail if it is missing or incompatible."""
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run addition_train.py first to create it."
        )

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    eval_config = train_config or AdditionTrainConfig()
    if not checkpoint_matches_train_config(checkpoint, eval_config):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not match the requested model spec. "
            "Run addition_train.py to regenerate models/addition_mlp.pt for the current config."
        )

    model, config, checkpoint = load_variable_width_mlp_checkpoint(str(checkpoint_path), device)
    x_validation, y_validation = build_factual_tensors(
        problem,
        eval_config.n_validation,
        eval_config.seed + 2,
    )
    factual_metrics = evaluate_factual_model(
        model=model,
        inputs=x_validation,
        labels=y_validation,
        batch_size=eval_config.eval_batch_size,
        device=device,
    )
    return model, config, {
        "checkpoint_path": str(checkpoint_path),
        "loaded_existing_checkpoint": True,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "factual_validation_metrics": factual_metrics,
    }
