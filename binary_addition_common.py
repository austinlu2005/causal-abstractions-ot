"""Shared utilities for the fixed binary-addition C1 benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from addition_experiment.pyvene_utils import DASSearchSpec
from addition_experiment.runtime import set_seed
from variable_width_mlp import VariableWidthMLPConfig, VariableWidthMLPForClassification


SEED = 42
TARGET_VAR = "C1"
INPUT_DIM = 12
NUM_CLASSES = 16
CANONICAL_INPUT_VARS = ("A2", "A1", "A0", "B2", "B1", "B0")
FACTUAL_HIDDEN_DIMS = (13, 13)
FACTUAL_CHECKPOINT = Path("models") / "binary_addition_c1_d2w13_seed42.pt"


@dataclass(frozen=True)
class BinaryAdditionConfig:
    seed: int = SEED
    hidden_dims: tuple[int, ...] = FACTUAL_HIDDEN_DIMS
    learning_rate: float = 2e-3
    max_epochs: int = 500
    train_batch_size: int = 64
    eval_batch_size: int = 64
    perfect_streak: int = 20
    train_bases: int = 40
    validation_bases: int = 12
    test_bases: int = 12
    das_batch_size: int = 128
    das_max_epochs: int = 1000
    das_min_epochs: int = 5
    das_plateau_patience: int = 3
    das_plateau_rel_delta: float = 5e-3


@dataclass(frozen=True)
class PairBank:
    name: str
    seed: int
    base_rows: np.ndarray
    source_rows: np.ndarray
    changed_c1: np.ndarray
    base_inputs: torch.Tensor
    source_inputs: torch.Tensor
    base_labels: torch.Tensor
    cf_labels: torch.Tensor
    stats: dict[str, float]

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "size": int(self.base_labels.shape[0]),
            "stats": self.stats,
            "target_variable": TARGET_VAR,
        }


class PairDataset(Dataset):
    def __init__(self, base_inputs: torch.Tensor, source_inputs: torch.Tensor, labels: torch.Tensor):
        self.base_inputs = base_inputs
        self.source_inputs = source_inputs
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.base_inputs[index],
            "source_input_ids": self.source_inputs[index],
            "labels": self.labels[index],
        }


def default_config() -> BinaryAdditionConfig:
    return BinaryAdditionConfig()


def fixed_model_config(config: BinaryAdditionConfig | None = None) -> VariableWidthMLPConfig:
    cfg = default_config() if config is None else config
    return VariableWidthMLPConfig(
        input_dim=INPUT_DIM,
        hidden_dims=list(cfg.hidden_dims),
        num_classes=NUM_CLASSES,
        dropout=0.0,
        activation="relu",
    )


def count_model_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def enumerate_base_rows() -> np.ndarray:
    rows = []
    for value in range(64):
        bits = [(value >> shift) & 1 for shift in range(5, -1, -1)]
        rows.append(bits)
    return np.asarray(rows, dtype=np.int64)


ALL_BASE_ROWS = enumerate_base_rows()


def compute_states(bits: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(bits, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    a2, a1, a0, b2, b1, b0 = [arr[:, i] for i in range(6)]
    s0 = (a0 + b0) % 2
    c1 = (a0 + b0) // 2
    s1 = (a1 + b1 + c1) % 2
    c2 = (a1 + b1 + c1) // 2
    s2 = (a2 + b2 + c2) % 2
    c3 = (a2 + b2 + c2) // 2
    o = 8 * c3 + 4 * s2 + 2 * s1 + s0
    return {
        "A2": a2,
        "A1": a1,
        "A0": a0,
        "B2": b2,
        "B1": b1,
        "B0": b0,
        "S0": s0,
        TARGET_VAR: c1,
        "S1": s1,
        "C2": c2,
        "S2": s2,
        "C3": c3,
        "O": o,
    }


def bits_to_inputs_embeds(bits: np.ndarray) -> torch.Tensor:
    arr = np.asarray(bits, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    onehots = np.eye(2, dtype=np.float32)[arr]
    return torch.tensor(onehots.reshape(arr.shape[0], -1), dtype=torch.float32)


def factual_tensors(base_rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    states = compute_states(base_rows)
    return bits_to_inputs_embeds(base_rows), torch.tensor(states["O"], dtype=torch.long)


def evaluate_factual(
    model: VariableWidthMLPForClassification,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size].to(device)
            logits = model(inputs_embeds=batch.unsqueeze(1))[0]
            predictions.append(torch.argmax(logits, dim=1).cpu())
    predicted = torch.cat(predictions, dim=0)
    return {
        "exact_acc": float((predicted == labels).float().mean().item()),
        "num_examples": int(labels.numel()),
    }


def train_factual_model(
    config: BinaryAdditionConfig,
    device: torch.device,
) -> tuple[VariableWidthMLPForClassification, dict[str, object]]:
    set_seed(config.seed)
    model_config = fixed_model_config(config)
    model = VariableWidthMLPForClassification(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x_train, y_train = factual_tensors(ALL_BASE_ROWS)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config.train_batch_size, shuffle=True)

    best_state = None
    best_metrics = {"exact_acc": -1.0}
    perfect_streak = 0
    for _epoch in range(config.max_epochs):
        model.train()
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs_embeds=batch_inputs.unsqueeze(1))[0]
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()

        metrics = evaluate_factual(model, x_train, y_train, config.eval_batch_size, device)
        if metrics["exact_acc"] >= best_metrics["exact_acc"]:
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        perfect_streak = perfect_streak + 1 if metrics["exact_acc"] >= 1.0 else 0
        if perfect_streak >= config.perfect_streak:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    exact_metrics = evaluate_factual(model, x_train, y_train, config.eval_batch_size, device)
    payload = {
        "seed": config.seed,
        "model_config": model_config.to_dict(),
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "factual_exact_acc": float(exact_metrics["exact_acc"]),
        "num_parameters": count_model_parameters(model),
        "benchmark": {
            "input_dim": INPUT_DIM,
            "num_classes": NUM_CLASSES,
            "target_variable": TARGET_VAR,
            "base_rows": int(ALL_BASE_ROWS.shape[0]),
        },
        "training_config": asdict(config),
    }
    if payload["factual_exact_acc"] < 1.0:
        raise RuntimeError(
            f"Fixed binary-addition model failed to reach exact accuracy: {payload['factual_exact_acc']:.4f}"
        )
    return model, payload


def save_factual_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_factual_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[VariableWidthMLPForClassification, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    model_config = VariableWidthMLPConfig(**checkpoint["model_config"])
    model = VariableWidthMLPForClassification(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def ensure_factual_model(
    device: torch.device,
    checkpoint_path: Path = FACTUAL_CHECKPOINT,
    force_retrain: bool = False,
    config: BinaryAdditionConfig | None = None,
) -> tuple[VariableWidthMLPForClassification, dict[str, object], bool]:
    cfg = default_config() if config is None else config
    if checkpoint_path.exists() and not force_retrain:
        model, payload = load_factual_checkpoint(checkpoint_path, device)
        trained_now = False
    else:
        model, payload = train_factual_model(cfg, device)
        save_factual_checkpoint(checkpoint_path, payload)
        trained_now = True
    x_eval, y_eval = factual_tensors(ALL_BASE_ROWS)
    exact_metrics = evaluate_factual(model, x_eval, y_eval, cfg.eval_batch_size, device)
    if exact_metrics["exact_acc"] < 1.0:
        raise RuntimeError(f"Loaded binary-addition checkpoint is not exact: {exact_metrics['exact_acc']:.4f}")
    payload = {
        **payload,
        "factual_exact_acc": float(exact_metrics["exact_acc"]),
        "checkpoint_path": str(checkpoint_path),
    }
    return model, payload, trained_now


def split_base_rows(seed: int, config: BinaryAdditionConfig | None = None) -> dict[str, np.ndarray]:
    cfg = default_config() if config is None else config
    rows = np.array(ALL_BASE_ROWS, copy=True)
    rng = np.random.default_rng(seed)
    rows = rows[rng.permutation(rows.shape[0])]
    n_train = cfg.train_bases
    n_val = cfg.validation_bases
    n_test = cfg.test_bases
    if n_train + n_val + n_test != rows.shape[0]:
        raise ValueError("Pair-bank base split must cover exactly the 64 base rows.")
    return {
        "train": rows[:n_train],
        "validation": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }


def sample_random_other(base: np.ndarray, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    candidates = rows[~np.all(rows == base, axis=1)]
    return np.array(candidates[int(rng.integers(0, candidates.shape[0]))], copy=True)


def find_c1_flip(base: np.ndarray, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base_c1 = int(compute_states(base)[TARGET_VAR][0])
    candidates = []
    for row in rows:
        if np.all(row == base):
            continue
        row_c1 = int(compute_states(row)[TARGET_VAR][0])
        if row_c1 != base_c1:
            hamming = int(np.sum(row != base))
            candidates.append((hamming, row.copy()))
    candidates.sort(key=lambda item: item[0])
    best_hamming = candidates[0][0]
    best_rows = [row for hamming, row in candidates if hamming == best_hamming]
    return np.array(best_rows[int(rng.integers(0, len(best_rows)))], copy=True)


def generate_family(base: np.ndarray, all_rows: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    family = [sample_random_other(base, all_rows, rng)]
    for bit_index in range(base.shape[0]):
        source = np.array(base, copy=True)
        source[bit_index] = 1 - source[bit_index]
        family.append(source)
    family.append(find_c1_flip(base, all_rows, rng))
    deduplicated = []
    seen = set()
    base_key = tuple(int(value) for value in base.tolist())
    for row in family:
        key = tuple(int(value) for value in row.tolist())
        if key != base_key and key not in seen:
            deduplicated.append(row)
            seen.add(key)
    return deduplicated


def compute_counterfactual_c1_labels(base_bits: np.ndarray, source_bits: np.ndarray) -> np.ndarray:
    base = compute_states(base_bits)
    source = compute_states(source_bits)
    s0 = base["S0"]
    c1 = source[TARGET_VAR]
    s1 = (base["A1"] + base["B1"] + c1) % 2
    c2 = (base["A1"] + base["B1"] + c1) // 2
    s2 = (base["A2"] + base["B2"] + c2) % 2
    c3 = (base["A2"] + base["B2"] + c2) // 2
    return 8 * c3 + 4 * s2 + 2 * s1 + s0


def build_pair_bank(
    name: str,
    base_rows: np.ndarray,
    all_rows: np.ndarray,
    seed: int,
    positive_fraction: float,
) -> PairBank:
    rng = np.random.default_rng(seed)
    candidate_base = []
    candidate_source = []
    candidate_changed = []
    for base in base_rows:
        base_c1 = int(compute_states(base)[TARGET_VAR][0])
        for source in generate_family(base, all_rows, rng):
            source_c1 = int(compute_states(source)[TARGET_VAR][0])
            candidate_base.append(np.array(base, copy=True))
            candidate_source.append(np.array(source, copy=True))
            candidate_changed.append(source_c1 != base_c1)

    base_arr = np.asarray(candidate_base, dtype=np.int64)
    source_arr = np.asarray(candidate_source, dtype=np.int64)
    changed = np.asarray(candidate_changed, dtype=bool)
    base_labels = compute_states(base_arr)["O"]
    cf_labels = compute_counterfactual_c1_labels(base_arr, source_arr)

    if not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("positive_fraction must be in [0, 1].")
    if positive_fraction in {0.0, 1.0}:
        selected = np.where(changed if positive_fraction == 1.0 else ~changed)[0]
    else:
        positives = np.where(changed)[0]
        negatives = np.where(~changed)[0]
        target_pos = min(len(positives), int(round(len(base_arr) * positive_fraction)))
        target_neg = min(len(negatives), target_pos)
        selected = np.concatenate(
            [
                rng.permutation(positives)[:target_pos],
                rng.permutation(negatives)[:target_neg],
            ]
        )
        selected = rng.permutation(selected)

    base_arr = base_arr[selected]
    source_arr = source_arr[selected]
    changed = changed[selected]
    base_labels = base_labels[selected]
    cf_labels = cf_labels[selected]
    return PairBank(
        name=name,
        seed=seed,
        base_rows=base_arr,
        source_rows=source_arr,
        changed_c1=changed,
        base_inputs=bits_to_inputs_embeds(base_arr),
        source_inputs=bits_to_inputs_embeds(source_arr),
        base_labels=torch.tensor(base_labels, dtype=torch.long),
        cf_labels=torch.tensor(cf_labels, dtype=torch.long),
        stats={
            "size": int(base_arr.shape[0]),
            "changed_c1_rate": float(changed.mean()) if changed.size else 0.0,
        },
    )


def build_default_pair_banks(config: BinaryAdditionConfig | None = None) -> dict[str, PairBank]:
    cfg = default_config() if config is None else config
    base_splits = split_base_rows(cfg.seed, cfg)
    return {
        "fit": build_pair_bank("fit_c1_mib", base_splits["train"], ALL_BASE_ROWS, cfg.seed + 201, positive_fraction=0.5),
        "calibration_positive": build_pair_bank(
            "calibration_c1_positive",
            base_splits["validation"],
            ALL_BASE_ROWS,
            cfg.seed + 301,
            positive_fraction=1.0,
        ),
        "calibration_invariant": build_pair_bank(
            "calibration_c1_invariant",
            base_splits["validation"],
            ALL_BASE_ROWS,
            cfg.seed + 302,
            positive_fraction=0.0,
        ),
        "test_positive": build_pair_bank(
            "test_c1_positive",
            base_splits["test"],
            ALL_BASE_ROWS,
            cfg.seed + 401,
            positive_fraction=1.0,
        ),
        "test_invariant": build_pair_bank(
            "test_c1_invariant",
            base_splits["test"],
            ALL_BASE_ROWS,
            cfg.seed + 402,
            positive_fraction=0.0,
        ),
    }


def labels_to_bits(labels: torch.Tensor | Sequence[int], num_bits: int = 4) -> torch.Tensor:
    flat = torch.as_tensor(labels, dtype=torch.long).view(-1)
    bits = []
    for shift in range(num_bits - 1, -1, -1):
        bits.append((flat >> shift) & 1)
    return torch.stack(bits, dim=1)


def metrics_from_binary_logits(logits: torch.Tensor, labels: torch.Tensor | Sequence[int]) -> dict[str, float]:
    predictions = torch.argmax(logits, dim=1)
    targets = torch.as_tensor(labels, dtype=torch.long).view(-1)
    pred_bits = labels_to_bits(predictions)
    target_bits = labels_to_bits(targets)
    return {
        "exact_acc": float((predictions == targets).float().mean().item()),
        "mean_shared_bits": float((pred_bits == target_bits).float().sum(dim=1).mean().item()),
    }


def collect_hidden_by_layer(
    model: VariableWidthMLPForClassification,
    inputs: torch.Tensor,
    device: torch.device,
) -> list[np.ndarray]:
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for layer in range(model.config.n_layer):
        def _make_hook(name: str):
            def _hook(_module, _inp, out):
                activations[name] = out.detach().cpu()

            return _hook

        hooks.append(model.h[layer].register_forward_hook(_make_hook(f"L{layer}")))
    with torch.no_grad():
        _ = model(inputs_embeds=inputs.to(device).unsqueeze(1))
    for hook in hooks:
        hook.remove()
    return [activations[f"L{layer}"].reshape(inputs.shape[0], -1).numpy() for layer in range(model.config.n_layer)]


def fit_linear_probe(
    train_hidden: np.ndarray,
    train_labels: np.ndarray,
    test_hidden: np.ndarray,
    test_labels: np.ndarray,
    epochs: int = 500,
    lr: float = 0.1,
) -> dict[str, float]:
    x_train = torch.tensor(train_hidden, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_test = torch.tensor(test_hidden, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    probe = torch.nn.Linear(x_train.shape[1], 2)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        logits = probe(x_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_preds = torch.argmax(probe(x_train), dim=1)
        test_preds = torch.argmax(probe(x_test), dim=1)
    return {
        "train_acc": float((train_preds == y_train).float().mean().item()),
        "test_acc": float((test_preds == y_test).float().mean().item()),
    }


def compute_layer_probes(
    model: VariableWidthMLPForClassification,
    config: BinaryAdditionConfig | None,
    device: torch.device,
) -> list[dict[str, float]]:
    cfg = default_config() if config is None else config
    base_splits = split_base_rows(cfg.seed, cfg)
    x_train, _ = factual_tensors(base_splits["train"])
    x_test, _ = factual_tensors(base_splits["test"])
    hidden_train = collect_hidden_by_layer(model, x_train, device)
    hidden_test = collect_hidden_by_layer(model, x_test, device)
    y_train = compute_states(base_splits["train"])[TARGET_VAR]
    y_test = compute_states(base_splits["test"])[TARGET_VAR]
    results = []
    for layer, (train_hidden, test_hidden) in enumerate(zip(hidden_train, hidden_test)):
        probe = fit_linear_probe(train_hidden, y_train, test_hidden, y_test)
        results.append({"layer": layer, **probe})
    return results


def iter_das_specs(model: VariableWidthMLPForClassification) -> list[DASSearchSpec]:
    specs = []
    for layer in range(model.config.n_layer):
        width = int(model.config.hidden_dims[layer])
        for subspace_dim in range(1, width + 1):
            specs.append(DASSearchSpec(layer=layer, subspace_dim=subspace_dim, component=f"h[{layer}].output"))
    return specs


def as_float_dict(layer_mass: dict[str, float], num_layers: int) -> dict[str, float]:
    return {f"L{layer}": float(layer_mass.get(f"L{layer}", 0.0)) for layer in range(num_layers)}
