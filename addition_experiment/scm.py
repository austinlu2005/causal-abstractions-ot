"""Symbolic two-digit addition SCM and tensor conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import _env  # noqa: F401

from pyvene import CausalModel

from .constants import CANONICAL_INPUT_VARS, DEFAULT_TARGET_VARS


ONE_HOT_DIGITS = tuple(np.eye(10, dtype=np.float32)[digit] for digit in range(10))


@dataclass(frozen=True)
class AdditionProblem:
    """Bundle of the symbolic addition SCM and its canonical input ordering."""

    causal_model: CausalModel
    input_var_order: tuple[str, ...]


def as_digit(value: Any) -> int:
    """Decode either a scalar digit or one-hot vector into an integer digit."""
    arr = np.asarray(value).reshape(-1)
    if arr.size == 1:
        return int(arr[0])
    return int(arr.argmax())


def build_addition_causal_model() -> CausalModel:
    """Construct the symbolic SCM for two-digit base-10 addition."""
    variables = ["A1", "B1", "A2", "B2", "S1", "C1", "S2", "C2", "O"]
    values = {
        "A1": ONE_HOT_DIGITS,
        "B1": ONE_HOT_DIGITS,
        "A2": ONE_HOT_DIGITS,
        "B2": ONE_HOT_DIGITS,
        "S1": list(range(10)),
        "C1": [0, 1],
        "S2": list(range(10)),
        "C2": [0, 1],
        "O": list(range(200)),
    }
    parents = {
        "A1": [],
        "B1": [],
        "A2": [],
        "B2": [],
        "S1": ["A1", "B1"],
        "C1": ["A1", "B1"],
        "S2": ["A2", "B2", "C1"],
        "C2": ["A2", "B2", "C1"],
        "O": ["C2", "S2", "S1"],
    }

    def filler() -> np.ndarray:
        """Provide a default digit input for root SCM variables."""
        return ONE_HOT_DIGITS[0]

    functions = {
        "A1": filler,
        "B1": filler,
        "A2": filler,
        "B2": filler,
        "S1": lambda a1, b1: (as_digit(a1) + as_digit(b1)) % 10,
        "C1": lambda a1, b1: (as_digit(a1) + as_digit(b1)) // 10,
        "S2": lambda a2, b2, c1: (as_digit(a2) + as_digit(b2) + int(c1)) % 10,
        "C2": lambda a2, b2, c1: (as_digit(a2) + as_digit(b2) + int(c1)) // 10,
        "O": lambda c2, s2, s1: int(100 * int(c2) + 10 * int(s2) + int(s1)),
    }
    return CausalModel(variables, values, parents, functions)


def assignment_from_digits(digits: np.ndarray | list[int] | tuple[int, ...]) -> dict[str, np.ndarray]:
    """Map four raw digits to the SCM's one-hot input assignment."""
    a1, b1, a2, b2 = [int(part) for part in np.asarray(digits).reshape(-1)]
    return {
        "A1": ONE_HOT_DIGITS[a1],
        "B1": ONE_HOT_DIGITS[b1],
        "A2": ONE_HOT_DIGITS[a2],
        "B2": ONE_HOT_DIGITS[b2],
    }


def compute_states_for_digits(digits: np.ndarray | list[list[int]] | list[int]) -> dict[str, np.ndarray]:
    """Compute the addition inputs, internal variables, and final output."""
    arr = np.asarray(digits, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(CANONICAL_INPUT_VARS):
        raise ValueError(f"Expected digit rows shaped [N, 4], got {arr.shape}")

    a1 = arr[:, 0]
    b1 = arr[:, 1]
    a2 = arr[:, 2]
    b2 = arr[:, 3]

    s1 = (a1 + b1) % 10
    c1 = (a1 + b1) // 10
    second_digit_total = a2 + b2
    s2 = (second_digit_total + c1) % 10
    c2 = (second_digit_total + c1) // 10
    o = 100 * c2 + 10 * s2 + s1

    return {
        "A1": a1,
        "B1": b1,
        "A2": a2,
        "B2": b2,
        "S1": s1,
        "C1": c1,
        "S2": s2,
        "C2": c2,
        "O": o,
    }


def compute_counterfactual_labels(
    base_states: dict[str, np.ndarray],
    source_states: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute SCM counterfactual outputs for each abstract variable swap."""
    base_second_digit_total = base_states["A2"] + base_states["B2"]
    base_s1 = base_states["S1"]
    base_s2 = base_states["S2"]
    base_c2 = base_states["C2"]

    swapped_c1_s2 = (base_second_digit_total + source_states["C1"]) % 10
    swapped_c1_c2 = (base_second_digit_total + source_states["C1"]) // 10

    return {
        "S1": 100 * base_c2 + 10 * base_s2 + source_states["S1"],
        "C1": 100 * swapped_c1_c2 + 10 * swapped_c1_s2 + base_s1,
        "S2": 100 * base_c2 + 10 * source_states["S2"] + base_s1,
        "C2": 100 * source_states["C2"] + 10 * base_s2 + base_s1,
    }


def sample_digit_rows(size: int, seed: int) -> np.ndarray:
    """Sample random four-digit rows used for factual or pair-bank data."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=(size, len(CANONICAL_INPUT_VARS)), dtype=np.int64)


def digits_to_inputs_embeds(
    digits: np.ndarray | list[list[int]] | list[int],
    input_var_order: tuple[str, ...],
) -> torch.Tensor:
    """Pack digit rows into the concatenated one-hot neural input format."""
    arr = np.asarray(digits, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(CANONICAL_INPUT_VARS):
        raise ValueError(f"Expected digit rows shaped [N, 4], got {arr.shape}")

    canonical_index = {var: index for index, var in enumerate(CANONICAL_INPUT_VARS)}
    ordered = np.stack([arr[:, canonical_index[var]] for var in input_var_order], axis=1)
    onehots = np.eye(10, dtype=np.float32)[ordered]
    return torch.tensor(onehots.reshape(arr.shape[0], -1), dtype=torch.float32)


def infer_input_var_order(causal_model: CausalModel) -> tuple[str, ...]:
    """Infer how the SCM packs the four one-hot input digit blocks."""
    marker_digits = {"A1": 1, "B1": 2, "A2": 3, "B2": 4}
    marker_to_var = {marker: var for var, marker in marker_digits.items()}
    marker_assignment = {var: ONE_HOT_DIGITS[digit] for var, digit in marker_digits.items()}

    example = causal_model.generate_factual_dataset(1, lambda: marker_assignment)[0]
    packed = example["input_ids"].detach().cpu().numpy().reshape(-1)
    if packed.size % 10 != 0:
        raise ValueError(f"Unexpected packed input width {packed.size}")

    inferred = []
    for index in range(packed.size // 10):
        segment = packed[index * 10 : (index + 1) * 10]
        marker = int(segment.argmax())
        if marker not in marker_to_var:
            raise ValueError(f"Could not decode segment {index} with marker {marker}")
        inferred.append(marker_to_var[marker])

    if tuple(sorted(inferred)) != tuple(sorted(CANONICAL_INPUT_VARS)):
        raise ValueError(f"Inferred input order {inferred} does not match canonical inputs")
    return tuple(inferred)


def verify_scm_truth_table(causal_model: CausalModel) -> None:
    """Check the SCM against the full two-digit addition truth table."""
    for a1 in range(10):
        for b1 in range(10):
            for a2 in range(10):
                for b2 in range(10):
                    digits = np.array([a1, b1, a2, b2], dtype=np.int64)
                    states = compute_states_for_digits(digits)
                    setting = causal_model.run_forward(assignment_from_digits(digits))
                    for var in ["S1", "C1", "S2", "C2", "O"]:
                        if int(setting[var]) != int(states[var][0]):
                            raise AssertionError(f"SCM mismatch for {digits.tolist()} at {var}")


def verify_input_var_order(causal_model: CausalModel, input_var_order: tuple[str, ...]) -> None:
    """Assert that the inferred input order matches the expected packing."""
    inferred = infer_input_var_order(causal_model)
    if tuple(inferred) != tuple(input_var_order):
        raise AssertionError(f"Expected input order {input_var_order}, got {inferred}")


def load_addition_problem(run_checks: bool = True) -> AdditionProblem:
    """Build the addition SCM bundle and optionally run consistency checks."""
    causal_model = build_addition_causal_model()
    input_var_order = infer_input_var_order(causal_model)
    if run_checks:
        verify_scm_truth_table(causal_model)
        verify_input_var_order(causal_model, input_var_order)
    return AdditionProblem(
        causal_model=causal_model,
        input_var_order=input_var_order,
    )


def verify_counterfactual_labels_with_scm(
    problem: AdditionProblem,
    base_digits: np.ndarray,
    source_digits: np.ndarray,
    cf_labels_by_var: dict[str, np.ndarray],
) -> None:
    """Cross-check vectorized counterfactual labels against SCM interchange."""
    size = base_digits.shape[0]
    for index in range(size):
        base_assignment = assignment_from_digits(base_digits[index])
        source_assignment = assignment_from_digits(source_digits[index])
        for var in DEFAULT_TARGET_VARS:
            expected = int(
                problem.causal_model.run_interchange(base_assignment, {var: source_assignment})["O"]
            )
            actual = int(cf_labels_by_var[var][index])
            if expected != actual:
                raise AssertionError(
                    f"Counterfactual mismatch at index={index}, var={var}, expected={expected}, actual={actual}"
                )
