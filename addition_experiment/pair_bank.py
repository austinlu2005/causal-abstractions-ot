"""Shared pair-bank data structures for base/source counterfactual splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_TARGET_VARS
from .scm import (
    AdditionProblem,
    compute_counterfactual_labels,
    compute_states_for_digits,
    digits_to_inputs_embeds,
    verify_counterfactual_labels_with_scm,
)


@dataclass(frozen=True)
class PairBank:
    """Shared base/source pair split with factual and counterfactual labels."""

    split: str
    seed: int
    base_digits: torch.Tensor
    source_digits: torch.Tensor
    base_inputs: torch.Tensor
    source_inputs: torch.Tensor
    base_labels: torch.Tensor
    cf_labels_by_var: dict[str, torch.Tensor]

    @property
    def size(self) -> int:
        """Return the number of base/source pairs in the bank."""
        return int(self.base_inputs.shape[0])

    def metadata(self) -> dict[str, int | str]:
        """Return a compact summary of the pair-bank split."""
        return {
            "split": self.split,
            "seed": self.seed,
            "size": self.size,
        }


class PairBankVariableDataset(Dataset):
    """Dataset view exposing one abstract variable's counterfactual labels."""

    def __init__(self, bank: PairBank, variable_name: str):
        """Wrap one abstract variable view of a shared pair bank for DAS."""
        if variable_name not in bank.cf_labels_by_var:
            raise KeyError(f"Unknown variable {variable_name}")
        self.bank = bank
        self.variable_name = variable_name
        self.variable_index = DEFAULT_TARGET_VARS.index(variable_name)

    def __len__(self) -> int:
        """Return the number of examples exposed by this dataset view."""
        return self.bank.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch one base/source intervention example for the chosen variable."""
        return {
            "input_ids": self.bank.base_inputs[index],
            "source_input_ids": self.bank.source_inputs[index],
            "labels": self.bank.cf_labels_by_var[self.variable_name][index],
            "base_labels": self.bank.base_labels[index],
            "intervention_id": torch.tensor(self.variable_index, dtype=torch.long),
        }


def build_pair_bank(
    problem: AdditionProblem,
    size: int,
    seed: int,
    split: str,
    verify_with_scm: bool = False,
) -> PairBank:
    """Create a deterministic base/source pair bank and counterfactual labels."""
    rng = np.random.default_rng(seed)
    base_digits = rng.integers(0, 10, size=(size, 4), dtype=np.int64)
    source_digits = rng.integers(0, 10, size=(size, 4), dtype=np.int64)

    base_states = compute_states_for_digits(base_digits)
    source_states = compute_states_for_digits(source_digits)
    cf_labels_np = compute_counterfactual_labels(base_states, source_states)

    if verify_with_scm:
        verify_counterfactual_labels_with_scm(problem, base_digits, source_digits, cf_labels_np)

    return PairBank(
        split=split,
        seed=seed,
        base_digits=torch.tensor(base_digits, dtype=torch.long),
        source_digits=torch.tensor(source_digits, dtype=torch.long),
        base_inputs=digits_to_inputs_embeds(base_digits, problem.input_var_order),
        source_inputs=digits_to_inputs_embeds(source_digits, problem.input_var_order),
        base_labels=torch.tensor(base_states["O"], dtype=torch.long),
        cf_labels_by_var={
            var: torch.tensor(cf_labels_np[var], dtype=torch.long) for var in DEFAULT_TARGET_VARS
        },
    )
