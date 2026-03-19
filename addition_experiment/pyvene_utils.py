"""Small adapters for building pyvene interventions on the custom MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import _env  # noqa: F401

import torch
import pyvene.models.modeling_utils as modeling_utils
from pyvene import IntervenableConfig, IntervenableModel, RepresentationConfig
from pyvene import RotatedSpaceIntervention, VanillaIntervention
from pyvene.models.mlp.modelings_mlp import MLPForClassification as PyveneMLP

from variable_width_mlp import VariableWidthMLPForClassification, logits_from_output

from .constants import DEFAULT_SITE_MAX_UNITS, DEFAULT_SITE_POSITION, DEFAULT_SITE_UNIT


@dataclass(frozen=True)
class CanonicalSite:
    """Canonical hidden-layer intervention site described by layer and coordinates."""

    layer: int
    dims: tuple[int, ...]
    component: str
    unit: str = DEFAULT_SITE_UNIT
    position: int = DEFAULT_SITE_POSITION
    max_units: int = DEFAULT_SITE_MAX_UNITS

    @property
    def subspace_dims(self) -> list[int]:
        """Return the intervention dimensions for this canonical site."""
        return list(self.dims)

    @property
    def label(self) -> str:
        """Return a short human-readable label for this site."""
        if len(self.dims) == 1:
            return f"L{self.layer}-d{self.dims[0]}"
        return f"L{self.layer}-d{self.dims[0]}:{self.dims[-1]}"


@dataclass(frozen=True)
class DASSearchSpec:
    """Candidate layer and subspace size for a DAS search run."""

    layer: int
    subspace_dim: int
    component: str
    unit: str = DEFAULT_SITE_UNIT
    position: int = DEFAULT_SITE_POSITION
    max_units: int = DEFAULT_SITE_MAX_UNITS

    @property
    def subspace_dims(self) -> list[int]:
        """Return the rotated coordinates swapped by this DAS candidate."""
        return list(range(self.subspace_dim))

    @property
    def label(self) -> str:
        """Return a short human-readable label for this DAS candidate."""
        return f"L{self.layer}-k{self.subspace_dim}"


def register_model_with_pyvene(model: VariableWidthMLPForClassification, layer: int) -> None:
    """Register the custom MLP type with pyvene's intervention mappings."""
    for name, mapping in vars(modeling_utils).items():
        if name.startswith("type_to_") and isinstance(mapping, dict) and PyveneMLP in mapping:
            mapping.setdefault(type(model), mapping[PyveneMLP])
    model.config.h_dim = int(model.config.hidden_dims[layer])


def enumerate_canonical_sites(
    model: VariableWidthMLPForClassification,
    resolution: int = 1,
    layers: Sequence[int] | None = None,
) -> list[CanonicalSite]:
    """Split each requested hidden layer into contiguous intervention sites."""
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    layer_ids = (
        list(range(model.config.n_layer))
        if layers is None
        else [int(layer) for layer in layers]
    )
    sites = []
    for layer in layer_ids:
        width = int(model.config.hidden_dims[layer])
        component = f"h[{layer}].output"
        for start in range(0, width, resolution):
            stop = min(start + resolution, width)
            sites.append(
                CanonicalSite(
                    layer=layer,
                    dims=tuple(range(start, stop)),
                    component=component,
                )
            )
    return sites


def build_intervenable(
    model: VariableWidthMLPForClassification,
    layer: int,
    component: str,
    intervention: VanillaIntervention | RotatedSpaceIntervention,
    device: torch.device | str,
    unit: str = DEFAULT_SITE_UNIT,
    max_units: int = DEFAULT_SITE_MAX_UNITS,
    freeze_model: bool = True,
    freeze_intervention: bool = False,
    use_fast: bool = False,
) -> IntervenableModel:
    """Build a pyvene intervenable wrapper around one model representation."""
    register_model_with_pyvene(model, layer)
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer=layer,
                component=component,
                unit=unit,
                max_number_of_units=max_units,
                intervention=intervention,
            )
        ],
    )
    intervenable = IntervenableModel(config, model, use_fast=use_fast)
    intervenable.set_device(device)
    if freeze_model:
        intervenable.disable_model_gradients()
    if freeze_intervention and hasattr(intervenable, "disable_intervention_gradients"):
        intervenable.disable_intervention_gradients()
    return intervenable


def prepare_base_batch(inputs: torch.Tensor) -> torch.Tensor:
    """Normalize base inputs to the rank expected by pyvene."""
    if inputs.ndim == 2:
        return inputs.unsqueeze(1)
    if inputs.ndim == 3:
        return inputs
    raise ValueError(f"Unexpected base input shape: {tuple(inputs.shape)}")


def prepare_source_batch(inputs: torch.Tensor) -> torch.Tensor:
    """Normalize source inputs to the rank expected by pyvene."""
    if inputs.ndim == 2:
        return inputs.unsqueeze(1)
    if inputs.ndim == 3:
        return inputs[:, :1, :]
    raise ValueError(f"Unexpected source input shape: {tuple(inputs.shape)}")


def run_intervenable_logits(
    intervenable: IntervenableModel,
    base_inputs: torch.Tensor,
    source_inputs: torch.Tensor,
    subspace_dims: Sequence[int],
    position: int,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Run a batched intervention and collect the resulting logits."""
    outputs = []
    device = torch.device(device)
    with torch.no_grad():
        for start in range(0, base_inputs.shape[0], batch_size):
            end = min(start + batch_size, base_inputs.shape[0])
            batch_base = prepare_base_batch(base_inputs[start:end].to(device))
            batch_source = prepare_source_batch(source_inputs[start:end].to(device))
            current_batch = batch_base.shape[0]

            positions = [[position]] * current_batch
            subspaces = [list(subspace_dims)] * current_batch
            _, cf_output = intervenable(
                {"inputs_embeds": batch_base},
                [{"inputs_embeds": batch_source}],
                {"sources->base": ([positions], [positions])},
                subspaces=[subspaces],
            )
            outputs.append(logits_from_output(cf_output).detach().cpu())
    return torch.cat(outputs, dim=0)
