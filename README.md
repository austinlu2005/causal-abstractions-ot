# Causal Abstractions for Two-Digit Addition

This repo is now organized around Python experiment scripts rather than notebooks.
The active workflow trains a shared MLP for two-digit addition, builds symbolic
counterfactual pair banks from an SCM, and compares multiple causal abstraction
methods on the same held-out counterfactual test split.

## Main Entry Points

- `addition_train.py`
  - Trains the base MLP and writes a checkpoint plus factual validation metrics.
- `addition_compare.py`
  - Runs the shared end-to-end alignment/evaluation pipeline.
  - Set `METHODS` at the top of the file to any subset of `("gw", "ot", "fgw", "das")`.
  - Loads an existing checkpoint from `models/addition_mlp.pt`.
  - Writes a compact JSON result, a plain-text summary, and comparison plots into a timestamped run folder under `results/`.
- `addition_seed_sweep.py`
  - Trains or reuses one checkpoint per seed, then runs the full comparison pipeline for each seed.
  - Writes per-seed comparison artifacts plus one aggregate JSON/text summary and cross-seed plots for both accuracy and method runtime.

These scripts are meant to be edited directly. The config block near the top of
each file is the intended control surface.

## Current Default Experimental Spec

- Input:
  - Two 2-digit numbers encoded as concatenated one-hot digit vectors.
  - Input dimension is `40`.
- Abstract variables:
  - By default all four internal addition variables are used:
    - `S1`, `C1`, `S2`, `C2`
- Output:
  - `200`-class classification over sums `0..199`.
- Neural model:
  - Four-hidden-layer ReLU MLP with hidden width `192`.
- Factual supervised data:
  - `30,000` training examples
  - `4,000` validation examples
- Counterfactual pair splits:
  - train `1,000`
  - calibration `1,000`
  - test `5,000`
- GW / OT / FGW sites:
  - By default collect one site per neuron across all hidden layers
  - Controlled by a `RESOLUTION` parameter
  - `RESOLUTION = 1` means one neuron per site
  - `RESOLUTION = 2` means adjacent pairs, etc.
  - Fit transport on the train split, then choose both `top_k` and `lambda` separately for each abstract variable on the calibration split
  - After truncation, renormalize each selected row to sum to `1`
- DAS search:
  - Sweeps intervention dimensionalities within each layer
  - The default top-level scripts sweep `k` every 16 neurons plus `1`
  - Trains on the train split and selects the best `(layer, k)` on the calibration split

## Repository Layout

- `addition_experiment/`
  - Shared implementation modules for SCMs, pair banks, metrics, pyvene
    utilities, GW/OT/FGW alignment, DAS, plotting, and backbone training.
- `models/`
  - Shared checkpoint location for `addition_mlp.pt`.
- `results/`
  - Timestamped run folders containing compact JSON outputs, plain-text summaries, and generated plots.
- `paper/`
  - Draft paper materials, including `addition_methodology.tex`.
- `deprecated/`
  - Old notebook-based experiments kept only for reference.

## Typical Workflow

1. Edit the config block in `addition_train.py`.
2. Run:

```bash
python addition_train.py
```

3. Edit the config block in `addition_compare.py`.
4. Run:

```bash
python addition_compare.py
```

5. Inspect:
  - JSON results under `results/<timestamp>/`
  - plain-text summaries under `results/<timestamp>/`
  - plot images directly under `results/<timestamp>/`

For multi-seed runs, edit the config block in `addition_seed_sweep.py` and run:

```bash
python addition_seed_sweep.py
```

This keeps all hyperparameters fixed except `seed`, trains a different backbone
for each seed, reuses `models/addition_mlp_seed<seed>.pt` if present unless
`RETRAIN_BACKBONES = True`, and writes aggregate cross-seed plots into the
top-level sweep run directory.

To run only some methods, edit `METHODS` in `addition_compare.py`, for example:

```python
METHODS = ("ot",)
```

## Methods Implemented

- `gw`
  - Entropic Gromov-Wasserstein on relational effect geometry.
- `ot`
  - Entropic optimal transport on direct abstract-to-neural signature costs.
- `fgw`
  - Fused Gromov-Wasserstein using
    `ot.gromov.BAPG_fused_gromov_wasserstein`.
- `das`
  - Rotated-space intervention search with model selection on a dedicated calibration bank.

All methods are evaluated on the same:

- trained MLP
- abstract variable set
- pair-bank split protocol
- exact-match metric
- shared-digit metric

## Metrics

- `exact_acc`
  - Whether the predicted counterfactual output matches the true output exactly.
- `mean_shared_digits`
  - Number of matching digits in the zero-padded output triple `(C2, S2, S1)`,
    averaged over examples.

## Paper Draft

The main methodology writeup is:

- `paper/addition_methodology.tex`

It documents the SCM, neural model, pair-bank protocol, OT/GW/FGW objectives,
DAS procedure, and figure paths used by the current repo.

## Deprecated Material

The older notebook workflow is still kept in:

- `deprecated/`
