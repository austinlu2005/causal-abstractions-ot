# Causal Abstractions for Addition and Hierarchical Equality

This repo is organized around Python experiment scripts rather than notebooks.
It currently supports three task families:

- two-digit addition
- three-bit binary addition with a fixed `C1` carry benchmark
- hierarchical equality (the `WX`, `YZ`, `O` task)

Both pipelines follow the same high-level pattern:

- train or load a backbone MLP
- build symbolic counterfactual pair banks from an SCM
- compare transport-based alignment methods and DAS on held-out counterfactual data

## Main Entry Points

### Binary Addition (`C1`)

- `binary_addition_train.py`
  - Trains or loads the fixed factual MLP used in the binary carry benchmark.
- `binary_addition_das.py`
  - Runs DAS sweeps over layer, subspace size, learning rate, and calibration strategy.
- `binary_addition_ot_uot.py`
  - Runs OT and UOT sweeps over entropic regularization, support size, intervention strength, site resolution, and calibration strategy.
- `binary_addition_plots.py`
  - Produces the binary-addition accuracy, runtime, and cross-layer plots from the DAS and OT/UOT result JSON files.

### Addition

- `addition_train.py`
  - Trains the addition MLP and writes a checkpoint plus factual validation metrics.
- `addition_run.py`
  - Loads one checkpoint per seed and runs the comparison pipeline.
  - Supports sweeps over OT `epsilon` and Gibbs-kernel `tau`.
  - Writes per-sweep and aggregate summaries under `results/<timestamp>_addition/`.
- `addition_run_gradient.py`
  - Runs the gradient-based transport-policy variant for the addition task.

### Hierarchical Equality

- `equality_run.py`
  - Trains or loads the equality MLP, builds one fixed set of train/calibration/test pair banks, and runs the comparison pipeline.
  - Supports OT-only sweeps over `epsilon` and Gibbs-kernel `tau` while reusing the same pair-bank splits across sweep points.
  - Writes outputs under `results/<timestamp>_equality/`.

These scripts are meant to be edited directly. The config block near the top of
each file is the intended control surface.

## Tasks

### Addition Task

- Input:
  - Two 2-digit numbers encoded as concatenated one-hot digit vectors.
  - Input dimension is `40`.
- Abstract variables:
  - `S1`, `C1`, `S2`, `C2`
- Output:
  - `200`-class classification over sums `0..199`
- Default backbone:
  - ReLU MLP with hidden width `192`

### Binary Addition Task

- Input:
  - Two 3-bit numbers encoded as concatenated one-hot bit vectors.
  - Input dimension is `12`.
- Abstract variable:
  - `C1`, the carry from the least significant bit.
- Output:
  - `16`-class classification over 4-bit sums `0..15`.
- Default backbone:
  - ReLU MLP with hidden widths `(13, 13)`

### Hierarchical Equality Task

- Input:
  - Four entity slots `W, X, Y, Z`, each represented by an `EMBEDDING_DIM` vector
- Abstract variables:
  - `WX = [W == X]`
  - `YZ = [Y == Z]`
- Output:
  - binary label `O = int(WX == YZ)`
- Default backbone:
  - ReLU MLP with hidden widths `(16, 16, 16)`

The equality task is implemented in `equality_experiment/` and is intended to
mirror the notebook’s causal structure while using the same repo-style pipeline
for single-variable DAS and OT/GW/FGW evaluation.

## Methods Implemented

- `gw`
  - Entropic Gromov-Wasserstein on relational effect geometry
- `ot`
  - Entropic optimal transport on direct abstract-to-neural signature costs
- `fgw`
  - Fused Gromov-Wasserstein
- `das`
  - Rotated-space intervention search with calibration-based model selection

For OT-family methods, both experiment runners now expose:

- `OT_EPSILONS`
  - entropic regularization sweep
- `OT_TAUS`
  - Gibbs-kernel temperature sweep

The current implementation uses `tau` directly as the entropic denominator in
the transport solve, so:

- smaller `tau` sharpens the kernel
- larger `tau` smooths the kernel

## Outputs and Metrics

All runs write JSON outputs, text summaries, and plots under `results/`.

Top-level text summaries include:

- the relevant experiment hyperparameters
- per-split pair-bank change statistics such as:
  - total pairs
  - `changed_any`
  - per-variable changed counts and rates

### Addition Metrics

- `exact_acc`
  - predicted counterfactual sum matches exactly
- `mean_shared_digits`
  - average number of matching digits in `(C2, S2, S1)`

### Equality Metrics

- `exact_acc`
  - predicted counterfactual binary label matches exactly

The equality pipeline does not report a separate `shared` metric.

## Repository Layout

- `addition_experiment/`
  - task-specific implementation for addition SCMs, pair banks, backbone training, OT/GW/FGW, DAS, reporting, and plotting
- `equality_experiment/`
  - task-specific implementation for hierarchical equality
- `models/`
  - checkpoint storage such as `addition_mlp_seed<seed>.pt` and `equality_mlp_seed<seed>.pt`
- `results/`
  - timestamped run folders like `results/<timestamp>_addition/` and `results/<timestamp>_equality/`
- `paper/`
  - draft paper materials

## Typical Workflow

### Addition

1. Edit the config block in `addition_train.py`.
2. Run:

```bash
python addition_train.py
```

3. Edit the config block in `addition_run.py`.
4. Run:

```bash
python addition_run.py
```

For the gradient-based transport-policy workflow:

```bash
python addition_run_gradient.py
```

### Binary Addition

1. Train or load the fixed factual model:

```bash
python binary_addition_train.py
```

2. Run the DAS sweep:

```bash
python binary_addition_das.py
```

3. Run the OT/UOT sweep:

```bash
python binary_addition_ot_uot.py
```

4. Generate the paper plots:

```bash
python binary_addition_plots.py --das-results <path-to-das_results.json> --ot-results <path-to-ot_uot_results.json>
```

### Hierarchical Equality

1. Edit the config block in `equality_run.py`.
2. Run:

```bash
python equality_run.py
```

Depending on `RETRAIN_BACKBONE`, this will either train a fresh equality
backbone or load `models/equality_mlp_seed<seed>.pt`.

## Notes on the Equality Pipeline

- Equality OT epsilon/tau sweeps reuse a single prebuilt train/calibration/test
  pair-bank split so different sweep points are directly comparable.
- If `METHODS` includes both OT-family methods and `das`, the non-OT methods run
  once outside the epsilon/tau sweep.
- The equality task currently uses its own task-specific OT, DAS, reporting, and
  pair-bank code rather than sharing those files with the addition task.
