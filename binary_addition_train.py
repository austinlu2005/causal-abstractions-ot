"""Train or load the fixed binary-addition factual model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from addition_experiment.runtime import resolve_device, write_json
from binary_addition_common import FACTUAL_CHECKPOINT, ALL_BASE_ROWS, default_config, ensure_factual_model, factual_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument("--checkpoint", type=Path, default=FACTUAL_CHECKPOINT, help="Checkpoint path.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if the checkpoint already exists.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for JSON summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_config()
    device = resolve_device(args.device)
    model, payload, trained_now = ensure_factual_model(
        device=device,
        checkpoint_path=args.checkpoint,
        force_retrain=args.force_retrain,
        config=config,
    )
    _x_all, y_all = factual_tensors(ALL_BASE_ROWS)
    factual_metrics = {
        "exact_acc": float(payload["factual_exact_acc"]),
        "num_examples": int(y_all.numel()),
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / f"{timestamp}_binary_addition_c1_train"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "train_summary.json"
    summary = {
        "checkpoint_path": str(args.checkpoint),
        "trained_now": trained_now,
        "device": str(device),
        "model": {
            "hidden_dims": list(model.config.hidden_dims),
            "num_parameters": int(payload["num_parameters"]),
            "input_dim": int(model.config.input_dim),
            "num_classes": int(model.config.num_classes),
        },
        "benchmark": payload["benchmark"],
        "training_config": payload["training_config"],
        "factual_metrics": factual_metrics,
    }
    write_json(out_path, summary)
    print(json.dumps({"json": str(out_path.resolve()), **summary["model"], **summary["factual_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
