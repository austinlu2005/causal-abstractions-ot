from __future__ import annotations

from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import os

from huggingface_hub import login as hf_login
import torch

from ravel_experiment.compare_runner import CompareExperimentConfig, run_comparison
from ravel_experiment.data import SOURCE_TYPES, TARGET_ATTRIBUTES, build_pair_banks, load_filtered_ravel_pipeline
from ravel_experiment.ot import (
    OTConfig,
    load_prepared_alignment_artifacts,
    prepare_alignment_artifacts,
    save_prepared_alignment_artifacts,
)
from ravel_experiment.runtime import resolve_device, write_json
from ravel_experiment.sites import enumerate_residual_sites


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return int(lowered)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_str_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_int_list(name: str, default: list[int]) -> list[int]:
    return [int(item) for item in _env_str_list(name, [str(value) for value in default])]


def _env_float_list(name: str, default: list[float]) -> list[float]:
    return [float(item) for item in _env_str_list(name, [str(value) for value in default])]


def _env_layers(name: str, default: list[int] | str) -> list[int] | str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    return [int(item.strip()) for item in value.split(",") if item.strip()]


DEVICE = os.environ.get("RAVEL_DEVICE", "cuda")
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_ravel"
OUTPUT_PATH = RUN_DIR / "ravel_run_results.json"
SIGNATURES_DIR = Path("signatures")
SPLIT_PRINT_ORDER = ("train", "calibration", "test")

MODEL_NAME = os.environ.get("RAVEL_MODEL_NAME", "google/gemma-2-2b")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
PROMPT_HF_LOGIN = os.environ.get("RAVEL_PROMPT_HF_LOGIN", "1").strip().lower() not in {"0", "false", "no"}

# Data
RAVEL_DATASET_PATH = os.environ.get("RAVEL_DATASET_PATH", "mib-bench/ravel")
RAVEL_DATASET_CONFIG = os.environ.get("RAVEL_DATASET_CONFIG") or None
DATASET_SIZE = _env_optional_int("RAVEL_DATASET_SIZE", None)
SPLIT_SEED = _env_int("RAVEL_SPLIT_SEED", 0)
TRAIN_POOL_SIZE = _env_int("RAVEL_TRAIN_POOL_SIZE", 180)
CALIBRATION_POOL_SIZE = _env_int("RAVEL_CALIBRATION_POOL_SIZE", 90)
TEST_POOL_SIZE = _env_int("RAVEL_TEST_POOL_SIZE", 90)

# Experiment
METHODS = _env_str_list("RAVEL_METHODS", ["ot"])
TARGET_VARS = _env_str_list("RAVEL_TARGET_VARS", list(TARGET_ATTRIBUTES))
RAVEL_SOURCE_TYPES = _env_str_list("RAVEL_SOURCE_TYPES", list(SOURCE_TYPES))

LAYERS = _env_layers("RAVEL_LAYERS", [25])
TOKEN_POSITION_IDS = _env_str_list("RAVEL_TOKEN_POSITION_IDS", ["entity_last_token", "last_token"])

BATCH_SIZE = _env_int("RAVEL_BATCH_SIZE", 32)

RESOLUTIONS = _env_int_list("RAVEL_RESOLUTIONS", [1])
OT_EPSILONS = _env_float_list("RAVEL_OT_EPSILONS", [20, 30, 40, 50])
UOT_BETA_ABSTRACTS = _env_float_list("RAVEL_UOT_BETA_ABSTRACTS", [0.1, 1.0])
UOT_BETA_NEURALS = _env_float_list("RAVEL_UOT_BETA_NEURALS", [0.1, 1.0])
SIGNATURE_MODES = _env_str_list("RAVEL_SIGNATURE_MODES", ["answer_token_delta"])
OT_TOP_K_VALUES = _env_int_list("RAVEL_OT_TOP_K_VALUES", list(range(1, 11)))
OT_LAMBDAS = _env_float_list("RAVEL_OT_LAMBDAS", [i for i in range(1, 31)])

DAS_MAX_EPOCHS = _env_int("RAVEL_DAS_MAX_EPOCHS", 200)
DAS_MIN_EPOCHS = _env_int("RAVEL_DAS_MIN_EPOCHS", 5)
DAS_PLATEAU_PATIENCE = _env_int("RAVEL_DAS_PLATEAU_PATIENCE", 1)
DAS_PLATEAU_REL_DELTA = float(os.environ.get("RAVEL_DAS_PLATEAU_REL_DELTA", "1e-3"))
DAS_LEARNING_RATE = float(os.environ.get("RAVEL_DAS_LEARNING_RATE", "1e-3"))
DAS_SUBSPACE_DIMS = _env_int_list("RAVEL_DAS_SUBSPACE_DIMS", [576, 1152, 1728, 2304])


def _signature_cache_spec(
    *,
    train_bank,
    resolution: int,
    signature_mode: str,
    selected_layers: list[int],
    token_position_ids: tuple[str, ...],
) -> dict[str, object]:
    train_rows_digest = hashlib.sha256(
        "\n".join(
            f"{base.get('prompt', '')}|||{source.get('prompt', '')}"
            for base, source in zip(train_bank.base_inputs, train_bank.source_inputs)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "ravel_alignment_signatures",
        "model_name": MODEL_NAME,
        "dataset_path": RAVEL_DATASET_PATH,
        "dataset_config": RAVEL_DATASET_CONFIG,
        "source_type": train_bank.source_type,
        "target_var": train_bank.target_var,
        "split_seed": int(SPLIT_SEED),
        "resolution": int(resolution),
        "signature_mode": str(signature_mode),
        "train_pool_size": int(TRAIN_POOL_SIZE),
        "train_bank": train_bank.metadata(),
        "train_rows_digest": train_rows_digest,
        "selected_layers": [int(layer) for layer in selected_layers],
        "token_position_ids": list(token_position_ids if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS)),
        "batch_size": int(BATCH_SIZE),
    }


def _signature_cache_path(
    *,
    source_type: str,
    target_var: str,
    resolution: int,
    signature_mode: str,
    cache_spec: dict[str, object],
) -> Path:
    spec_json = json.dumps(cache_spec, sort_keys=True, separators=(",", ":"))
    spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()[:12]
    source_slug = source_type.replace("_", "-")
    target_slug = target_var.lower()
    stem = (
        f"ravel_{source_slug}_{target_slug}_res-{int(resolution)}_sig-{str(signature_mode)}"
        f"_train-{int(cache_spec['train_pool_size'])}_{spec_hash}.pt"
    )
    return SIGNATURES_DIR / stem


def ensure_hf_login(token: str | None, prompt_login: bool) -> str | None:
    if token:
        hf_login(token=token, add_to_git_credential=False)
        return token
    if prompt_login:
        hf_login(add_to_git_credential=False)
    return token


def _is_memory_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or any(
        needle in message
        for needle in (
            "out of memory",
            "cuda out of memory",
            "mps backend out of memory",
            "cublas_status_alloc_failed",
            "cuda error: out of memory",
            "hip out of memory",
        )
    )


def _clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _print_source_type_summary(data_metadata: dict[str, object], source_type: str) -> None:
    print(f"[run] source_type={source_type}")
    for split in SPLIT_PRINT_ORDER:
        split_metadata = data_metadata.get(split, {})
        if isinstance(split_metadata, dict) and split_metadata:
            for variable, variable_stats in sorted(split_metadata.items()):
                total_pairs = int(variable_stats.get("size", 0))
                changed_count = int(variable_stats.get("changed_count", 0))
                unchanged_count = max(0, total_pairs - changed_count)
                print(
                    f"{split} pair bank [{variable}] | changed={changed_count} "
                    f"| unchanged={unchanged_count} "
                    f"| changed_rate={float(variable_stats.get('changed_rate', 0.0)):.4f} "
                    f"| query_distribution={variable_stats.get('query_distribution', {})}"
                )


def main() -> None:
    device = resolve_device(DEVICE)
    hf_token = ensure_hf_login(HF_TOKEN, PROMPT_HF_LOGIN)
    print(f"[run] starting RAVEL run model={MODEL_NAME} device={device}")
    model, tokenizer, token_positions, catalog, filtered_datasets = load_filtered_ravel_pipeline(
        model_name=MODEL_NAME,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        dataset_size=DATASET_SIZE,
        hf_token=hf_token,
        dataset_path=RAVEL_DATASET_PATH,
        dataset_name=RAVEL_DATASET_CONFIG,
    )
    print("[run] building pair banks")
    banks_by_split, data_metadata = build_pair_banks(
        tokenizer=tokenizer,
        token_positions=token_positions,
        rows_by_split=filtered_datasets,
        catalog=catalog,
        target_vars=tuple(TARGET_VARS),
        source_types=tuple(RAVEL_SOURCE_TYPES),
        split_seed=SPLIT_SEED,
        train_pool_size=TRAIN_POOL_SIZE,
        calibration_pool_size=CALIBRATION_POOL_SIZE,
        test_pool_size=TEST_POOL_SIZE,
        dataset_name=RAVEL_DATASET_PATH,
    )
    print(f"[run] built splits={list(banks_by_split.keys())}")
    selected_layers = list(range(int(model.config.num_hidden_layers))) if LAYERS == "auto" else list(LAYERS)
    print(f"[run] selected_layers={selected_layers}")
    token_position_ids = tuple(token_position.id for token_position in token_positions)

    all_payloads = []
    for source_type in RAVEL_SOURCE_TYPES:
        source_banks_by_split = {
            split: split_banks[source_type]
            for split, split_banks in banks_by_split.items()
        }
        source_metadata = {
            split: split_metadata[source_type]
            for split, split_metadata in data_metadata.items()
        }
        _print_source_type_summary(source_metadata, source_type)
        for method in METHODS:
            for signature_mode in SIGNATURE_MODES:
                for resolution in RESOLUTIONS:
                    prepared_ot_artifacts_by_target: dict[str, dict[str, object]] | None = None
                    try:
                        if method in {"ot", "uot"} and TARGET_VARS:
                            ot_sites = enumerate_residual_sites(
                                num_layers=int(model.config.num_hidden_layers),
                                hidden_size=int(model.config.hidden_size),
                                token_position_ids=token_position_ids,
                                resolution=int(resolution),
                                layers=tuple(selected_layers),
                                selected_token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                            )
                            prepared_ot_artifacts_by_target = {}
                            for target_var in TARGET_VARS:
                                train_bank = source_banks_by_split["train"][target_var]
                                cache_spec = _signature_cache_spec(
                                    train_bank=train_bank,
                                    resolution=int(resolution),
                                    signature_mode=signature_mode,
                                    selected_layers=selected_layers,
                                    token_position_ids=token_position_ids,
                                )
                                cache_path = _signature_cache_path(
                                    source_type=source_type,
                                    target_var=target_var,
                                    resolution=int(resolution),
                                    signature_mode=signature_mode,
                                    cache_spec=cache_spec,
                                )
                                prepared_artifacts = load_prepared_alignment_artifacts(
                                    cache_path,
                                    expected_spec=cache_spec,
                                )
                                if prepared_artifacts is not None:
                                    print(
                                        f"[signatures] loaded cache path={cache_path} "
                                        f"source_type={source_type} target={target_var} "
                                        f"prepare_time={float(prepared_artifacts.get('prepare_runtime_seconds', 0.0)):.2f}s"
                                    )
                                else:
                                    prepared_artifacts = prepare_alignment_artifacts(
                                        model=model,
                                        fit_bank=train_bank,
                                        sites=ot_sites,
                                        device=device,
                                        config=OTConfig(
                                            method=method,
                                            batch_size=BATCH_SIZE,
                                            epsilon=1.0,
                                            signature_mode=signature_mode,
                                            top_k_values=tuple(OT_TOP_K_VALUES),
                                            lambda_values=tuple(OT_LAMBDAS),
                                        ),
                                    )
                                    prepared_artifacts["cache_spec"] = cache_spec
                                    prepared_artifacts["cache_path"] = str(cache_path)
                                    save_prepared_alignment_artifacts(
                                        cache_path,
                                        prepared_artifacts=prepared_artifacts,
                                        cache_spec=cache_spec,
                                    )
                                    print(
                                        f"[signatures] saved cache path={cache_path} "
                                        f"source_type={source_type} target={target_var} "
                                        f"prepare_time={float(prepared_artifacts.get('prepare_runtime_seconds', 0.0)):.2f}s"
                                    )
                                prepared_ot_artifacts_by_target[target_var] = prepared_artifacts
                        epsilon_values = OT_EPSILONS if method in {"ot", "uot"} else [None]
                        for epsilon in epsilon_values:
                            if method == "uot":
                                for beta_abstract in UOT_BETA_ABSTRACTS:
                                    for beta_neural in UOT_BETA_NEURALS:
                                        output_stem = (
                                            f"ravel_{source_type}_uot_res-{int(resolution)}_sig-{signature_mode}_"
                                            f"eps-{float(epsilon):g}_ba-{beta_abstract:g}_bn-{beta_neural:g}"
                                        )
                                        uot_config = CompareExperimentConfig(
                                            model_name=MODEL_NAME,
                                            source_type=source_type,
                                            output_path=RUN_DIR / f"{output_stem}.json",
                                            summary_path=RUN_DIR / f"{output_stem}.txt",
                                            methods=("uot",),
                                            target_vars=tuple(TARGET_VARS),
                                            batch_size=BATCH_SIZE,
                                            ot_epsilon=float(epsilon),
                                            uot_beta_abstract=float(beta_abstract),
                                            uot_beta_neural=float(beta_neural),
                                            signature_mode=signature_mode,
                                            ot_top_k_values=tuple(OT_TOP_K_VALUES),
                                            ot_lambdas=tuple(OT_LAMBDAS),
                                            resolution=int(resolution),
                                            layers=tuple(selected_layers),
                                            token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                                        )
                                        all_payloads.append(
                                            run_comparison(
                                                model=model,
                                                tokenizer=tokenizer,
                                                token_positions=token_positions,
                                                banks_by_split=source_banks_by_split,
                                                data_metadata=source_metadata,
                                                device=device,
                                                config=uot_config,
                                                prepared_ot_artifacts_by_target=prepared_ot_artifacts_by_target,
                                            )
                                        )
                            else:
                                output_stem = f"ravel_{source_type}_{method}_res-{int(resolution)}_sig-{signature_mode}"
                                if epsilon is not None:
                                    output_stem = f"{output_stem}_eps-{float(epsilon):g}"
                                config = CompareExperimentConfig(
                                    model_name=MODEL_NAME,
                                    source_type=source_type,
                                    output_path=RUN_DIR / f"{output_stem}.json",
                                    summary_path=RUN_DIR / f"{output_stem}.txt",
                                    methods=(method,),
                                    target_vars=tuple(TARGET_VARS),
                                    batch_size=BATCH_SIZE,
                                    ot_epsilon=float(epsilon) if epsilon is not None else 1.0,
                                    signature_mode=signature_mode,
                                    ot_top_k_values=tuple(OT_TOP_K_VALUES),
                                    ot_lambdas=tuple(OT_LAMBDAS),
                                    das_max_epochs=DAS_MAX_EPOCHS,
                                    das_min_epochs=DAS_MIN_EPOCHS,
                                    das_plateau_patience=DAS_PLATEAU_PATIENCE,
                                    das_plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
                                    das_learning_rate=DAS_LEARNING_RATE,
                                    das_subspace_dims=tuple(DAS_SUBSPACE_DIMS),
                                    resolution=int(resolution),
                                    layers=tuple(selected_layers),
                                    token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                                )
                                all_payloads.append(
                                    run_comparison(
                                        model=model,
                                        tokenizer=tokenizer,
                                        token_positions=token_positions,
                                        banks_by_split=source_banks_by_split,
                                        data_metadata=source_metadata,
                                        device=device,
                                        config=config,
                                        prepared_ot_artifacts_by_target=prepared_ot_artifacts_by_target,
                                    )
                                )
                    except Exception as exc:
                        if not _is_memory_error(exc):
                            raise
                        print(
                            f"[oom] skipping source_type={source_type} method={method} "
                            f"signature_mode={signature_mode} resolution={int(resolution)} "
                            f"after memory failure: {exc}"
                        )
                        prepared_ot_artifacts_by_target = None
                        _clear_torch_memory()
                        continue
                    finally:
                        prepared_ot_artifacts_by_target = None
                        _clear_torch_memory()

    write_json(
        OUTPUT_PATH,
        {
            "model_name": MODEL_NAME,
            "dataset_path": RAVEL_DATASET_PATH,
            "dataset_config": RAVEL_DATASET_CONFIG,
            "methods": METHODS,
            "target_vars": TARGET_VARS,
            "source_types": RAVEL_SOURCE_TYPES,
            "token_positions": [token_position.id for token_position in token_positions],
            "catalog_sizes": {
                attribute: len(catalog.texts_by_attribute[attribute]) for attribute in TARGET_ATTRIBUTES
            },
            "runs": all_payloads,
        },
    )
    print(f"Wrote aggregate RAVEL run payload to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
