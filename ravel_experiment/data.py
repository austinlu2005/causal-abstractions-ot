"""RAVEL task definitions, dataset loading, candidate catalogs, and pair banks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Callable

from datasets import get_dataset_split_names, load_dataset
import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from mcqa_experiment.intervention import forward_factual_logits

from .runtime import resolve_device


DATASET_PATH = os.environ.get("RAVEL_DATASET_PATH", "mib-bench/ravel")
DATASET_NAME = os.environ.get("RAVEL_DATASET_CONFIG")
_DATASET_CONFIG_UNSET = object()

TARGET_ATTRIBUTES = ("Continent", "Country", "Language")
SOURCE_TYPES = ("attribute_counterfactual", "wikipedia_counterfactual")


@dataclass(frozen=True)
class TokenPosition:
    """Minimal task-local token-position descriptor."""

    resolver: Callable[[dict[str, object], object], list[int]]
    id: str

    def resolve(self, input_dict: dict[str, object], tokenizer) -> int:
        positions = self.resolver(input_dict, tokenizer)
        if not positions:
            raise ValueError(f"Token position {self.id} returned no indices")
        return int(positions[0])


@dataclass(frozen=True)
class CandidateCatalog:
    """Token-level answer catalog used for filtering, metrics, and signatures."""

    texts_by_attribute: dict[str, tuple[str, ...]]
    token_ids_by_attribute: dict[str, torch.Tensor]
    variant_token_ids_by_attribute: dict[str, torch.Tensor]
    token_id_to_local_index_by_attribute: dict[str, dict[int, int]]
    token_id_to_text_by_attribute: dict[str, dict[int, str]]
    text_to_token_id_by_attribute: dict[str, dict[str, int]]
    global_token_ids: torch.Tensor
    global_variant_token_ids: torch.Tensor
    global_token_id_to_index: dict[int, int]


@dataclass(frozen=True)
class RAVELPairBank:
    """Tokenized base/source split for one RAVEL target variable and source type."""

    split: str
    source_type: str
    target_var: str
    dataset_name: str
    base_input_ids: torch.Tensor
    base_attention_mask: torch.Tensor
    source_input_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    labels: torch.Tensor
    answer_token_ids: torch.Tensor
    base_answer_token_ids: torch.Tensor
    base_inputs: list[dict[str, object]]
    source_inputs: list[dict[str, object]]
    base_position_by_id: dict[str, torch.Tensor]
    source_position_by_id: dict[str, torch.Tensor]
    candidate_token_ids: torch.Tensor
    candidate_variant_token_ids: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_texts: list[list[str]]
    global_candidate_token_ids: torch.Tensor
    global_candidate_variant_token_ids: torch.Tensor
    base_global_labels: torch.Tensor
    expected_global_labels: torch.Tensor
    query_attributes: list[str]
    changed_mask: torch.Tensor
    base_answer_texts: list[str]
    expected_answer_texts: list[str]

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])

    def metadata(self) -> dict[str, object]:
        query_counts = {attribute: self.query_attributes.count(attribute) for attribute in TARGET_ATTRIBUTES}
        return {
            "split": self.split,
            "source_type": self.source_type,
            "target_var": self.target_var,
            "size": self.size,
            "dataset_name": self.dataset_name,
            "changed_count": int(self.changed_mask.sum().item()),
            "changed_rate": float(self.changed_mask.float().mean().item()) if self.size else 0.0,
            "query_distribution": query_counts,
        }


class RAVELPairDataset(torch.utils.data.Dataset):
    """Dataset view for DAS training and evaluation."""

    def __init__(self, bank: RAVELPairBank):
        self.bank = bank

    def __len__(self) -> int:
        return self.bank.size

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "base_input_ids": self.bank.base_input_ids[index],
            "base_attention_mask": self.bank.base_attention_mask[index],
            "source_input_ids": self.bank.source_input_ids[index],
            "source_attention_mask": self.bank.source_attention_mask[index],
            "labels": self.bank.labels[index],
            "answer_token_id": self.bank.answer_token_ids[index],
            "base_positions": {key: value[index] for key, value in self.bank.base_position_by_id.items()},
            "source_positions": {key: value[index] for key, value in self.bank.source_position_by_id.items()},
        }


def normalize_answer_text(text: object) -> str:
    return str(text).strip()


def _normalize_split_name(split: str) -> str:
    lowered = str(split).lower()
    if lowered in {"validation", "valid", "dev"}:
        return "val"
    return lowered


def _parse_record(record: dict[str, object]) -> dict[str, object]:
    parsed = {
        "prompt": str(record.get("prompt", "")),
        "template": str(record.get("template", "")),
        "entity": str(record.get("entity", "")),
        "attribute": str(record.get("attribute", "")),
    }
    for attribute in TARGET_ATTRIBUTES:
        parsed[attribute] = normalize_answer_text(record.get(attribute, ""))
    return parsed


def parse_ravel_example(row: dict[str, object]) -> dict[str, object]:
    parsed = _parse_record(row)
    for source_type in SOURCE_TYPES:
        source_record = row.get(source_type)
        parsed[source_type] = _parse_record(source_record) if isinstance(source_record, dict) else None
    return parsed


def _encode_first_token_variants(answer_text: str, tokenizer) -> tuple[int, int] | None:
    normalized = normalize_answer_text(answer_text)
    if not normalized:
        return None
    variant_ids = []
    for candidate in (" " + normalized, normalized):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            variant_ids.append(int(ids[0]))
    if not variant_ids:
        return None
    if len(variant_ids) == 1:
        variant_ids.append(variant_ids[0])
    return (variant_ids[0], variant_ids[1])


def build_candidate_catalog(rows_by_split: dict[str, list[dict[str, object]]], tokenizer) -> CandidateCatalog:
    texts_by_attribute: dict[str, set[str]] = {attribute: set() for attribute in TARGET_ATTRIBUTES}
    for rows in rows_by_split.values():
        for row in rows:
            for attribute in TARGET_ATTRIBUTES:
                text = normalize_answer_text(row.get(attribute, ""))
                if text:
                    texts_by_attribute[attribute].add(text)
            for source_type in SOURCE_TYPES:
                source = row.get(source_type)
                if not isinstance(source, dict):
                    continue
                for attribute in TARGET_ATTRIBUTES:
                    text = normalize_answer_text(source.get(attribute, ""))
                    if text:
                        texts_by_attribute[attribute].add(text)

    final_texts_by_attribute: dict[str, tuple[str, ...]] = {}
    token_ids_by_attribute: dict[str, torch.Tensor] = {}
    variant_token_ids_by_attribute: dict[str, torch.Tensor] = {}
    token_id_to_local_index_by_attribute: dict[str, dict[int, int]] = {}
    token_id_to_text_by_attribute: dict[str, dict[int, str]] = {}
    text_to_token_id_by_attribute: dict[str, dict[str, int]] = {}
    global_variant_by_token_id: dict[int, tuple[int, int]] = {}

    for attribute in TARGET_ATTRIBUTES:
        variants_by_text = {}
        texts_by_token_id: dict[int, set[str]] = {}
        for text in sorted(texts_by_attribute[attribute]):
            variants = _encode_first_token_variants(text, tokenizer)
            if variants is None:
                continue
            variants_by_text[text] = variants
            texts_by_token_id.setdefault(int(variants[0]), set()).add(text)

        valid_texts = [
            text
            for text, variants in variants_by_text.items()
            if len(texts_by_token_id[int(variants[0])]) == 1
        ]
        valid_texts = tuple(sorted(valid_texts))
        final_texts_by_attribute[attribute] = valid_texts
        attribute_token_ids = [int(variants_by_text[text][0]) for text in valid_texts]
        attribute_variants = [list(variants_by_text[text]) for text in valid_texts]
        token_ids_by_attribute[attribute] = torch.tensor(attribute_token_ids, dtype=torch.long)
        variant_token_ids_by_attribute[attribute] = torch.tensor(attribute_variants, dtype=torch.long)
        token_id_to_local_index_by_attribute[attribute] = {
            int(token_id): index for index, token_id in enumerate(attribute_token_ids)
        }
        token_id_to_text_by_attribute[attribute] = {
            int(token_id): text for token_id, text in zip(attribute_token_ids, valid_texts)
        }
        text_to_token_id_by_attribute[attribute] = {
            text: int(variants_by_text[text][0]) for text in valid_texts
        }
        for text in valid_texts:
            token_id = int(variants_by_text[text][0])
            global_variant_by_token_id.setdefault(token_id, variants_by_text[text])

    global_token_ids = torch.tensor(sorted(global_variant_by_token_id), dtype=torch.long)
    global_variant_token_ids = torch.tensor(
        [list(global_variant_by_token_id[int(token_id)]) for token_id in global_token_ids.tolist()],
        dtype=torch.long,
    )
    global_token_id_to_index = {
        int(token_id): index for index, token_id in enumerate(global_token_ids.tolist())
    }
    return CandidateCatalog(
        texts_by_attribute=final_texts_by_attribute,
        token_ids_by_attribute=token_ids_by_attribute,
        variant_token_ids_by_attribute=variant_token_ids_by_attribute,
        token_id_to_local_index_by_attribute=token_id_to_local_index_by_attribute,
        token_id_to_text_by_attribute=token_id_to_text_by_attribute,
        text_to_token_id_by_attribute=text_to_token_id_by_attribute,
        global_token_ids=global_token_ids,
        global_variant_token_ids=global_variant_token_ids,
        global_token_id_to_index=global_token_id_to_index,
    )


def _find_entity_last_token(input_dict: dict[str, object], tokenizer) -> list[int]:
    prompt = str(input_dict["prompt"])
    entity = str(input_dict["entity"])
    end_index = prompt.rfind(entity)
    if end_index < 0:
        raise ValueError(f"Could not find entity {entity!r} in prompt: {prompt!r}")
    substring = prompt[: end_index + len(entity)]
    tokenized = tokenizer(substring, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    return [len(tokenized) - 1]


def get_token_positions(tokenizer) -> list[TokenPosition]:
    def entity_last_token(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        return _find_entity_last_token(input_dict, current_tokenizer)

    def last_token(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        prompt = str(input_dict["prompt"])
        tokenized = current_tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        return [len(tokenized) - 1]

    return [
        TokenPosition(entity_last_token, "entity_last_token"),
        TokenPosition(last_token, "last_token"),
    ]


def load_public_ravel_datasets(
    *,
    size: int | None = None,
    hf_token: str | None = None,
    dataset_path: str | None = None,
    dataset_name: str | None | object = _DATASET_CONFIG_UNSET,
) -> dict[str, list[dict[str, object]]]:
    datasets_by_split: dict[str, list[dict[str, object]]] = {}
    resolved_dataset_path = dataset_path or DATASET_PATH
    resolved_dataset_name = DATASET_NAME if dataset_name is _DATASET_CONFIG_UNSET else dataset_name
    dataset_path_obj = Path(resolved_dataset_path)
    if dataset_path_obj.exists():
        candidate_splits = tuple(
            split_file.stem for split_file in sorted(dataset_path_obj.glob("*.jsonl")) if split_file.stem
        )
        if not candidate_splits:
            raise FileNotFoundError(f"No .jsonl splits found under local dataset path {dataset_path_obj}")
    else:
        if resolved_dataset_name:
            candidate_splits = tuple(
                get_dataset_split_names(resolved_dataset_path, resolved_dataset_name, token=hf_token)
            )
        else:
            candidate_splits = tuple(get_dataset_split_names(resolved_dataset_path, token=hf_token))

    for split in candidate_splits:
        normalized_split = _normalize_split_name(split)
        if dataset_path_obj.exists():
            split_file = dataset_path_obj / f"{split}.jsonl"
            dataset = load_dataset("json", data_files={split: str(split_file)}, split=split)
        else:
            if resolved_dataset_name:
                dataset = load_dataset(resolved_dataset_path, resolved_dataset_name, split=split, token=hf_token)
            else:
                dataset = load_dataset(resolved_dataset_path, split=split, token=hf_token)
        if size is not None:
            dataset = dataset.select(range(min(int(size), len(dataset))))
        datasets_by_split[normalized_split] = [parse_ravel_example(row) for row in dataset]
    return datasets_by_split


def _gather_variant_logits(logits: torch.Tensor, variant_token_ids: torch.Tensor) -> torch.Tensor:
    batch_size, num_classes, num_variants = variant_token_ids.shape
    gathered = torch.gather(
        logits,
        dim=1,
        index=variant_token_ids.to(logits.device).reshape(batch_size, num_classes * num_variants),
    )
    gathered = gathered.reshape(batch_size, num_classes, num_variants)
    return gathered.max(dim=-1).values


def filter_correct_examples(
    *,
    model,
    tokenizer,
    rows_by_split: dict[str, list[dict[str, object]]],
    catalog: CandidateCatalog,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[dict[str, object]]]:
    filtered: dict[str, list[dict[str, object]]] = {}
    for split, rows in rows_by_split.items():
        valid_rows = [
            row
            for row in rows
            if str(row.get("attribute")) in TARGET_ATTRIBUTES
            and normalize_answer_text(row.get(str(row.get("attribute")), "")) in catalog.text_to_token_id_by_attribute[str(row.get("attribute"))]
        ]
        print(f"[filter] split={split} total_rows={len(rows)} valid_rows={len(valid_rows)}")
        kept_rows: list[dict[str, object]] = []
        for attribute in TARGET_ATTRIBUTES:
            attribute_rows = [row for row in valid_rows if str(row["attribute"]) == attribute]
            if not attribute_rows:
                continue
            candidate_variant_token_ids = catalog.variant_token_ids_by_attribute[attribute].to(torch.long)
            labels = torch.tensor(
                [
                    catalog.token_id_to_local_index_by_attribute[attribute][
                        catalog.text_to_token_id_by_attribute[attribute][normalize_answer_text(row[attribute])]
                    ]
                    for row in attribute_rows
                ],
                dtype=torch.long,
            )
            prompts = [str(row["prompt"]) for row in attribute_rows]
            row_keep_mask: list[bool] = []
            batch_starts = range(0, len(attribute_rows), batch_size)
            batch_iterator = batch_starts
            if tqdm is not None:
                batch_iterator = tqdm(
                    batch_starts,
                    desc=f"Filtering {split}:{attribute}",
                    leave=False,
                    total=(len(attribute_rows) + batch_size - 1) // batch_size,
                )
            for start in batch_iterator:
                end = min(start + batch_size, len(attribute_rows))
                encoded = tokenizer(
                    prompts[start:end],
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                logits = forward_factual_logits(
                    model=model,
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                )
                variant_ids = candidate_variant_token_ids.unsqueeze(0).expand(logits.shape[0], -1, -1)
                target_logits = _gather_variant_logits(logits, variant_ids)
                predictions = target_logits.argmax(dim=-1).detach().cpu()
                row_keep_mask.extend(
                    bool(prediction == label)
                    for prediction, label in zip(predictions.tolist(), labels[start:end].tolist())
                )
            kept_rows.extend([row for row, keep in zip(attribute_rows, row_keep_mask) if keep])
        filtered[split] = kept_rows
        print(f"[filter] split={split} kept={len(filtered[split])}/{len(rows)}")
    return filtered


def _resolve_answer_text(record: dict[str, object], query_attribute: str) -> str:
    return normalize_answer_text(record.get(query_attribute, ""))


def _row_is_valid_for_target(
    row: dict[str, object],
    *,
    target_var: str,
    source_type: str,
    catalog: CandidateCatalog,
) -> bool:
    query_attribute = str(row.get("attribute", ""))
    if query_attribute not in TARGET_ATTRIBUTES:
        return False
    base_answer_text = _resolve_answer_text(row, query_attribute)
    if base_answer_text not in catalog.text_to_token_id_by_attribute[query_attribute]:
        return False
    source = row.get(source_type)
    if not isinstance(source, dict):
        return False
    source_target_text = normalize_answer_text(source.get(target_var, ""))
    return source_target_text in catalog.text_to_token_id_by_attribute[target_var]


def build_pair_bank(
    *,
    tokenizer,
    token_positions: list[TokenPosition],
    rows: list[dict[str, object]],
    catalog: CandidateCatalog,
    split: str,
    source_type: str,
    target_var: str,
    size: int,
    seed: int,
    dataset_name: str,
) -> RAVELPairBank:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    rng = random.Random(int(seed))
    valid_rows = [
        row
        for row in rows
        if _row_is_valid_for_target(row, target_var=target_var, source_type=source_type, catalog=catalog)
    ]
    positive_rows = [row for row in valid_rows if str(row["attribute"]) == target_var]
    negative_rows = [row for row in valid_rows if str(row["attribute"]) != target_var]
    requested_positive = min(len(positive_rows), (size + 1) // 2)
    requested_negative = size - requested_positive
    if requested_negative > len(negative_rows):
        requested_negative = len(negative_rows)
        requested_positive = size - requested_negative
    if requested_positive > len(positive_rows):
        raise ValueError(
            f"Not enough rows for target_var={target_var} source_type={source_type}: "
            f"positives={len(positive_rows)} negatives={len(negative_rows)} requested={size}"
        )
    rng.shuffle(positive_rows)
    rng.shuffle(negative_rows)
    selected_rows = positive_rows[:requested_positive] + negative_rows[:requested_negative]
    if len(selected_rows) != size:
        raise ValueError(
            f"Could not build size={size} bank for target_var={target_var} source_type={source_type}; "
            f"constructed only {len(selected_rows)} rows"
        )
    rng.shuffle(selected_rows)

    base_inputs = [row for row in selected_rows]
    source_inputs = [row[source_type] for row in selected_rows]
    base_prompts = [str(row["prompt"]) for row in base_inputs]
    source_prompts = [str(row["prompt"]) for row in source_inputs]
    base_encoded = tokenizer(base_prompts, padding=True, return_tensors="pt", add_special_tokens=True)
    source_encoded = tokenizer(source_prompts, padding=True, return_tensors="pt", add_special_tokens=True)

    base_position_by_id = {
        token_position.id: torch.tensor(
            [token_position.resolve(base_input, tokenizer) for base_input in base_inputs],
            dtype=torch.long,
        )
        for token_position in token_positions
    }
    source_position_by_id = {
        token_position.id: torch.tensor(
            [token_position.resolve(source_input, tokenizer) for source_input in source_inputs],
            dtype=torch.long,
        )
        for token_position in token_positions
    }

    max_classes = max(int(token_ids.shape[0]) for token_ids in catalog.token_ids_by_attribute.values())
    candidate_token_rows = []
    candidate_variant_rows = []
    candidate_mask_rows = []
    candidate_texts: list[list[str]] = []
    labels = []
    answer_token_ids = []
    base_answer_token_ids = []
    base_global_labels = []
    expected_global_labels = []
    query_attributes = []
    changed_mask = []
    base_answer_texts = []
    expected_answer_texts = []

    for base_input, source_input in zip(base_inputs, source_inputs):
        query_attribute = str(base_input["attribute"])
        is_changed = query_attribute == target_var
        base_answer_text = _resolve_answer_text(base_input, query_attribute)
        expected_answer_text = (
            normalize_answer_text(source_input[target_var]) if is_changed else base_answer_text
        )
        base_token_id = catalog.text_to_token_id_by_attribute[query_attribute][base_answer_text]
        expected_token_id = catalog.text_to_token_id_by_attribute[query_attribute][expected_answer_text]
        local_index = catalog.token_id_to_local_index_by_attribute[query_attribute][expected_token_id]
        base_global_index = catalog.global_token_id_to_index[base_token_id]
        expected_global_index = catalog.global_token_id_to_index[expected_token_id]

        attribute_token_ids = catalog.token_ids_by_attribute[query_attribute]
        attribute_variant_token_ids = catalog.variant_token_ids_by_attribute[query_attribute]
        attribute_texts = list(catalog.texts_by_attribute[query_attribute])
        pad_amount = max_classes - int(attribute_token_ids.shape[0])
        candidate_token_row = torch.full((max_classes,), 0, dtype=torch.long)
        candidate_variant_row = torch.zeros((max_classes, 2), dtype=torch.long)
        candidate_mask_row = torch.zeros((max_classes,), dtype=torch.bool)
        candidate_token_row[: attribute_token_ids.shape[0]] = attribute_token_ids
        candidate_variant_row[: attribute_variant_token_ids.shape[0]] = attribute_variant_token_ids
        candidate_mask_row[: attribute_token_ids.shape[0]] = True

        candidate_token_rows.append(candidate_token_row)
        candidate_variant_rows.append(candidate_variant_row)
        candidate_mask_rows.append(candidate_mask_row)
        candidate_texts.append(attribute_texts)
        labels.append(int(local_index))
        answer_token_ids.append(int(expected_token_id))
        base_answer_token_ids.append(int(base_token_id))
        base_global_labels.append(int(base_global_index))
        expected_global_labels.append(int(expected_global_index))
        query_attributes.append(query_attribute)
        changed_mask.append(bool(is_changed))
        base_answer_texts.append(base_answer_text)
        expected_answer_texts.append(expected_answer_text)

    return RAVELPairBank(
        split=split,
        source_type=source_type,
        target_var=target_var,
        dataset_name=dataset_name,
        base_input_ids=base_encoded["input_ids"].to(torch.long),
        base_attention_mask=base_encoded["attention_mask"].to(torch.long),
        source_input_ids=source_encoded["input_ids"].to(torch.long),
        source_attention_mask=source_encoded["attention_mask"].to(torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        answer_token_ids=torch.tensor(answer_token_ids, dtype=torch.long),
        base_answer_token_ids=torch.tensor(base_answer_token_ids, dtype=torch.long),
        base_inputs=base_inputs,
        source_inputs=source_inputs,
        base_position_by_id=base_position_by_id,
        source_position_by_id=source_position_by_id,
        candidate_token_ids=torch.stack(candidate_token_rows, dim=0),
        candidate_variant_token_ids=torch.stack(candidate_variant_rows, dim=0),
        candidate_mask=torch.stack(candidate_mask_rows, dim=0),
        candidate_texts=candidate_texts,
        global_candidate_token_ids=catalog.global_token_ids,
        global_candidate_variant_token_ids=catalog.global_variant_token_ids,
        base_global_labels=torch.tensor(base_global_labels, dtype=torch.long),
        expected_global_labels=torch.tensor(expected_global_labels, dtype=torch.long),
        query_attributes=query_attributes,
        changed_mask=torch.tensor(changed_mask, dtype=torch.bool),
        base_answer_texts=base_answer_texts,
        expected_answer_texts=expected_answer_texts,
    )


def build_pair_banks(
    *,
    tokenizer,
    token_positions: list[TokenPosition],
    rows_by_split: dict[str, list[dict[str, object]]],
    catalog: CandidateCatalog,
    target_vars: tuple[str, ...],
    source_types: tuple[str, ...],
    split_seed: int = 0,
    train_pool_size: int | None = None,
    calibration_pool_size: int | None = None,
    test_pool_size: int | None = None,
    dataset_name: str = "mib-bench/ravel",
) -> tuple[dict[str, dict[str, dict[str, RAVELPairBank]]], dict[str, object]]:
    train_rows = list(rows_by_split.get("train", []))
    calibration_rows = list(rows_by_split.get("val", rows_by_split.get("validation", [])))
    test_rows = list(rows_by_split.get("test", []))
    if not train_rows or not calibration_rows or not test_rows:
        raise ValueError("Expected non-empty train/val/test rows for RAVEL pair-bank construction")

    requested_sizes = {
        "train": int(train_pool_size) if train_pool_size is not None else len(train_rows),
        "calibration": int(calibration_pool_size) if calibration_pool_size is not None else len(calibration_rows),
        "test": int(test_pool_size) if test_pool_size is not None else len(test_rows),
    }

    split_rows = {
        "train": train_rows,
        "calibration": calibration_rows,
        "test": test_rows,
    }
    banks_by_split: dict[str, dict[str, dict[str, RAVELPairBank]]] = {
        "train": {},
        "calibration": {},
        "test": {},
    }
    for split_index, split in enumerate(("train", "calibration", "test")):
        for source_index, source_type in enumerate(source_types):
            banks_by_split[split][source_type] = {}
            for target_index, target_var in enumerate(target_vars):
                seed = int(split_seed) + 10_000 * split_index + 1_000 * source_index + 100 * target_index + 17
                banks_by_split[split][source_type][target_var] = build_pair_bank(
                    tokenizer=tokenizer,
                    token_positions=token_positions,
                    rows=split_rows[split],
                    catalog=catalog,
                    split=split,
                    source_type=source_type,
                    target_var=target_var,
                    size=requested_sizes[split],
                    seed=seed,
                    dataset_name=dataset_name,
                )

    metadata = {
        split: {
            source_type: {
                target_var: bank.metadata()
                for target_var, bank in source_banks.items()
            }
            for source_type, source_banks in split_banks.items()
        }
        for split, split_banks in banks_by_split.items()
    }
    return banks_by_split, metadata


def load_filtered_ravel_pipeline(
    *,
    model_name: str,
    device: str | None = None,
    batch_size: int = 16,
    dataset_size: int | None = None,
    hf_token: str | None = None,
    dataset_path: str | None = None,
    dataset_name: str | None | object = _DATASET_CONFIG_UNSET,
) -> tuple[object, object, list[TokenPosition], CandidateCatalog, dict[str, list[dict[str, object]]]]:
    import transformers

    torch_device = resolve_device(device)
    print(f"[load] device={torch_device} model={model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if torch_device.type in {"cuda", "mps"} else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=hf_token,
        attn_implementation="eager",
    )
    model.to(torch_device)
    model.eval()
    token_positions = get_token_positions(tokenizer)
    resolved_dataset_path = dataset_path or DATASET_PATH
    resolved_dataset_name = DATASET_NAME if dataset_name is _DATASET_CONFIG_UNSET else dataset_name
    print(
        f"[load] loading RAVEL datasets path={resolved_dataset_path} "
        f"config={resolved_dataset_name!r} size_cap={dataset_size}"
    )
    public_datasets = load_public_ravel_datasets(
        size=dataset_size,
        hf_token=hf_token,
        dataset_path=resolved_dataset_path,
        dataset_name=resolved_dataset_name,
    )
    catalog = build_candidate_catalog(public_datasets, tokenizer)
    print(
        "[load] candidate catalog sizes "
        + ", ".join(
            f"{attribute}={len(catalog.texts_by_attribute[attribute])}" for attribute in TARGET_ATTRIBUTES
        )
    )
    print("[load] starting factual filtering")
    filtered_datasets = filter_correct_examples(
        model=model,
        tokenizer=tokenizer,
        rows_by_split=public_datasets,
        catalog=catalog,
        batch_size=batch_size,
        device=torch_device,
    )
    print("[load] factual filtering complete")
    return model, tokenizer, token_positions, catalog, filtered_datasets
