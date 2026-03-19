from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from addition_experiment import _env  # noqa: F401
from pyvene import RotatedSpaceIntervention, VanillaIntervention

from addition_experiment.backbone import AdditionTrainConfig, train_backbone
from addition_experiment.das import DASConfig, run_das_pipeline
from addition_experiment.ot import OTConfig, run_alignment_pipeline, truncate_transport_rows
from addition_experiment.metrics import labels_to_digits, shared_digit_counts
from addition_experiment.pair_bank import build_pair_bank
from addition_experiment.pyvene_utils import build_intervenable, enumerate_canonical_sites, run_intervenable_logits
from addition_experiment.runtime import write_json
from addition_experiment.scm import load_addition_problem
from addition_experiment.seed_sweep import build_seed_sweep_payload


class AdditionExperimentTests(unittest.TestCase):
    def _small_train_config(self) -> AdditionTrainConfig:
        return AdditionTrainConfig(
            seed=42,
            n_train=64,
            n_validation=32,
            hidden_dims=(16, 16),
            train_epochs=1,
            train_batch_size=16,
            eval_batch_size=16,
        )

    def _load_small_model(self, checkpoint_path: str) -> tuple[object, object, object]:
        problem = load_addition_problem(run_checks=False)
        model, config, meta = train_backbone(
            problem=problem,
            train_config=self._small_train_config(),
            checkpoint_path=checkpoint_path,
            device="cpu",
        )
        return problem, model, meta

    def test_metric_spot_checks(self) -> None:
        self.assertEqual(labels_to_digits(torch.tensor([197, 47])).tolist(), [[1, 9, 7], [0, 4, 7]])
        self.assertEqual(shared_digit_counts(torch.tensor([197]), torch.tensor([197])).item(), 3.0)
        self.assertEqual(shared_digit_counts(torch.tensor([197]), torch.tensor([187])).item(), 2.0)
        self.assertEqual(shared_digit_counts(torch.tensor([47]), torch.tensor([147])).item(), 2.0)

    def test_transport_row_truncation_can_renormalize_per_row(self) -> None:
        transport = np.array(
            [
                [0.6, 0.3, 0.1, 0.0],
                [0.4, 0.3, 0.2, 0.1],
            ],
            dtype=np.float64,
        )
        truncated = truncate_transport_rows(transport, [2, 3], renormalize=True)

        np.testing.assert_allclose(truncated[0], np.array([2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(truncated[1], np.array([4.0 / 9.0, 3.0 / 9.0, 2.0 / 9.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(truncated.sum(axis=1), np.ones(2), atol=1e-8)

    def test_problem_checks_and_pair_bank_determinism(self) -> None:
        problem = load_addition_problem(run_checks=True)
        bank_a = build_pair_bank(problem, size=16, seed=123, split="test", verify_with_scm=True)
        bank_b = build_pair_bank(problem, size=16, seed=123, split="test")

        self.assertTrue(torch.equal(bank_a.base_digits, bank_b.base_digits))
        self.assertTrue(torch.equal(bank_a.source_digits, bank_b.source_digits))
        self.assertTrue(torch.equal(bank_a.base_inputs, bank_b.base_inputs))
        self.assertTrue(torch.equal(bank_a.source_inputs, bank_b.source_inputs))
        self.assertTrue(torch.equal(bank_a.base_labels, bank_b.base_labels))
        for variable in ("S1", "C1", "S2", "C2"):
            self.assertTrue(torch.equal(bank_a.cf_labels_by_var[variable], bank_b.cf_labels_by_var[variable]))

    def test_pyvene_intervention_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = str(Path(temp_dir) / "mini_model.pt")
            problem, model, _ = self._load_small_model(checkpoint_path)
            bank = build_pair_bank(problem, size=8, seed=99, split="smoke")
            site = enumerate_canonical_sites(model, resolution=1)[0]

            vanilla = build_intervenable(
                model=model,
                layer=site.layer,
                component=site.component,
                intervention=VanillaIntervention(),
                device="cpu",
                freeze_model=True,
                freeze_intervention=True,
            )
            vanilla_logits = run_intervenable_logits(
                intervenable=vanilla,
                base_inputs=bank.base_inputs,
                source_inputs=bank.source_inputs,
                subspace_dims=site.subspace_dims,
                position=site.position,
                batch_size=4,
                device="cpu",
            )
            self.assertEqual(tuple(vanilla_logits.shape), (8, 200))
            self.assertTrue(torch.isfinite(vanilla_logits).all().item())

            rotated = build_intervenable(
                model=model,
                layer=0,
                component="h[0].output",
                intervention=RotatedSpaceIntervention(embed_dim=int(model.config.hidden_dims[0])),
                device="cpu",
                freeze_model=True,
                freeze_intervention=False,
            )
            rotated_logits = run_intervenable_logits(
                intervenable=rotated,
                base_inputs=bank.base_inputs,
                source_inputs=bank.source_inputs,
                subspace_dims=[0],
                position=0,
                batch_size=4,
                device="cpu",
            )
            self.assertEqual(tuple(rotated_logits.shape), (8, 200))
            self.assertTrue(torch.isfinite(rotated_logits).all().item())

    def test_small_compare_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = str(Path(temp_dir) / "mini_model.pt")
            problem, model, backbone_meta = self._load_small_model(checkpoint_path)
            train_bank = build_pair_bank(problem, size=16, seed=1001, split="train")
            calibration_bank = build_pair_bank(problem, size=16, seed=1002, split="calibration")
            test_bank = build_pair_bank(problem, size=16, seed=1003, split="test")

            gw_payload = run_alignment_pipeline(
                model=model,
                fit_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device="cpu",
                config=OTConfig(
                    method="gw",
                    batch_size=8,
                    ranking_k=2,
                    resolution=2,
                    top_k_values=(1, 2, 4),
                    lambda_values=(0.5, 1.0),
                    selection_verbose=False,
                ),
            )
            ot_payload = run_alignment_pipeline(
                model=model,
                fit_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device="cpu",
                config=OTConfig(
                    method="ot",
                    batch_size=8,
                    ranking_k=2,
                    resolution=2,
                    top_k_values=(1, 2, 4),
                    lambda_values=(0.5, 1.0),
                    selection_verbose=False,
                ),
            )
            fgw_payload = run_alignment_pipeline(
                model=model,
                fit_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device="cpu",
                config=OTConfig(
                    method="fgw",
                    batch_size=8,
                    ranking_k=2,
                    resolution=2,
                    alpha=0.5,
                    top_k_values=(1, 2, 4),
                    lambda_values=(0.5, 1.0),
                    selection_verbose=False,
                ),
            )
            das_payload = run_das_pipeline(
                model=model,
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device="cpu",
                config=DASConfig(
                    batch_size=8,
                    max_epochs=1,
                    learning_rate=1e-3,
                    subspace_dims=(1,),
                    search_layers=(0,),
                    verbose=False,
                ),
            )

            self.assertEqual(len(gw_payload["results"]), 4)
            self.assertEqual(len(ot_payload["results"]), 4)
            self.assertEqual(len(fgw_payload["results"]), 4)
            self.assertEqual(len(das_payload["results"]), 4)

            output_path = Path(temp_dir) / "compare.json"
            write_json(
                output_path,
                {
                    "backbone": backbone_meta,
                    "gw": gw_payload,
                    "ot": ot_payload,
                    "fgw": fgw_payload,
                    "das": das_payload,
                    "results": (
                        gw_payload["results"]
                        + ot_payload["results"]
                        + fgw_payload["results"]
                        + das_payload["results"]
                    ),
                },
            )
            self.assertTrue(output_path.exists())
            with open(output_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(len(payload["results"]), 16)

    def test_seed_sweep_aggregation_summarizes_macro_metrics(self) -> None:
        payload = build_seed_sweep_payload(
            [
                {
                    "seed": 11,
                    "comparison": {
                        "target_vars": ["S1", "C1"],
                        "backbone": {
                            "factual_validation_metrics": {"exact_acc": 0.91, "num_examples": 4000}
                        },
                        "method_runtime_seconds": {"gw": 12.0, "das": 30.0},
                        "method_summary": [
                            {"method": "gw", "exact_acc": 0.25, "mean_shared_digits": 1.0},
                            {"method": "das", "exact_acc": 0.50, "mean_shared_digits": 1.5},
                        ],
                        "results": [
                            {"method": "gw", "variable": "S1", "exact_acc": 0.20, "mean_shared_digits": 0.9},
                            {"method": "gw", "variable": "C1", "exact_acc": 0.30, "mean_shared_digits": 1.1},
                            {"method": "das", "variable": "S1", "exact_acc": 0.45, "mean_shared_digits": 1.4},
                            {"method": "das", "variable": "C1", "exact_acc": 0.55, "mean_shared_digits": 1.6},
                        ],
                    },
                },
                {
                    "seed": 12,
                    "comparison": {
                        "target_vars": ["S1", "C1"],
                        "backbone": {
                            "factual_validation_metrics": {"exact_acc": 0.95, "num_examples": 4000}
                        },
                        "method_runtime_seconds": {"gw": 16.0, "das": 24.0},
                        "method_summary": [
                            {"method": "gw", "exact_acc": 0.35, "mean_shared_digits": 1.2},
                            {"method": "das", "exact_acc": 0.40, "mean_shared_digits": 1.4},
                        ],
                        "results": [
                            {"method": "gw", "variable": "S1", "exact_acc": 0.40, "mean_shared_digits": 1.3},
                            {"method": "gw", "variable": "C1", "exact_acc": 0.30, "mean_shared_digits": 1.1},
                            {"method": "das", "variable": "S1", "exact_acc": 0.35, "mean_shared_digits": 1.3},
                            {"method": "das", "variable": "C1", "exact_acc": 0.45, "mean_shared_digits": 1.5},
                        ],
                    },
                },
            ]
        )

        self.assertEqual(payload["seeds"], [11, 12])
        self.assertEqual(payload["methods"], ["das", "gw"])
        self.assertEqual(payload["target_vars"], ["S1", "C1"])

        backbone_summary = payload["backbone_factual_validation_summary"]
        self.assertAlmostEqual(backbone_summary["exact_acc_mean"], 0.93)
        self.assertAlmostEqual(backbone_summary["exact_acc_std"], 0.02)

        method_summary = {
            str(record["method"]): record for record in payload["method_summary_across_seeds"]
        }
        self.assertAlmostEqual(method_summary["gw"]["exact_acc_mean"], 0.30)
        self.assertAlmostEqual(method_summary["gw"]["exact_acc_std"], 0.05)
        self.assertAlmostEqual(method_summary["gw"]["runtime_seconds_mean"], 14.0)
        self.assertAlmostEqual(method_summary["gw"]["runtime_seconds_std"], 2.0)
        self.assertAlmostEqual(method_summary["das"]["exact_acc_mean"], 0.45)
        self.assertAlmostEqual(method_summary["das"]["exact_acc_std"], 0.05)
        self.assertAlmostEqual(method_summary["das"]["runtime_seconds_mean"], 27.0)
        self.assertAlmostEqual(method_summary["das"]["runtime_seconds_std"], 3.0)

        per_seed_runtime = {
            (int(record["seed"]), str(record["method"])): float(record["runtime_seconds"])
            for record in payload["per_seed_method_runtime"]
        }
        self.assertEqual(per_seed_runtime[(11, "gw")], 12.0)
        self.assertEqual(per_seed_runtime[(12, "das")], 24.0)

        variable_summary = {
            (str(record["method"]), str(record["variable"])): record
            for record in payload["variable_summary_across_seeds"]
        }
        self.assertAlmostEqual(variable_summary[("gw", "S1")]["exact_acc_mean"], 0.30)
        self.assertAlmostEqual(variable_summary[("gw", "S1")]["exact_acc_std"], 0.10)
        self.assertAlmostEqual(variable_summary[("das", "C1")]["mean_shared_digits_mean"], 1.55)


if __name__ == "__main__":
    unittest.main()
