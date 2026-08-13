#!/usr/bin/env python3
"""Validate completeness, provenance, and numerical sanity of the final model matrix."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORRUPTIONS = (0, 25, 50, 100)
SEEDS = (20260808, 20260809, 20260810)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"incomplete manifest: {path}")
    return manifest


def main() -> None:
    frozen = json.loads(
        (ROOT / "configs/model_study_revision2_frozen_manifest.json").read_text()
    )
    expected_split_hash = {
        "test": frozen["sha256"]["test"],
        "conflict": frozen["sha256"]["conflict"],
    }
    training_seconds = 0.0
    evaluation_seconds = 0.0
    training_runs = 0
    evaluation_runs = 0

    for corruption in CORRUPTIONS:
        for seed in SEEDS:
            train_path = ROOT / (
                f"outputs/model_study/corruption_{corruption}_revision2_seed{seed}/manifest.json"
            )
            manifest = load_manifest(train_path)
            assert manifest["corruption_percent"] == corruption
            assert manifest["optimization_seed"] == seed
            assert manifest["recipe_revision"] == 2
            assert manifest["mps_fp16"] is True
            assert manifest["compute_dtype"] == "torch.float16"
            assert manifest["config_sha256"] == frozen["sha256"]["config"]
            assert manifest["train_file_sha256"] == frozen["sha256"][f"train_{corruption}"]
            values = [manifest["training_metrics"]["train_loss"]]
            values.extend(
                value
                for row in manifest["log_history"]
                for key, value in row.items()
                if key in {"loss", "grad_norm", "train_loss"}
            )
            if not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError(f"non-finite training value: {train_path}")
            training_seconds += float(manifest["seconds"])
            training_runs += 1

            adapter_dir = train_path.parent / "adapter"
            if not (adapter_dir / "adapter_model.safetensors").exists():
                raise FileNotFoundError(f"adapter missing: {adapter_dir}")

            for split in ("test", "conflict"):
                eval_dir = ROOT / (
                    f"outputs/model_study/eval_{split}_corruption{corruption}_seed{seed}"
                )
                eval_manifest = load_manifest(eval_dir / "manifest.json")
                assert eval_manifest["split"] == split
                assert eval_manifest["records"] == 600
                assert eval_manifest["split_sha256"] == expected_split_hash[split]
                assert eval_manifest["mps_fp16"] is True
                assert "revision2" in str(eval_manifest["adapter"])
                predictions = pd.read_parquet(eval_dir / "predictions.parquet")
                assert len(predictions) == predictions.record_id.nunique() == 600
                for column in ("value_token_nll", "whole_target_nll"):
                    if not predictions[column].map(math.isfinite).all():
                        raise RuntimeError(f"non-finite {column}: {eval_dir}")
                evaluation_seconds += float(eval_manifest["seconds"])
                evaluation_runs += 1

    for split in ("test", "conflict"):
        eval_dir = ROOT / f"outputs/model_study/eval_{split}_base"
        manifest = load_manifest(eval_dir / "manifest.json")
        assert manifest["base_only"] is True
        assert manifest["records"] == 600
        assert manifest["split_sha256"] == expected_split_hash[split]
        assert manifest["mps_fp16"] is True
        predictions = pd.read_parquet(eval_dir / "predictions.parquet")
        assert len(predictions) == predictions.record_id.nunique() == 600
        evaluation_seconds += float(manifest["seconds"])
        evaluation_runs += 1

    report = {
        "status": "PASS",
        "training_runs": training_runs,
        "sealed_adapter_evaluations": evaluation_runs - 2,
        "base_evaluations": 2,
        "records_per_evaluation": 600,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "paid_compute_cad": 0.0,
        "revision1_adapters_in_final_matrix": 0,
    }
    output = ROOT / "analysis/model_study"
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
