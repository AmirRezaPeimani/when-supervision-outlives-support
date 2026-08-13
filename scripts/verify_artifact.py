#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_close(observed: float, expected: float, tolerance: float = 5e-5) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"expected {expected}, observed {observed}")


def audit_row(corpus: str) -> pd.Series:
    directory = "glaive" if corpus == "glaive" else "toolace_retool"
    frame = pd.read_csv(ROOT / f"analysis/corpus_audit/{directory}/incidence.csv")
    row = frame[
        (frame.stratum == "all_admitted")
        & (frame.corpus == corpus)
        & (frame.policy == "bfd_split")
        & (frame.max_length == 512)
    ]
    if len(row) != 1:
        raise AssertionError(f"missing unique audit row for {corpus}")
    return row.iloc[0]


def main() -> None:
    expected = {
        "toolace": (353, 913, 0.3866),
        "retool": (315, 1606, 0.1961),
        "glaive": (431, 6698, 0.0643),
    }
    for corpus, (unsupported, trained, prevalence) in expected.items():
        row = audit_row(corpus)
        assert int(row.unsupported_units) == unsupported
        assert int(row.trained_units) == trained
        require_close(float(row.conditional_prevalence), prevalence)

    materialization = pd.read_csv(ROOT / "outputs/materialization/summary.csv")
    grouped = materialization[
        (materialization.corpus == "toolace")
        & (materialization.policy == "grouped_closure")
        & (materialization.max_length == 512)
    ].iloc[0]
    assert (int(grouped.retained_targets), int(grouped.valid_targets)) == (875, 918)
    require_close(float(grouped.input_token_ratio), 0.2747)

    schema = pd.read_csv(ROOT / "outputs/schema_repair/summary.csv")
    toolace = schema[(schema.corpus == "toolace") & (schema.max_length == 512)]
    rates = dict(zip(toolace["mode"], toolace.repair_rate, strict=True))
    require_close(float(rates["whole_schema"]), 0.2860)
    require_close(float(rates["parameter_slice"]), 0.7315)

    strict = pd.read_csv(ROOT / "analysis/case_sensitive/case_sensitive_512.csv")
    if float(strict.delta_conditional_percentage_points.abs().max()) > 0.23:
        raise AssertionError("exact-case sensitivity exceeds the reported bound")

    manifest = json.loads((ROOT / "outputs/materialization/manifest.json").read_text())
    assert manifest["serialized_validation_relation"] == "audit_edges_for_view/causal_attention_relation"
    assert manifest["serialized_examples_validated"] == 34128

    glaive_manifest = json.loads((ROOT / "data/processed/glaive_manifest.json").read_text())
    selected = ROOT / "data/processed/glaive_selected.parquet"
    assert sha256(selected) == glaive_manifest["selected_sha256"]
    print("artifact headline checks: PASS")


if __name__ == "__main__":
    main()
