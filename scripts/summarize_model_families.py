#!/usr/bin/env python3
"""Prespecified descriptive family analysis for the sealed model matrix.

This script does not select families or tests after seeing their outcomes. It reports every one of
the six frozen generators, both sealed splits, all four corruption doses, and all three seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_model_study import load_predictions


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("order", "error", "inventory", "route", "ticket", "checksum")


def main() -> None:
    output = ROOT / "analysis/model_study"
    output.mkdir(parents=True, exist_ok=True)
    data = load_predictions()
    observed_families = tuple(sorted(data.family.unique()))
    if set(observed_families) != set(FAMILIES):
        raise RuntimeError(
            f"family set differs from frozen design: observed={observed_families} expected={FAMILIES}"
        )

    per_seed = data.groupby(
        ["split", "family", "corruption", "seed"], as_index=False
    ).agg(
        records=("record_id", "size"),
        exact=("exact", "mean"),
        copied_stale=("copied_stale", "mean"),
        value_token_nll=("value_token_nll", "mean"),
        whole_target_nll=("whole_target_nll", "mean"),
    )
    per_seed.to_csv(output / "family_per_seed.csv", index=False)
    aggregate = per_seed.groupby(
        ["split", "family", "corruption"], as_index=False
    ).agg(
        seeds=("seed", "size"),
        exact_mean=("exact", "mean"),
        exact_seed_sd=("exact", "std"),
        copied_stale_mean=("copied_stale", "mean"),
        copied_stale_seed_sd=("copied_stale", "std"),
        value_token_nll_mean=("value_token_nll", "mean"),
        value_token_nll_seed_sd=("value_token_nll", "std"),
        whole_target_nll_mean=("whole_target_nll", "mean"),
        whole_target_nll_seed_sd=("whole_target_nll", "std"),
    )
    aggregate.to_csv(output / "family_aggregate.csv", index=False)

    contrasts: dict[str, dict[str, dict[str, float]]] = {}
    indexed = aggregate.set_index(["split", "family", "corruption"])
    for split in ("test", "conflict"):
        contrasts[split] = {}
        for family in FAMILIES:
            low = indexed.loc[(split, family, 0)]
            high = indexed.loc[(split, family, 100)]
            contrasts[split][family] = {
                "exact_100_minus_0": float(high.exact_mean - low.exact_mean),
                "value_nll_100_minus_0": float(
                    high.value_token_nll_mean - low.value_token_nll_mean
                ),
                "copied_stale_100_minus_0": float(
                    high.copied_stale_mean - low.copied_stale_mean
                ),
            }
    (output / "family_contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
