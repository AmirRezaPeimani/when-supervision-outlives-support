#!/usr/bin/env python3
"""Post-hoc diagnostic for format sensitivity in the frozen model endpoint.

The prespecified primary endpoint remains ``exact`` as produced by
``evaluate_model_study.py``.  This script was added only after observing that
one clean-support seed often generated ``VALUE:<target>`` instead of
``VALUE=<target>``.  It therefore cannot replace or redefine the primary
endpoint.  It reports whether the verbatim target (or stale distractor) occurs
anywhere in the retained raw generation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_model_study import load_predictions


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "analysis/model_study/posthoc_format_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    data = load_predictions().copy()
    data["target_substring"] = [
        str(target) in str(raw)
        for target, raw in zip(data["target"], data["raw_generation"])
    ]
    data["stale_substring"] = [
        False if pd.isna(stale) else str(stale) in str(raw)
        for stale, raw in zip(data["stale_value"], data["raw_generation"])
    ]

    per_seed = data.groupby(
        ["split", "corruption", "seed"], as_index=False
    ).agg(
        records=("record_id", "size"),
        primary_exact=("exact", "mean"),
        target_substring=("target_substring", "mean"),
        stale_substring=("stale_substring", "mean"),
    )
    per_seed.to_csv(output / "per_seed.csv", index=False)

    aggregate = per_seed.groupby(
        ["split", "corruption"], as_index=False
    ).agg(
        seeds=("seed", "size"),
        primary_exact_mean=("primary_exact", "mean"),
        primary_exact_seed_sd=("primary_exact", "std"),
        target_substring_mean=("target_substring", "mean"),
        target_substring_seed_sd=("target_substring", "std"),
        stale_substring_mean=("stale_substring", "mean"),
        stale_substring_seed_sd=("stale_substring", "std"),
    )
    aggregate.to_csv(output / "aggregate.csv", index=False)

    family = data.groupby(
        ["split", "family", "corruption", "seed"], as_index=False
    ).agg(
        records=("record_id", "size"),
        primary_exact=("exact", "mean"),
        target_substring=("target_substring", "mean"),
        stale_substring=("stale_substring", "mean"),
    )
    family.to_csv(output / "family_per_seed.csv", index=False)

    manifest = {
        "status": "complete",
        "analysis_status": "post-hoc; descriptive only; does not replace the frozen primary endpoint",
        "trigger": (
            "Observed VALUE:<target> rather than VALUE=<target> in one 0% corruption "
            "clean-test optimization seed"
        ),
        "definition": {
            "target_substring": "verbatim target occurs anywhere in raw_generation",
            "stale_substring": "verbatim stale_value occurs anywhere in raw_generation",
        },
        "rows": int(len(data)),
        "evaluations": int(data.groupby(["split", "corruption", "seed"]).ngroups),
        "primary_endpoint_unchanged": True,
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
