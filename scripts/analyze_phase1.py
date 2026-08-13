#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260808
BOOTSTRAPS = 2000


def cluster_interval(units: pd.DataFrame) -> tuple[float, float, float]:
    clusters = units.groupby("row_id", sort=False).agg(
        numerator=("unsupported_target", "sum"), denominator=("target_trained", "sum")
    )
    denominator = float(clusters.denominator.sum())
    point = float(clusters.numerator.sum() / denominator) if denominator else 0.0
    if denominator == 0 or len(clusters) < 2:
        return point, point, point
    values = clusters[["numerator", "denominator"]].to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAPS):
        sampled = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        if sampled[1] > 0:
            draws.append(sampled[0] / sampled[1])
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def aggregate_filtered_edges(edges: pd.DataFrame) -> pd.DataFrame:
    keys = ["corpus", "row_id", "unit_id", "policy", "max_length"]
    if edges.empty:
        return pd.DataFrame(
            columns=keys
            + [
                "valid_oracle",
                "target_trained",
                "unsupported_target",
                "unsupported_target_tokens",
                "unsupported_target_token_occurrences",
            ]
        )
    units = edges.groupby(keys, as_index=False, sort=False).agg(
        valid_oracle=("source_located", "min"),
        target_located=("target_located", "min"),
        target_trained=("target_trained", "max"),
        unsupported_target=("unsupported_target", "max"),
    ).assign(valid_oracle=lambda frame: frame.valid_oracle & frame.target_located)
    token_rows = edges[edges.unsupported_target].explode("unsupported_target_token_indices")
    token_rows = token_rows.dropna(subset=["unsupported_target_token_indices"])
    if token_rows.empty:
        units["unsupported_target_tokens"] = 0
        units["unsupported_target_token_occurrences"] = 0
        return units
    token_counts = token_rows.groupby(keys, as_index=False).unsupported_target_token_indices.nunique()
    token_counts = token_counts.rename(columns={"unsupported_target_token_indices": "unsupported_target_tokens"})
    units = units.merge(token_counts, on=keys, how="left")
    units["unsupported_target_tokens"] = units.unsupported_target_tokens.fillna(0).astype(int)
    position_rows = edges[edges.unsupported_target].explode("unsupported_target_positions")
    position_rows = position_rows.dropna(subset=["unsupported_target_positions"])
    position_counts = position_rows.groupby(keys, as_index=False).unsupported_target_positions.nunique()
    position_counts = position_counts.rename(
        columns={"unsupported_target_positions": "unsupported_target_token_occurrences"}
    )
    units = units.merge(position_counts, on=keys, how="left")
    units["unsupported_target_token_occurrences"] = (
        units.unsupported_target_token_occurrences.fillna(0).astype(int)
    )
    return units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/raw/primary_audit")
    parser.add_argument("--output-dir", default="analysis/corpus_audit/toolace_retool")
    args = parser.parse_args()
    source = ROOT / args.input_dir
    destination = ROOT / args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("Phase I manifest is incomplete")
    stats = pd.read_csv(source / "policy_stats.csv")
    edge_files = sorted((source / "parts").glob("*__edges.parquet"))
    edges = pd.concat([pd.read_parquet(path) for path in edge_files], ignore_index=True)
    strata = {
        "all_admitted": pd.Series(True, index=edges.index),
        "exclude_numeric_only": ~edges.numeric_only,
        "length_ge_6": edges.source_value_length >= 6,
        "length_ge_10": edges.source_value_length >= 10,
        "length_ge_20": edges.source_value_length >= 20,
        "opaque_identifier": edges.opaque_identifier,
    }
    rows = []
    for stratum, mask in strata.items():
        units = aggregate_filtered_edges(edges[mask])
        for key, group in units.groupby(["corpus", "policy", "max_length"], sort=True):
            corpus, policy, budget = key
            valid = group[group.valid_oracle]
            trained = valid[valid.target_trained]
            unsupported = trained[trained.unsupported_target]
            point, low, high = cluster_interval(valid)
            setting = stats[
                (stats.corpus == corpus) & (stats.policy == policy) & (stats.max_length == budget)
            ].iloc[0]
            unsupported_tokens = int(unsupported.unsupported_target_tokens.sum())
            unsupported_occurrences = int(
                unsupported.unsupported_target_token_occurrences.sum()
            )
            rows.append(
                {
                    "stratum": stratum,
                    "corpus": corpus,
                    "policy": policy,
                    "max_length": int(budget),
                    "oracle_units": len(valid),
                    "trained_units": len(trained),
                    "unsupported_units": len(unsupported),
                    "conditional_prevalence": point,
                    "ci_low": low,
                    "ci_high": high,
                    "affected_conversations": unsupported.row_id.nunique(),
                    "total_conversations": int(setting.conversations),
                    "conversation_incidence": unsupported.row_id.nunique() / setting.conversations,
                    "unsupported_target_tokens": unsupported_tokens,
                    "unsupported_target_token_occurrences": unsupported_occurrences,
                    "effective_supervised_tokens": int(setting.effective_supervised_tokens),
                    "nominal_supervised_tokens": int(setting.nominal_supervised_tokens),
                    "processed_objective_exposure": unsupported_occurrences / setting.effective_supervised_tokens
                    if setting.effective_supervised_tokens
                    else 0.0,
                    "nominal_objective_exposure": unsupported_tokens / setting.nominal_supervised_tokens
                    if setting.nominal_supervised_tokens
                    else 0.0,
                    "target_retention": len(trained) / len(valid) if len(valid) else 0.0,
                    "supervised_token_retention": setting.effective_supervised_tokens / setting.nominal_supervised_tokens
                    if setting.nominal_supervised_tokens
                    else 0.0,
                    "row_supervision_retention": setting.rows_with_supervision / setting.conversations,
                    "unique_token_coverage": setting.unique_retained_tokens / setting.nominal_tokens,
                    "unique_supervision_coverage": setting.unique_effective_supervised_tokens
                    / setting.nominal_supervised_tokens
                    if setting.nominal_supervised_tokens
                    else 0.0,
                    "generated_examples": int(setting.generated_examples),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(destination / "incidence.csv", index=False)
    stats.to_csv(destination / "policy_stats.csv", index=False)

    unique_edges = edges.drop_duplicates("edge_id")[
        [
            "corpus",
            "row_id",
            "edge_id",
            "unit_id",
            "source_value_length",
            "numeric_only",
            "opaque_identifier",
            "value_category",
            "source_target_distance",
            "critical_contiguous_budget",
            "critical_keep_end_budget",
        ]
    ]
    unique_edges.to_parquet(destination / "critical_budget_edges.parquet", index=False)
    distributions = unique_edges.groupby("corpus").agg(
        edges=("edge_id", "size"),
        distance_median=("source_target_distance", "median"),
        distance_p95=("source_target_distance", lambda value: value.quantile(0.95)),
        contiguous_budget_median=("critical_contiguous_budget", "median"),
        contiguous_budget_p95=("critical_contiguous_budget", lambda value: value.quantile(0.95)),
        keep_end_budget_median=("critical_keep_end_budget", "median"),
        keep_end_budget_p95=("critical_keep_end_budget", lambda value: value.quantile(0.95)),
    ).reset_index()
    distributions.to_csv(destination / "distance_and_budget_summary.csv", index=False)

    primary = summary[summary.stratum == "all_admitted"].copy()
    report = [
        "# Phase I contemporary preprocessing audit",
        "",
        "All numerators and denominators refer to exact observation-to-answer relations inherited from the verified V1 oracle.",
        "The conditional percentage is shown alongside corpus-wide conversation and supervised-token exposure.",
        "",
        "| Corpus | Policy | Limit | Unsupported / trained | Conditional | Affected / all conversations | Nominal / processed objective exposure | Processed supervision ratio | Unique supervision coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        report.append(
            f"| {row.corpus} | {row.policy} | {row.max_length:,} | {row.unsupported_units:,}/{row.trained_units:,} | "
            f"{row.conditional_prevalence:.2%} [{row.ci_low:.2%}, {row.ci_high:.2%}] | "
            f"{row.affected_conversations:,}/{row.total_conversations:,} ({row.conversation_incidence:.2%}) | "
            f"{row.nominal_objective_exposure:.4%} / {row.processed_objective_exposure:.4%} | "
            f"{row.supervised_token_retention:.2%} | {row.unique_supervision_coverage:.2%} |"
        )
    (destination / "RESULTS.md").write_text("\n".join(report) + "\n")
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "input_manifest": str((source / "manifest.json").relative_to(ROOT)),
                "edge_part_files": len(edge_files),
                "edge_records": len(edges),
                "summary_rows": len(summary),
                "cluster_bootstraps_per_cell": BOOTSTRAPS,
                "seed": SEED,
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
