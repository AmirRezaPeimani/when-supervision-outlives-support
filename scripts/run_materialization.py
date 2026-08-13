#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vsr.data import load_corpora, load_v1_edges
from vsr.materialize import (
    build_target_closures,
    materialize_grouped,
    materialize_last_complete_tool_round,
    materialize_masked_keep_end,
    materialize_single_target,
    materialize_source_target_round,
    target_messages_repaired,
    validate_serialized_materialized_example,
)
from vsr.render import load_tokenizer, render_qwen


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/materialization.json")
    parser.add_argument("--output-dir", default="outputs/materialization")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    destination = ROOT / args.output_dir
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to overwrite nonempty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    v1 = _resolve(config["v1_root"])
    edges_path = v1 / "outputs/raw/audit_support_edges.jsonl"
    edges_by_row = load_v1_edges(edges_path, {config["edge_kind"]})
    tokenizer = load_tokenizer(_resolve(config["tokenizer_path"]))
    example_rows = []
    conversation_rows = []
    validated_examples = 0

    for corpus in config["corpora"]:
        conversations = [
            conversation
            for conversation in load_corpora(v1 / "data/raw", [corpus])
            if conversation.row_id in edges_by_row
        ]
        for position, conversation in enumerate(conversations):
            rendered = render_qwen(conversation, tokenizer)
            edges = edges_by_row[conversation.row_id]
            closures = build_target_closures(conversation, rendered, edges)
            valid = [closure for closure in closures if closure.valid]
            nominal_target_supervision = sum(
                sum(
                    rendered.labels[index] != -100
                    for index in range(
                        rendered.message_spans[closure.target_message].block_start,
                        rendered.message_spans[closure.target_message].block_end,
                    )
                )
                for closure in valid
            )
            for budget in config["max_lengths"]:
                methods = {
                    "masked_keep_end": materialize_masked_keep_end(
                        conversation, rendered, closures, budget
                    ),
                    "last_complete_tool_round": materialize_last_complete_tool_round(
                        conversation, rendered, closures, edges, budget
                    ),
                    "source_target_round": materialize_source_target_round(
                        conversation, rendered, closures, edges, budget
                    ),
                    "single_target_closure": materialize_single_target(
                        conversation, rendered, closures, budget
                    ),
                    "grouped_closure": materialize_grouped(
                        conversation, rendered, closures, budget
                    ),
                }
                for method, examples in methods.items():
                    if method != "masked_keep_end":
                        for example in examples:
                            validate_serialized_materialized_example(
                                conversation, rendered, edges, example
                            )
                            validated_examples += 1
                    repaired = target_messages_repaired(examples)
                    all_indices = [index for example in examples for index in example.original_indices]
                    unique_indices = set(all_indices)
                    source_universe = set().union(*(closure.source_indices for closure in valid)) if valid else set()
                    source_occurrences = sum(index in source_universe for index in all_indices)
                    source_unique = len(source_universe & unique_indices)
                    conversation_rows.append(
                        {
                            "corpus": corpus,
                            "row_id": conversation.row_id,
                            "policy": method,
                            "max_length": budget,
                            "full_tokens": len(rendered.input_ids),
                            "nominal_supervised_tokens": sum(label != -100 for label in rendered.labels),
                            "nominal_target_supervised_tokens": nominal_target_supervision,
                            "valid_targets": len(valid),
                            "retained_targets": len(repaired),
                            "examples": len(examples),
                            "output_tokens": len(all_indices),
                            "unique_output_tokens": len(unique_indices),
                            "output_supervised_tokens": sum(example.supervised_tokens for example in examples),
                            "duplicate_source_tokens": source_occurrences - source_unique,
                            "split": len(examples) > 1,
                        }
                    )
                    for example_index, example in enumerate(examples):
                        example_rows.append(
                            {
                                "corpus": corpus,
                                "row_id": conversation.row_id,
                                "policy": method,
                                "max_length": budget,
                                "example_index": example_index,
                                "target_messages": list(example.target_messages),
                                "original_indices": list(example.original_indices),
                                "labels": list(example.labels),
                                "length": example.length,
                                "supervised_tokens": example.supervised_tokens,
                            }
                        )
            if (position + 1) % 250 == 0:
                print(f"materialized {corpus} {position + 1}/{len(conversations)}", flush=True)

    rows = pd.DataFrame(conversation_rows)
    examples = pd.DataFrame(example_rows)
    rows.to_parquet(destination / "conversation_metrics.parquet", index=False)
    examples.to_parquet(destination / "examples.parquet", index=False)
    summary_rows = []
    for key, group in rows.groupby(["corpus", "policy", "max_length"], sort=True):
        corpus, policy, budget = key
        lengths = examples[
            (examples.corpus == corpus)
            & (examples.policy == policy)
            & (examples.max_length == budget)
        ].length
        nominal_targets = int(group.valid_targets.sum())
        ratios = (
            group.output_supervised_tokens
            / group.nominal_supervised_tokens.replace(0, np.nan)
        ).dropna()
        summary_rows.append(
            {
                "corpus": corpus,
                "policy": policy,
                "max_length": int(budget),
                "conversations": len(group),
                "valid_targets": nominal_targets,
                "retained_targets": int(group.retained_targets.sum()),
                "target_retention": group.retained_targets.sum() / nominal_targets if nominal_targets else 0.0,
                "supervised_token_retention": group.output_supervised_tokens.sum()
                / group.nominal_supervised_tokens.sum(),
                "target_supervision_ratio": group.output_supervised_tokens.sum()
                / group.nominal_target_supervised_tokens.sum()
                if group.nominal_target_supervised_tokens.sum()
                else 0.0,
                "input_token_ratio": group.output_tokens.sum() / group.full_tokens.sum(),
                "unique_input_coverage": group.unique_output_tokens.sum() / group.full_tokens.sum(),
                "examples_per_conversation": group.examples.mean(),
                "dataset_expansion_factor": group.output_tokens.sum() / group.full_tokens.sum(),
                "split_conversations": int(group.split.sum()),
                "duplicate_source_tokens": int(group.duplicate_source_tokens.sum()),
                "weighting_ratio_cv": float(ratios.std(ddof=0) / ratios.mean())
                if len(ratios) and ratios.mean()
                else 0.0,
                "length_mean": float(lengths.mean()) if len(lengths) else 0.0,
                "length_median": float(lengths.median()) if len(lengths) else 0.0,
                "length_p95": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(destination / "summary.csv", index=False)
    report = [
        "# Dataset-level materialization results",
        "",
        "All methods use current Qwen/TRL rendering. Target retention is computed over valid exact-provenance target messages; expansion and weighting costs are explicit.",
        "",
        "| Corpus | Policy | Limit | Target retention | Supervision ratio | Token expansion | Examples/conversation | Split conversations | Duplicate source tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        report.append(
            f"| {row.corpus} | {row.policy} | {row.max_length:,} | {row.retained_targets:,}/{row.valid_targets:,} ({row.target_retention:.1%}) | "
            f"{row.supervised_token_retention:.1%} | {row.dataset_expansion_factor:.2f}x | {row.examples_per_conversation:.2f} | "
            f"{row.split_conversations:,} | {row.duplicate_source_tokens:,} |"
        )
    (destination / "RESULTS.md").write_text("\n".join(report) + "\n")
    manifest = {
        "status": "complete",
        "config": args.config,
        "config_sha256": _digest(config_path),
        "v1_edges_sha256": _digest(edges_path),
        "conversation_rows": len(rows),
        "materialized_examples": len(examples),
        "serialized_examples_validated": validated_examples,
        "serialized_validation_relation": "audit_edges_for_view/causal_attention_relation",
        "paid_compute_cad": 0.0,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
