#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from vsr.data import load_corpora, load_v1_edges
from vsr.render import load_tokenizer
from vsr.schema_repair import repair_action_target


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/action_repair.json")
    parser.add_argument("--output-dir", default="outputs/schema_repair")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    destination = ROOT / args.output_dir
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to overwrite nonempty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    v1 = _resolve(config["v1_root"])
    edges_path = v1 / "outputs/raw/audit_support_edges.jsonl"
    edges_by_row = load_v1_edges(edges_path, {"user_to_tool_call"})
    tokenizer = load_tokenizer(_resolve(config["tokenizer_path"]))
    rows = []
    for corpus in config["corpora"]:
        conversations = [
            conversation
            for conversation in load_corpora(v1 / "data/raw", [corpus])
            if conversation.row_id in edges_by_row
        ]
        for position, conversation in enumerate(conversations):
            edges = edges_by_row[conversation.row_id]
            targets = sorted({edge.target_message for edge in edges})
            for target in targets:
                for mode in config["modes"]:
                    result = repair_action_target(
                        conversation, tokenizer, edges, target, 10_000_000, mode
                    )
                    for budget in config["max_lengths"]:
                        success = result.success and result.length <= budget
                        reason = result.reason if not result.success else (
                            "" if success else "closure exceeds budget"
                        )
                        rows.append(
                            {
                                "corpus": corpus,
                                "row_id": result.row_id,
                                "target_message": result.target_message,
                                "mode": result.mode,
                                "max_length": budget,
                                "success": success,
                                "reason": reason,
                                "length": result.length,
                                "schema_tokens": result.schema_tokens,
                                "supervised_tokens": result.supervised_tokens if success else 0,
                                "invoked_tools": list(result.invoked_tools),
                                "original_indices": list(result.original_indices) if success else [],
                                "labels": list(result.labels) if success else [],
                            }
                        )
            if (position + 1) % 250 == 0:
                print(f"action repair {corpus} {position + 1}/{len(conversations)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(destination / "target_results.parquet", index=False)
    summary = (
        frame.groupby(["corpus", "mode", "max_length"], as_index=False)
        .agg(
            targets=("success", "size"),
            repaired=("success", "sum"),
            mean_length=("length", lambda values: values[frame.loc[values.index, "success"]].mean()),
            median_length=("length", lambda values: values[frame.loc[values.index, "success"]].median()),
            p95_length=("length", lambda values: values[frame.loc[values.index, "success"]].quantile(0.95)),
            mean_schema_tokens=("schema_tokens", lambda values: values[frame.loc[values.index, "success"]].mean()),
            supervised_tokens=("supervised_tokens", "sum"),
        )
    )
    summary["repair_rate"] = summary.repaired / summary.targets
    summary.to_csv(destination / "summary.csv", index=False)
    reasons = (
        frame[~frame.success]
        .groupby(["corpus", "mode", "max_length", "reason"], as_index=False)
        .size()
        .rename(columns={"size": "failures"})
    )
    reasons.to_csv(destination / "failure_reasons.csv", index=False)
    report = [
        "# Schema-aware action repair",
        "",
        "`whole_schema` is the V1 baseline; `invoked_tool` and `parameter_slice` are the two pre-bounded repair iterations.",
        "",
        "| Corpus | Mode | Limit | Repaired | Rate | Mean length | Mean schema tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        report.append(
            f"| {row.corpus} | {row.mode} | {row.max_length:,} | {int(row.repaired):,}/{int(row.targets):,} | "
            f"{row.repair_rate:.1%} | {row.mean_length:.1f} | {row.mean_schema_tokens:.1f} |"
        )
    (destination / "RESULTS.md").write_text("\n".join(report) + "\n")
    manifest = {
        "status": "complete",
        "config": args.config,
        "config_sha256": _digest(config_path),
        "v1_edges_sha256": _digest(edges_path),
        "target_policy_rows": len(frame),
        "paid_compute_cad": 0.0,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
