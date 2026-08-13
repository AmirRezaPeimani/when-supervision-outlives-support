#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import pandas as pd
import transformers
import trl

from vsr.audit import aggregate_units, audit_edges_for_view
from vsr.data import load_glaive, load_v1_edges
from vsr.policies import materialize_nonpacked, materialize_packed
from vsr.render import load_tokenizer, render_qwen
from vsr.support import discover_observation_edges


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/glaive.json")
    parser.add_argument("--output-dir", default="outputs/raw/glaive_audit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    output = ROOT / args.output_dir
    if output.exists() and any(output.rglob("*")) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    started = time.time()

    selected_path = resolve(config["selected_path"])
    edge_path = resolve(config["edge_path"])
    tokenizer = load_tokenizer(resolve(config["tokenizer_path"]))
    conversations = load_glaive(selected_path)
    case_sensitive = bool(config.get("oracle_case_sensitive", False))
    if case_sensitive:
        edges = {
            conversation.row_id: discovered
            for conversation in conversations
            if (discovered := discover_observation_edges(conversation, case_sensitive=True))
        }
    else:
        edges = load_v1_edges(edge_path, {"tool_to_assistant"})
    rendered = []
    for index, conversation in enumerate(conversations):
        rendered.append(render_qwen(conversation, tokenizer))
        if (index + 1) % 1000 == 0:
            print(f"rendered glaive {index + 1}/{len(conversations)}", flush=True)

    nominal_supervised = sum(sum(label != -100 for label in row.labels) for row in rendered)
    nominal_tokens = sum(len(row.input_ids) for row in rendered)
    policies = config["policies"]
    budgets = config["max_lengths"]
    settings = []
    total_edge_records = 0
    total_unit_records = 0
    for budget in budgets:
        for policy in policies:
            setting_started = time.time()
            if policy in {"keep_start", "keep_end", "drop_overlength"}:
                view = materialize_nonpacked(rendered, budget, policy)
            else:
                view = materialize_packed(rendered, budget, policy)
            edge_records = audit_edges_for_view(
                conversations,
                rendered,
                edges,
                view,
                case_sensitive=case_sensitive,
            )
            unit_records = aggregate_units(edge_records)
            stem = f"corpus=glaive__policy={policy}__length={budget}"
            pd.DataFrame(edge_records).to_parquet(parts / f"{stem}__edges.parquet", index=False)
            pd.DataFrame(unit_records).to_parquet(parts / f"{stem}__units.parquet", index=False)
            total_edge_records += len(edge_records)
            total_unit_records += len(unit_records)
            lengths = pd.Series(view.packed_lengths, dtype=float)
            settings.append(
                {
                    "corpus": "glaive",
                    "policy": policy,
                    "max_length": budget,
                    "conversations": len(conversations),
                    "edge_bearing_conversations": len(edges),
                    "nominal_tokens": nominal_tokens,
                    "nominal_supervised_tokens": nominal_supervised,
                    "retained_tokens": view.retained_original_tokens,
                    "unique_retained_tokens": view.unique_retained_original_tokens,
                    "effective_supervised_tokens": view.effective_supervised_tokens,
                    "unique_effective_supervised_tokens": view.unique_effective_supervised_tokens,
                    "generated_examples": view.generated_examples,
                    "attention_groups": view.attention_groups,
                    "rows_with_tokens": len(view.rows_with_tokens),
                    "rows_with_supervision": len(view.rows_with_supervision),
                    "mean_sequence_length": float(lengths.mean()) if len(lengths) else 0.0,
                    "median_sequence_length": float(lengths.median()) if len(lengths) else 0.0,
                    "p95_sequence_length": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
                    "seconds": round(time.time() - setting_started, 3),
                }
            )
            print(
                f"completed glaive/{policy}/{budget}: {len(edge_records)} edges, "
                f"{view.generated_examples} examples in {time.time() - setting_started:.1f}s",
                flush=True,
            )
            del view, edge_records, unit_records
            gc.collect()

    pd.DataFrame(settings).to_csv(output / "policy_stats.csv", index=False)
    manifest = {
        "status": "complete",
        "config": args.config,
        "config_sha256": digest(config_path),
        "selected_data_sha256": digest(selected_path),
        "support_edges_sha256": digest(edge_path),
        "oracle_case_sensitive": case_sensitive,
        "oracle_source": "rediscovered from frozen conversations" if case_sensitive else str(edge_path),
        "settings": len(settings),
        "edge_records": total_edge_records,
        "unit_records": total_unit_records,
        "seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "paid_compute_cad": 0.0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
