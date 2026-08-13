#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from vsr.data import load_glaive
from vsr.support import discover_observation_edges


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config_path = ROOT / "configs/glaive.json"
    config = json.loads(config_path.read_text())
    raw_path = ROOT / config["raw_parquet"]
    if digest(raw_path) != config["raw_sha256"]:
        raise RuntimeError("Glaive raw file hash does not match frozen configuration")
    conversations = load_glaive(raw_path)
    edge_map = {}
    for index, conversation in enumerate(conversations):
        edges = discover_observation_edges(conversation)
        if edges:
            edge_map[conversation.row_id] = edges
        if (index + 1) % 10_000 == 0:
            print(f"screened {index + 1}/{len(conversations)}", flush=True)
    ranked = sorted(
        edge_map,
        key=lambda row_id: hashlib.sha256(
            f"{config['selection_seed']}:{row_id}".encode()
        ).hexdigest(),
    )
    selected_ids = ranked[: config["eligible_sample_size"]]
    selected_indices = [int(row_id.split(":", 1)[1]) for row_id in selected_ids]
    selected_frame = pd.read_parquet(raw_path).iloc[selected_indices].copy()
    selected_frame["original_row_index"] = selected_indices
    processed = ROOT / "data/processed"
    raw_output = ROOT / "outputs/raw"
    processed.mkdir(parents=True, exist_ok=True)
    raw_output.mkdir(parents=True, exist_ok=True)
    selected_frame.to_parquet(processed / "glaive_selected.parquet", index=False)
    with (raw_output / "glaive_support_edges.jsonl").open("w") as handle:
        for row_id in selected_ids:
            for edge in edge_map[row_id]:
                handle.write(json.dumps(edge.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "complete",
        "source_rows": len(conversations),
        "eligible_rows": len(edge_map),
        "selected_rows": len(selected_ids),
        "selected_edges": sum(len(edge_map[row_id]) for row_id in selected_ids),
        "raw_sha256": digest(raw_path),
        "selected_sha256": digest(processed / "glaive_selected.parquet"),
        "edges_sha256": digest(raw_output / "glaive_support_edges.jsonl"),
        "paid_compute_cad": 0.0,
    }
    (processed / "glaive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
