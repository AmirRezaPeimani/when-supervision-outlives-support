#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    record = json.loads((ROOT / "provenance/model.json").read_text())
    destination = ROOT / record["local_path"]
    snapshot_download(
        repo_id=record["model"],
        revision=record["revision"],
        local_dir=destination,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )
    observed = sha256(destination / "tokenizer.json")
    if observed != record["tokenizer_sha256"]:
        raise RuntimeError(f"tokenizer hash mismatch: {observed}")
    print(f"verified tokenizer: sha256={observed}")


if __name__ == "__main__":
    main()
