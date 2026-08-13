#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.request
from pathlib import Path

import certifi


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=["toolace", "retool", "glaive"],
        default=["toolace", "retool", "glaive"],
    )
    args = parser.parse_args()
    records = json.loads((ROOT / "provenance/datasets.json").read_text())
    context = ssl.create_default_context(cafile=certifi.where())
    for name in args.datasets:
        record = records[name]
        target = ROOT / record["local_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"downloading {record['dataset']} -> {target.relative_to(ROOT)}")
            with urllib.request.urlopen(record["parquet_url"], context=context) as source:
                with target.open("wb") as sink:
                    while chunk := source.read(1024 * 1024):
                        sink.write(chunk)
        observed = sha256(target)
        if observed != record["sha256"]:
            raise RuntimeError(f"hash mismatch for {name}: {observed}")
        print(f"verified {name}: {record['rows']} rows, sha256={observed}")


if __name__ == "__main__":
    main()
