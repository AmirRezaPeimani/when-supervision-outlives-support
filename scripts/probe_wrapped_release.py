#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import datasets
import pyarrow
import transformers
import trl
import trl.data_utils as trl_data_utils
from datasets import Dataset
from trl import pack_dataset


ROOT = Path(__file__).resolve().parents[1]


def flatten(dataset, column):
    return [value for row in dataset[column] for value in row]


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    rows = 2_001
    source = Dataset.from_dict(
        {
            "input_ids": [[row, row] for row in range(rows)],
            "labels": [[-100, row] for row in range(rows)],
            "provenance_row": [[row, row] for row in range(rows)],
            "provenance_token": [[0, 1] for _ in range(rows)],
        }
    )
    native = pack_dataset(source, seq_length=17, strategy="wrapped")
    controlled = pack_dataset(
        source,
        seq_length=17,
        strategy="wrapped",
        map_kwargs={"batch_size": len(source)},
    )
    expected = flatten(source, "provenance_row")
    observed = flatten(native, "provenance_row")
    fixed = flatten(controlled, "provenance_row")
    mismatches = [index for index, pair in enumerate(zip(expected, observed, strict=True)) if pair[0] != pair[1]]
    source_code = inspect.getsource(trl_data_utils._pack_wrapped)
    payload = {
        "status": "confirmed_release_conformance_failure",
        "rows": rows,
        "tokens": len(expected),
        "seq_length": 17,
        "first_native_mismatch_index": mismatches[0],
        "native_mismatched_tokens": len(mismatches),
        "native_exact_preservation": observed == expected,
        "single_batch_exact_preservation": fixed == expected,
        "expected_at_first_mismatch": expected[mismatches[0]],
        "native_at_first_mismatch": observed[mismatches[0]],
        "versions": {
            "trl": trl.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "pack_wrapped_source_sha256": digest_text(source_code),
        "paid_compute_cad": 0.0,
    }
    destination = ROOT / "analysis/wrapped_conformance"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
