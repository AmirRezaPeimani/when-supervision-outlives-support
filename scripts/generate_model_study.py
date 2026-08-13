#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import string
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FAMILIES = {
    "order": ("lookup_order", "order_id", "confirmation", "ORD"),
    "error": ("diagnose_error", "trace_id", "recovery_code", "ERR"),
    "inventory": ("check_inventory", "item_id", "warehouse_sku", "SKU"),
    "route": ("resolve_route", "route_id", "route_token", "RTE"),
    "ticket": ("fetch_ticket", "ticket_id", "case_reference", "TKT"),
    "checksum": ("verify_artifact", "artifact_id", "checksum", "HEX"),
}
ALPHABET = string.ascii_uppercase + string.digits


def value(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-" + "".join(rng.choice(ALPHABET) for _ in range(8))


def build_record(rng: random.Random, family: str, split: str, index: int, conflict: bool) -> dict:
    tool_name, key_field, value_field, prefix = FAMILIES[family]
    key = f"{split.upper()}-{family[:3].upper()}-{index:04d}"
    correct = value(rng, prefix)
    stale = value(rng, prefix)
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Return the authoritative {value_field} for a supplied {key_field}.",
                "parameters": {
                    "type": "object",
                    "properties": {key_field: {"type": "string"}},
                    "required": [key_field],
                },
            },
        }
    ]
    stale_text = (
        f" A stale cache says VALUE={stale}; it may be wrong and must not override the tool."
        if conflict
        else ""
    )
    messages = [
        {
            "role": "system",
            "content": "Use the tool result as the sole authority. Reply with exactly VALUE=<value> and no other text.",
        },
        {
            "role": "user",
            "content": f"Find the current {value_field} for {key_field} {key}.{stale_text}",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": tool_name, "arguments": {key_field: key}},
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps({key_field: key, value_field: correct}, separators=(",", ":")),
        },
        {"role": "assistant", "content": f"VALUE={correct}"},
    ]
    return {
        "record_id": f"{split}:{family}:{index:04d}",
        "split": split,
        "family": family,
        "key": key,
        "target": correct,
        "stale_value": stale if conflict else None,
        "conflict": conflict,
        "tools": tools,
        "messages": messages,
    }


def with_corruption(record: dict, corrupted: bool) -> dict:
    output = json.loads(json.dumps(record))
    output["corrupted"] = corrupted
    if corrupted:
        output["messages"][-2]["content"] = json.dumps(
            {"status": "unavailable", "query_id": record["key"]}, separators=(",", ":")
        )
    return output


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config_path = ROOT / "configs/model_study.json"
    config = json.loads(config_path.read_text())
    destination = ROOT / "data/model_study"
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to overwrite nonempty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config["seed"])
    splits = {}
    for split, count, conflict in (
        ("train", config["train_per_family"], True),
        ("dev", config["dev_per_family"], True),
        ("test", config["test_per_family"], False),
        ("conflict", config["conflict_per_family"], True),
    ):
        splits[split] = [
            build_record(rng, family, split, index, conflict)
            for family in config["families"]
            for index in range(count)
        ]
    train = splits.pop("train")
    ordered = sorted(
        train,
        key=lambda record: hashlib.sha256(
            f"{config['seed']}:{record['record_id']}".encode()
        ).hexdigest(),
    )
    for level in config["corruption_levels"]:
        corrupted_ids = {
            record["record_id"] for record in ordered[: round(len(ordered) * level / 100)]
        }
        write_jsonl(
            destination / f"train_corruption_{level}.jsonl",
            [with_corruption(record, record["record_id"] in corrupted_ids) for record in train],
        )
    for split, records in splits.items():
        write_jsonl(destination / f"{split}.jsonl", [with_corruption(record, False) for record in records])
    manifest = {
        "status": "generated_and_sealed",
        "config_sha256": digest(config_path),
        "counts": {
            **{f"train_corruption_{level}": len(train) for level in config["corruption_levels"]},
            **{split: len(records) for split, records in splits.items()},
        },
        "hashes": {path.name: digest(path) for path in sorted(destination.glob("*.jsonl"))},
        "split_rule": "disjoint key namespaces and independently generated opaque values",
        "sealed_before_gate": ["test.jsonl", "conflict.jsonl"],
        "paid_compute_cad": 0.0,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
