import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def records(name: str) -> list[dict]:
    path = ROOT / f"data/model_study/{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_train_dev_test_and_conflict_identifiers_and_values_are_disjoint():
    splits = {
        "train": records("train_corruption_0"),
        "dev": records("dev"),
        "test": records("test"),
        "conflict": records("conflict"),
    }
    for left_name, left in splits.items():
        for right_name, right in splits.items():
            if left_name >= right_name:
                continue
            assert {row["key"] for row in left}.isdisjoint(row["key"] for row in right)
            assert {row["target"] for row in left}.isdisjoint(row["target"] for row in right)
            assert {row["record_id"] for row in left}.isdisjoint(row["record_id"] for row in right)


def test_stale_context_is_present_in_train_dev_and_conflict_but_not_clean_test():
    for split in ["train_corruption_0", "dev", "conflict"]:
        data = records(split)
        assert all(row["conflict"] and row["stale_value"] for row in data)
        assert all(row["target"] != row["stale_value"] for row in data)
        assert all(row["stale_value"] in row["messages"][1]["content"] for row in data)
    clean_test = records("test")
    assert all(not row["conflict"] and row["stale_value"] is None for row in clean_test)


def test_training_corruption_changes_support_but_not_target_or_prompt():
    supported = {row["record_id"]: row for row in records("train_corruption_0")}
    corrupted = {row["record_id"]: row for row in records("train_corruption_100")}
    assert supported.keys() == corrupted.keys()
    for record_id in supported:
        left, right = supported[record_id], corrupted[record_id]
        assert left["target"] == right["target"]
        assert left["messages"][-1] == right["messages"][-1]
        assert left["messages"][:-2] == right["messages"][:-2]
        assert left["messages"][-2]["role"] == right["messages"][-2]["role"] == "tool"
        assert left["target"] in left["messages"][-2]["content"]
        assert left["target"] not in right["messages"][-2]["content"]


def test_corruption_levels_have_frozen_exact_counts():
    expected = {0: 0, 25: 450, 50: 900, 100: 1800}
    expected_by_family = {
        0: {family: 0 for family in ["checksum", "error", "inventory", "order", "route", "ticket"]},
        25: {"checksum": 76, "error": 72, "inventory": 84, "order": 73, "route": 68, "ticket": 77},
        50: {"checksum": 155, "error": 147, "inventory": 167, "order": 147, "route": 140, "ticket": 144},
        100: {family: 300 for family in ["checksum", "error", "inventory", "order", "route", "ticket"]},
    }
    corrupted_ids = {}
    for level, count in expected.items():
        data = records(f"train_corruption_{level}")
        assert len(data) == 1800
        assert sum(bool(row["corrupted"]) for row in data) == count
        corrupted_ids[level] = {row["record_id"] for row in data if row["corrupted"]}
        family_counts = {}
        for row in data:
            family_counts[row["family"]] = family_counts.get(row["family"], 0) + int(
                row["corrupted"]
            )
        for family in expected_by_family[level]:
            family_counts.setdefault(family, 0)
        assert family_counts == expected_by_family[level]
    assert corrupted_ids[0] <= corrupted_ids[25] <= corrupted_ids[50] <= corrupted_ids[100]
