from datasets import Dataset
from trl import pack_dataset


def _flatten(dataset, column):
    return [value for row in dataset[column] for value in row]


def test_wrapped_default_map_repeats_earlier_storage_across_batches():
    """Pin the released 1.9.2 behavior so it is reported, not normalized away."""
    rows = 2_001
    source = Dataset.from_dict(
        {
            "input_ids": [[row, row] for row in range(rows)],
            "labels": [[-100, row] for row in range(rows)],
            "provenance_row": [[row, row] for row in range(rows)],
            "provenance_token": [[0, 1] for _ in range(rows)],
        }
    )
    packed = pack_dataset(source, seq_length=17, strategy="wrapped")
    packed_rows = _flatten(packed, "provenance_row")
    source_rows = _flatten(source, "provenance_row")
    assert packed_rows != source_rows
    assert packed_rows[:2_000] == source_rows[:2_000]
    assert packed_rows[2_000] == 0
    assert source_rows[2_000] == 1_000


def test_wrapped_single_batch_preserves_all_columns():
    rows = 2_001
    source = Dataset.from_dict(
        {
            "input_ids": [[row, row] for row in range(rows)],
            "labels": [[-100, row] for row in range(rows)],
            "provenance_row": [[row, row] for row in range(rows)],
            "provenance_token": [[0, 1] for _ in range(rows)],
        }
    )
    packed = pack_dataset(
        source,
        seq_length=17,
        strategy="wrapped",
        map_kwargs={"batch_size": len(source)},
    )
    for column in source.column_names:
        assert _flatten(packed, column) == _flatten(source, column)
