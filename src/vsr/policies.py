from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from datasets import Dataset
from trl import pack_dataset

from .types import PolicyView, RenderedConversation, RetainedInterval


def _compress_intervals(
    row_ids: list[int],
    token_ids: list[int],
    group_id: int,
    packed_example: int,
) -> list[RetainedInterval]:
    if not row_ids:
        return []
    output = []
    run_row = row_ids[0]
    run_start = token_ids[0]
    previous = token_ids[0]
    run_position = 0
    for position, (row_index, token_index) in enumerate(
        zip(row_ids[1:], token_ids[1:], strict=True), start=1
    ):
        if row_index == run_row and token_index == previous + 1:
            previous = token_index
            continue
        output.append(
            RetainedInterval(
                run_row,
                run_start,
                previous + 1,
                group_id,
                packed_example,
                run_position,
                position,
            )
        )
        run_row, run_start, previous = row_index, token_index, token_index
        run_position = position
    output.append(
        RetainedInterval(
            run_row,
            run_start,
            previous + 1,
            group_id,
            packed_example,
            run_position,
            len(row_ids),
        )
    )
    return output


def _view_from_groups(
    policy: str,
    budget: int,
    groups: list[tuple[int, list[int], list[int], list[int]]],
    generated_examples: int,
    packed_lengths: Iterable[int],
) -> PolicyView:
    """Build a view from `(packed_example, rows, original_indices, labels)` attention groups."""
    row_intervals: dict[int, list[RetainedInterval]] = defaultdict(list)
    first_tokens: set[tuple[int, int, int]] = set()
    effective_by_row_group: dict[tuple[int, int], set[int]] = defaultdict(set)
    effective_positions_by_group: dict[int, set[int]] = defaultdict(set)
    retained_pairs: set[tuple[int, int]] = set()
    supervised_pairs: set[tuple[int, int]] = set()
    rows_with_tokens: set[int] = set()
    rows_with_supervision: set[int] = set()
    supervised = 0
    retained = 0
    for group_id, (packed_example, rows, indices, labels) in enumerate(groups):
        if not rows:
            continue
        first_tokens.add((group_id, rows[0], indices[0]))
        retained += len(rows)
        for position, (row_index, label) in enumerate(zip(rows, labels, strict=True)):
            rows_with_tokens.add(row_index)
            retained_pairs.add((row_index, indices[position]))
            if position > 0 and label != -100:
                supervised += 1
                rows_with_supervision.add(row_index)
                effective_by_row_group[(row_index, group_id)].add(indices[position])
                effective_positions_by_group[group_id].add(position)
                supervised_pairs.add((row_index, indices[position]))
        for interval in _compress_intervals(rows, indices, group_id, packed_example):
            row_intervals[interval.row_index].append(interval)
    for intervals in row_intervals.values():
        intervals.sort(key=lambda item: (item.start, item.end, item.group_id))
    return PolicyView(
        policy=policy,
        budget=budget,
        row_intervals=dict(row_intervals),
        group_first_tokens=first_tokens,
        effective_indices_by_row_group={
            key: frozenset(value) for key, value in effective_by_row_group.items()
        },
        effective_label_positions_by_group={
            key: frozenset(value) for key, value in effective_positions_by_group.items()
        },
        retained_original_tokens=retained,
        unique_retained_original_tokens=len(retained_pairs),
        effective_supervised_tokens=supervised,
        unique_effective_supervised_tokens=len(supervised_pairs),
        generated_examples=generated_examples,
        attention_groups=len(groups),
        rows_with_tokens=rows_with_tokens,
        rows_with_supervision=rows_with_supervision,
        packed_lengths=tuple(int(value) for value in packed_lengths),
    )


def policy_view_from_groups(
    policy: str,
    budget: int,
    groups: list[tuple[int, list[int], list[int], list[int]]],
    generated_examples: int,
    packed_lengths: Iterable[int],
) -> PolicyView:
    """Public constructor for provenance-traced attention groups."""
    return _view_from_groups(policy, budget, groups, generated_examples, packed_lengths)


def materialize_nonpacked(
    rendered_rows: list[RenderedConversation], budget: int, policy: str
) -> PolicyView:
    groups = []
    for row_index, rendered in enumerate(rendered_rows):
        length = len(rendered.input_ids)
        if policy == "drop_overlength" and length > budget:
            continue
        if policy in {"keep_start", "drop_overlength"}:
            start, end = 0, min(length, budget)
        elif policy == "keep_end":
            start, end = max(0, length - budget), length
        else:
            raise ValueError(policy)
        indices = list(range(start, end))
        labels = list(rendered.labels[start:end])
        # Current non-packed TRL drops examples whose prepared labels are fully masked.
        if policy in {"keep_start", "keep_end"} and not any(label != -100 for label in labels):
            continue
        groups.append((len(groups), [row_index] * len(indices), indices, labels))
    return _view_from_groups(policy, budget, groups, len(groups), [len(item[1]) for item in groups])


def materialize_packed(
    rendered_rows: list[RenderedConversation], budget: int, strategy: str
) -> PolicyView:
    if strategy not in {"bfd", "bfd_split", "wrapped", "wrapped_single_batch"}:
        raise ValueError(strategy)
    native_strategy = "wrapped" if strategy == "wrapped_single_batch" else strategy
    dataset = Dataset.from_dict(
        {
            "input_ids": [list(row.input_ids) for row in rendered_rows],
            "labels": [list(row.labels) for row in rendered_rows],
            "provenance_row": [[row_index] * len(row.input_ids) for row_index, row in enumerate(rendered_rows)],
            "provenance_token": [list(range(len(row.input_ids))) for row in rendered_rows],
        }
    )
    map_kwargs = {"batch_size": len(dataset)} if strategy == "wrapped_single_batch" else None
    packed = pack_dataset(dataset, seq_length=budget, strategy=native_strategy, map_kwargs=map_kwargs)
    groups = []
    packed_lengths = []
    for packed_index, example in enumerate(packed):
        rows = list(example["provenance_row"])
        indices = list(example["provenance_token"])
        labels = list(example["labels"])
        packed_lengths.append(len(rows))
        if native_strategy in {"bfd", "bfd_split"}:
            offset = 0
            for length in example["seq_lengths"]:
                length = int(length)
                groups.append(
                    (
                        packed_index,
                        rows[offset : offset + length],
                        indices[offset : offset + length],
                        labels[offset : offset + length],
                    )
                )
                offset += length
            if offset != len(rows):
                raise AssertionError("packed sequence lengths do not cover the output row")
        else:
            groups.append((packed_index, rows, indices, labels))
    return _view_from_groups(strategy, budget, groups, len(packed), packed_lengths)


def groups_covering_span(view: PolicyView, row_index: int, span: tuple[int, int]) -> set[int]:
    start, end = span
    return {
        interval.group_id
        for interval in view.row_intervals.get(row_index, [])
        if interval.start <= start and interval.end >= end
    }


def span_occurrences(
    view: PolicyView, row_index: int, span: tuple[int, int]
) -> list[tuple[int, int, int]]:
    """Return `(group, start, end)` for complete, provenance-matched span copies."""
    start, end = span
    occurrences = []
    for interval in view.row_intervals.get(row_index, []):
        if interval.start <= start and interval.end >= end:
            position_start = interval.group_position_start + start - interval.start
            occurrences.append(
                (interval.group_id, position_start, position_start + end - start)
            )
    return occurrences


def causal_attention_relation(
    target: tuple[int, int], source: tuple[int, int]
) -> bool:
    """Causal-decoder attention: same attention group and source strictly earlier."""
    target_group, target_position = target
    source_group, source_position = source
    return target_group == source_group and source_position < target_position


def span_has_effective_label(
    view: PolicyView,
    row_index: int,
    span: tuple[int, int],
    rendered: RenderedConversation,
    allowed_groups: set[int] | None = None,
) -> bool:
    groups = groups_covering_span(view, row_index, span)
    if allowed_groups is not None:
        groups &= allowed_groups
    if not groups:
        return False
    start, end = span
    return any(
        any(start <= index < end for index in view.effective_indices_by_row_group.get((row_index, group_id), ()))
        for group_id in groups
    )


def effective_indices_for_span(
    view: PolicyView,
    row_index: int,
    span: tuple[int, int],
    allowed_groups: set[int],
) -> set[int]:
    start, end = span
    return {
        index
        for group_id in allowed_groups
        for index in view.effective_indices_by_row_group.get((row_index, group_id), ())
        if start <= index < end
    }


def effective_positions_for_span(
    view: PolicyView,
    row_index: int,
    span: tuple[int, int],
    allowed_groups: set[int],
) -> set[tuple[int, int]]:
    return {
        (group_id, position)
        for group_id, occurrence_start, occurrence_end in span_occurrences(view, row_index, span)
        if group_id in allowed_groups
        for position in view.effective_label_positions_by_group.get(group_id, ())
        if occurrence_start <= position < occurrence_end
    }
