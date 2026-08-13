from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .audit import audit_edges_for_view
from .policies import policy_view_from_groups
from .render import locate_value
from .types import Conversation, RenderedConversation, SupportEdge


@dataclass(frozen=True)
class TargetClosure:
    row_id: str
    target_message: int
    indices: frozenset[int]
    source_indices: frozenset[int]
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class MaterializedExample:
    row_id: str
    target_messages: tuple[int, ...]
    original_indices: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.original_indices)

    @property
    def supervised_tokens(self) -> int:
        return sum(label != -100 for label in self.labels)


def validate_serialized_materialized_example(
    conversation: Conversation,
    rendered: RenderedConversation,
    edges: Iterable[SupportEdge],
    example: MaterializedExample,
) -> None:
    """Validate final example arrays with the corpus audit's reachability semantics.

    One serialized example is one causal attention group. Original token indices are exact row
    provenance, and their serialized order is their position in that group. The shared corpus
    auditor therefore checks exact source/target message provenance, complete occurrences, and
    `causal_attention_relation` for every supervised target-value position and source position.
    """
    if example.row_id != conversation.row_id:
        raise ValueError("materialized row provenance does not match the conversation")
    if len(example.original_indices) != len(example.labels):
        raise ValueError("materialized indices and labels have different lengths")
    if not example.original_indices:
        raise ValueError("materialized example is empty")
    if len(set(example.original_indices)) != len(example.original_indices):
        raise ValueError("materialized example repeats original-token provenance")
    if any(index < 0 or index >= len(rendered.input_ids) for index in example.original_indices):
        raise ValueError("materialized example contains out-of-range provenance")

    target_messages = set(example.target_messages)
    if not target_messages or len(target_messages) != len(example.target_messages):
        raise ValueError("materialized target-message identifiers are empty or duplicated")
    if any(target < 0 or target >= len(rendered.message_spans) for target in target_messages):
        raise ValueError("materialized example names an out-of-range target message")

    retained = set(example.original_indices)
    target_blocks = {
        index
        for target in target_messages
        for index in range(
            rendered.message_spans[target].block_start,
            rendered.message_spans[target].block_end,
        )
    }
    for target in target_messages:
        span = rendered.message_spans[target]
        if not set(range(span.block_start, span.block_end)) <= retained:
            raise ValueError(f"materialized target message {target} is incomplete")
    for original_index, label in zip(example.original_indices, example.labels, strict=True):
        expected = rendered.labels[original_index] if original_index in target_blocks else -100
        if label != expected:
            raise ValueError("materialized labels do not match the selected complete targets")

    relevant_edges = [edge for edge in edges if edge.target_message in target_messages]
    targets_with_edges = {edge.target_message for edge in relevant_edges}
    missing_edges = target_messages - targets_with_edges
    if missing_edges:
        raise ValueError(f"materialized targets lack admitted support edges: {sorted(missing_edges)}")

    view = policy_view_from_groups(
        "serialized_materialization",
        len(example.original_indices),
        [
            (
                0,
                [0] * len(example.original_indices),
                list(example.original_indices),
                list(example.labels),
            )
        ],
        generated_examples=1,
        packed_lengths=[len(example.original_indices)],
    )
    records = audit_edges_for_view(
        [conversation],
        [rendered],
        {conversation.row_id: relevant_edges},
        view,
    )
    by_edge = {record["edge_id"]: record for record in records}
    failures = [
        edge.edge_id
        for edge in relevant_edges
        if edge.edge_id not in by_edge
        or not by_edge[edge.edge_id]["target_trained"]
        or not by_edge[edge.edge_id]["source_reachable"]
    ]
    if failures:
        raise ValueError(
            "materialized targets lack a complete provenance-matched source causally visible "
            f"to every supervised target-value token; failing edges: {failures}"
        )


def _wrapper_indices(rendered: RenderedConversation, message_index: int) -> set[int]:
    span = rendered.message_spans[message_index]
    return set(range(span.block_start, span.content_start)) | set(
        range(span.content_end, span.block_end)
    )


def build_target_closures(
    conversation: Conversation,
    rendered: RenderedConversation,
    edges: Iterable[SupportEdge],
) -> list[TargetClosure]:
    """Build one verified closure per supervised target message.

    Every admitted edge to a target contributes its recorded source-message occurrence. This is
    deliberately stricter than merely finding the same value somewhere earlier in the trajectory.
    """
    by_target: dict[int, list[SupportEdge]] = {}
    for edge in edges:
        by_target.setdefault(edge.target_message, []).append(edge)
    closures = []
    for target_message, target_edges in sorted(by_target.items()):
        target_span = rendered.message_spans[target_message]
        selected = set(range(target_span.block_start, target_span.block_end))
        source_indices = set()
        valid = True
        reason = ""
        for edge in target_edges:
            source_message = conversation.messages[edge.source_message]
            hits = locate_value(
                rendered, edge.source_message, edge.source_value, source_message.content
            )
            if not hits:
                valid = False
                reason = f"source value not located in recorded message {edge.source_message}"
                break
            # The oracle edge names a message. Repeated occurrences inside it are redundant; the
            # final occurrence is closest to the downstream target and is selected deterministically.
            hit = max(hits, key=lambda item: item[0])
            hit_indices = set(range(*hit))
            source_indices |= hit_indices
            selected |= hit_indices
            selected |= _wrapper_indices(rendered, edge.source_message)
        closures.append(
            TargetClosure(
                conversation.row_id,
                target_message,
                frozenset(selected),
                frozenset(source_indices),
                valid,
                reason,
            )
        )
    return closures


def _fill_complete_blocks(
    selected: set[int],
    rendered: RenderedConversation,
    budget: int,
    latest_target: int,
) -> set[int]:
    for span in reversed(rendered.message_spans[: latest_target + 1]):
        block = set(range(span.block_start, span.block_end))
        if block <= selected:
            continue
        if len(selected | block) <= budget:
            selected |= block
    return selected


def _make_example(
    conversation: Conversation,
    rendered: RenderedConversation,
    closures: list[TargetClosure],
    budget: int,
    fill: bool = True,
) -> MaterializedExample | None:
    selected = set().union(*(closure.indices for closure in closures))
    if not closures or len(selected) > budget:
        return None
    targets = tuple(sorted({closure.target_message for closure in closures}))
    if fill:
        selected = _fill_complete_blocks(selected, rendered, budget, max(targets))
    indices = tuple(sorted(selected))
    target_indices = set()
    for target in targets:
        span = rendered.message_spans[target]
        target_indices |= set(range(span.block_start, span.block_end))
    labels = tuple(
        rendered.labels[index] if index in target_indices else -100 for index in indices
    )
    if not any(label != -100 for label in labels):
        return None
    return MaterializedExample(conversation.row_id, targets, indices, labels)


def materialize_single_target(
    conversation: Conversation,
    rendered: RenderedConversation,
    closures: list[TargetClosure],
    budget: int,
) -> list[MaterializedExample]:
    output = []
    for closure in closures:
        if not closure.valid:
            continue
        example = _make_example(conversation, rendered, [closure], budget)
        if example is not None:
            output.append(example)
    return output


def materialize_grouped(
    conversation: Conversation,
    rendered: RenderedConversation,
    closures: list[TargetClosure],
    budget: int,
) -> list[MaterializedExample]:
    """Deterministic overlap-aware greedy binning of target closures.

    The largest closure seeds each bin; compatible targets are then chosen by minimum marginal
    token cost, breaking ties chronologically. Target counts are small in these corpora, making
    this transparent strategy preferable to an opaque optimizer.
    """
    remaining = [closure for closure in closures if closure.valid and len(closure.indices) <= budget]
    output = []
    while remaining:
        seed = max(remaining, key=lambda item: (len(item.indices), -item.target_message))
        group = [seed]
        selected = set(seed.indices)
        remaining.remove(seed)
        while remaining:
            candidates = [
                (len(set(item.indices) - selected), item.target_message, item)
                for item in remaining
                if len(selected | set(item.indices)) <= budget
            ]
            if not candidates:
                break
            _, _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
            group.append(chosen)
            selected |= set(chosen.indices)
            remaining.remove(chosen)
        example = _make_example(conversation, rendered, group, budget)
        if example is not None:
            output.append(example)
    return output


def materialize_source_target_round(
    conversation: Conversation,
    rendered: RenderedConversation,
    closures: list[TargetClosure],
    edges: Iterable[SupportEdge],
    budget: int,
) -> list[MaterializedExample]:
    by_target: dict[int, list[SupportEdge]] = {}
    for edge in edges:
        by_target.setdefault(edge.target_message, []).append(edge)
    output = []
    valid_targets = {closure.target_message for closure in closures if closure.valid}
    for target in sorted(valid_targets):
        target_block = rendered.message_spans[target]
        selected = set(range(target_block.block_start, target_block.block_end))
        for edge in by_target[target]:
            source = rendered.message_spans[edge.source_message]
            selected |= set(range(source.block_start, source.block_end))
        synthetic = TargetClosure(
            conversation.row_id, target, frozenset(selected), frozenset(), True
        )
        example = _make_example(conversation, rendered, [synthetic], budget, fill=False)
        if example is not None:
            output.append(example)
    return output


def materialize_last_complete_tool_round(
    conversation: Conversation,
    rendered: RenderedConversation,
    closures: list[TargetClosure],
    edges: Iterable[SupportEdge],
    budget: int,
) -> list[MaterializedExample]:
    """Keep complete blocks from the latest recorded support message through each target."""
    by_target: dict[int, list[SupportEdge]] = {}
    for edge in edges:
        by_target.setdefault(edge.target_message, []).append(edge)
    output = []
    valid_targets = {closure.target_message for closure in closures if closure.valid}
    for target in sorted(valid_targets):
        source = max(edge.source_message for edge in by_target[target])
        selected = set()
        for span in rendered.message_spans[source : target + 1]:
            selected |= set(range(span.block_start, span.block_end))
        synthetic = TargetClosure(
            conversation.row_id, target, frozenset(selected), frozenset(), True
        )
        example = _make_example(conversation, rendered, [synthetic], budget, fill=False)
        if example is not None:
            output.append(example)
    return output


def materialize_masked_keep_end(
    conversation: Conversation,
    rendered: RenderedConversation,
    closures: list[TargetClosure],
    budget: int,
) -> list[MaterializedExample]:
    """Historical keep-end with labels removed from oracle targets whose sources are absent."""
    start = max(0, len(rendered.input_ids) - budget)
    indices = tuple(range(start, len(rendered.input_ids)))
    kept = set(indices)
    labels = list(rendered.labels[start:])
    supported_targets = []
    for closure in closures:
        if not closure.valid:
            continue
        target = rendered.message_spans[closure.target_message]
        target_label_indices = {
            index
            for index in range(target.block_start, target.block_end)
            if rendered.labels[index] != -100
        }
        target_trained = bool(target_label_indices & kept)
        supported = target_trained and closure.source_indices <= kept
        if supported:
            supported_targets.append(closure.target_message)
        elif target_trained:
            for original_index in target_label_indices & kept:
                labels[original_index - start] = -100
    if not any(label != -100 for label in labels):
        return []
    return [
        MaterializedExample(
            conversation.row_id,
            tuple(sorted(supported_targets)),
            indices,
            tuple(labels),
        )
    ]


def target_messages_repaired(examples: Iterable[MaterializedExample]) -> set[int]:
    return {target for example in examples for target in example.target_messages}
