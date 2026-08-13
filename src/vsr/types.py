from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Conversation:
    corpus: str
    row_id: str
    messages: tuple[Message, ...]
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class SupportEdge:
    edge_id: str
    row_id: str
    source_message: int
    target_message: int
    source_value: str
    target_value: str
    kind: str
    transform: str = "identity"
    evidence_path: str | None = None


@dataclass(frozen=True)
class MessageSpan:
    message_index: int
    role: str
    block_start: int
    content_start: int
    content_end: int
    block_end: int


@dataclass(frozen=True)
class RenderedConversation:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    message_spans: tuple[MessageSpan, ...]
    template: str
    content_offsets: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True)
class RetainedInterval:
    """Contiguous original-row interval and its positions in one attention group."""

    row_index: int
    start: int
    end: int
    group_id: int
    packed_example: int
    group_position_start: int
    group_position_end: int


@dataclass
class PolicyView:
    policy: str
    budget: int
    row_intervals: dict[int, list[RetainedInterval]]
    group_first_tokens: set[tuple[int, int, int]]
    effective_indices_by_row_group: dict[tuple[int, int], frozenset[int]]
    effective_label_positions_by_group: dict[int, frozenset[int]]
    retained_original_tokens: int
    unique_retained_original_tokens: int
    effective_supervised_tokens: int
    unique_effective_supervised_tokens: int
    generated_examples: int
    attention_groups: int
    rows_with_tokens: set[int]
    rows_with_supervision: set[int]
    packed_lengths: tuple[int, ...]
