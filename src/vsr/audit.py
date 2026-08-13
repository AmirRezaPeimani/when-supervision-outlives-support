from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .policies import (
    causal_attention_relation,
    span_occurrences,
)
from .render import locate_value
from .types import Conversation, PolicyView, RenderedConversation, SupportEdge


NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?$")
OPAQUE = re.compile(r"^(?=.{6,}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_.:/-]+$")


def value_category(value: str) -> str:
    stripped = value.strip()
    lowered = stripped.casefold()
    if NUMERIC.fullmatch(stripped):
        return "numeric"
    if OPAQUE.fullmatch(stripped):
        if "error" in lowered or re.search(r"(?:^|[-_])e\d", lowered):
            return "error_code"
        return "opaque_identifier"
    if len(stripped.split()) <= 4:
        return "short_string"
    return "long_string"


def locate_edge(
    conversation: Conversation,
    rendered: RenderedConversation,
    edge: SupportEdge,
    case_sensitive: bool = False,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    support_role = "tool" if edge.kind == "tool_to_assistant" else "user"
    source_message = conversation.messages[edge.source_message]
    source_hits = (
        locate_value(
            rendered,
            edge.source_message,
            edge.source_value,
            source_message.content,
            case_sensitive=case_sensitive,
        )
        if source_message.role == support_role
        else []
    )
    target_message = conversation.messages[edge.target_message]
    target_hits = locate_value(
        rendered,
        edge.target_message,
        edge.target_value,
        target_message.content,
        case_sensitive=case_sensitive,
    )
    return source_hits, target_hits


def audit_edges_for_view(
    conversations: list[Conversation],
    rendered_rows: list[RenderedConversation],
    edges_by_row: dict[str, list[SupportEdge]],
    view: PolicyView,
    case_sensitive: bool = False,
) -> list[dict[str, Any]]:
    records = []
    row_lookup = {conversation.row_id: index for index, conversation in enumerate(conversations)}
    for row_id, edges in edges_by_row.items():
        if row_id not in row_lookup:
            continue
        row_index = row_lookup[row_id]
        conversation = conversations[row_index]
        rendered = rendered_rows[row_index]
        for edge in edges:
            source_hits, target_hits = locate_edge(
                conversation, rendered, edge, case_sensitive=case_sensitive
            )
            source_occurrences = [
                occurrence
                for source_hit in source_hits
                for occurrence in span_occurrences(view, row_index, source_hit)
            ]
            trained_target_occurrences = []
            unsupported_occurrences = []
            for target_hit in target_hits:
                for group_id, target_start, target_end in span_occurrences(
                    view, row_index, target_hit
                ):
                    effective_positions = {
                        position
                        for position in view.effective_label_positions_by_group.get(group_id, ())
                        if target_start <= position < target_end
                    }
                    if not effective_positions:
                        continue
                    occurrence = (group_id, target_start, target_end, effective_positions, target_hit)
                    trained_target_occurrences.append(occurrence)
                    occurrence_supported = any(
                        all(
                            causal_attention_relation(
                                (group_id, target_position), (source_group, source_position)
                            )
                            for target_position in effective_positions
                            for source_position in range(source_start, source_end)
                        )
                        for source_group, source_start, source_end in source_occurrences
                    )
                    if not occurrence_supported:
                        unsupported_occurrences.append(occurrence)
            target_trained = bool(trained_target_occurrences)
            unsupported = bool(unsupported_occurrences)
            supported = target_trained and not unsupported
            effective_target_tokens = set()
            effective_target_positions = set()
            for group_id, target_start, _, positions, target_hit in unsupported_occurrences:
                original_start, _ = target_hit
                for position in positions:
                    effective_target_positions.add((group_id, position))
                    effective_target_tokens.add(original_start + position - target_start)
            closest = None
            for source_start, source_end in source_hits:
                for target_start, target_end in target_hits:
                    if source_start < target_start:
                        candidate = {
                            "distance": target_start - source_end,
                            "span": target_end - source_start,
                            "keep_end": len(rendered.input_ids) - source_start,
                        }
                        if closest is None or candidate["span"] < closest["span"]:
                            closest = candidate
            records.append(
                {
                    "corpus": conversation.corpus,
                    "row_id": row_id,
                    "row_index": row_index,
                    "edge_id": edge.edge_id,
                    "unit_id": f"{row_id}:{edge.source_message}-{edge.target_message}:{edge.kind}",
                    "kind": edge.kind,
                    "source_message": edge.source_message,
                    "target_message": edge.target_message,
                    "policy": view.policy,
                    "max_length": view.budget,
                    "full_length": len(rendered.input_ids),
                    "source_located": bool(source_hits),
                    "target_located": bool(target_hits),
                    "target_trained": target_trained,
                    "source_reachable": supported,
                    "unsupported_target": unsupported,
                    "unsupported_target_tokens": len(effective_target_tokens),
                    "unsupported_target_token_indices": sorted(effective_target_tokens),
                    "unsupported_target_token_occurrences": len(effective_target_positions),
                    "unsupported_target_positions": [
                        f"{group_id}:{index}"
                        for group_id, index in sorted(effective_target_positions)
                    ],
                    "source_value_length": len(edge.source_value),
                    "target_value_length": len(edge.target_value),
                    "numeric_only": bool(NUMERIC.fullmatch(edge.source_value.strip())),
                    "opaque_identifier": bool(OPAQUE.fullmatch(edge.source_value.strip())),
                    "value_category": value_category(edge.source_value),
                    "source_target_distance": None if closest is None else closest["distance"],
                    "critical_contiguous_budget": None if closest is None else closest["span"],
                    "critical_keep_end_budget": None if closest is None else closest["keep_end"],
                }
            )
    return records


def aggregate_units(edge_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_records:
        grouped[(row["corpus"], row["row_id"], row["unit_id"], row["policy"], row["max_length"])].append(row)
    output = []
    for key, rows in grouped.items():
        valid = all(row["source_located"] and row["target_located"] for row in rows)
        output.append(
            {
                "corpus": key[0],
                "row_id": key[1],
                "unit_id": key[2],
                "policy": key[3],
                "max_length": key[4],
                "edge_count": len(rows),
                "valid_oracle": valid,
                "target_trained": any(row["target_trained"] for row in rows),
                "unsupported_target": any(row["unsupported_target"] for row in rows),
                "unsupported_target_tokens": len(
                    {
                        index
                        for row in rows
                        for index in row["unsupported_target_token_indices"]
                    }
                ),
                "unsupported_target_token_occurrences": len(
                    {
                        position
                        for row in rows
                        for position in row["unsupported_target_positions"]
                    }
                ),
                "max_value_length": max(row["source_value_length"] for row in rows),
                "all_nonnumeric": all(not row["numeric_only"] for row in rows),
                "has_opaque_identifier": any(row["opaque_identifier"] for row in rows),
                "source_target_distance": min(
                    (row["source_target_distance"] for row in rows if row["source_target_distance"] is not None),
                    default=None,
                ),
                "critical_contiguous_budget": min(
                    (row["critical_contiguous_budget"] for row in rows if row["critical_contiguous_budget"] is not None),
                    default=None,
                ),
                "critical_keep_end_budget": min(
                    (row["critical_keep_end_budget"] for row in rows if row["critical_keep_end_budget"] is not None),
                    default=None,
                ),
            }
        )
    return output
