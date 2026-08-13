from __future__ import annotations

import json
import re
from collections.abc import Iterator

from .types import Conversation, SupportEdge


COMMON = {
    "true", "false", "null", "none", "yes", "no", "success", "error", "assistant",
    "user", "tool", "function", "python", "string", "number", "object", "array",
}
NUMBER = re.compile(r"(?<![\w.])-?(?:\d+\.\d+|\d{2,})(?![\w.])")


def _walk(value: object, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield str(value), path


def atomic_values(text: str, minimum_length: int = 3) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    candidates = [text, *re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)]
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        for value, path in _walk(payload):
            value = value.strip()
            if len(value) >= minimum_length and value.casefold() not in COMMON:
                found.setdefault(value, path)
    for value in NUMBER.findall(text):
        found.setdefault(value, "regex:number")
    return sorted(found.items(), key=lambda item: (-len(item[0]), item[0]))


def _contains(text: str, value: str, case_sensitive: bool = False) -> bool:
    if value.replace(".", "", 1).lstrip("-").isdigit():
        return re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", text) is not None
    return value in text if case_sensitive else value.casefold() in text.casefold()


def discover_observation_edges(
    conversation: Conversation, case_sensitive: bool = False
) -> list[SupportEdge]:
    edges = []
    for source_index, source in enumerate(conversation.messages[:-1]):
        if source.role != "tool":
            continue
        target_index = source_index + 1
        while target_index < len(conversation.messages) and conversation.messages[target_index].role == "tool":
            target_index += 1
        if target_index >= len(conversation.messages) or conversation.messages[target_index].role != "assistant":
            continue
        target = conversation.messages[target_index].content
        earlier_non_tool = "\n".join(
            message.content
            for message in conversation.messages[:source_index]
            if message.role != "tool"
        )
        for value, path in atomic_values(source.content):
            if not _contains(target, value, case_sensitive) or _contains(
                earlier_non_tool, value, case_sensitive
            ):
                continue
            edges.append(
                SupportEdge(
                    f"{conversation.row_id}:t{source_index}-{target_index}:{len(edges)}",
                    conversation.row_id,
                    source_index,
                    target_index,
                    value,
                    value,
                    "tool_to_assistant",
                    "identity",
                    path,
                )
            )
    unique = {}
    for edge in edges:
        unique.setdefault(
            (edge.source_message, edge.target_message, edge.source_value), edge
        )
    return list(unique.values())
