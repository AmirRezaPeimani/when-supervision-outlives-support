from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .types import Conversation, Message, SupportEdge


def _pythonize(value: object):
    if isinstance(value, dict):
        return {str(key): _pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pythonize(item) for item in value]
    if hasattr(value, "tolist"):
        return _pythonize(value.tolist())
    return value


def _records(value: object) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    value = _pythonize(value)
    return [dict(item) for item in value]  # type: ignore[arg-type]


def load_toolace(path: str | Path, maximum_rows: int | None = None) -> list[Conversation]:
    frame = pd.read_parquet(path)
    if maximum_rows is not None:
        frame = frame.iloc[:maximum_rows]
    output = []
    for index, row in frame.iterrows():
        messages = [Message("system", str(row["system"]))]
        for item in _records(row["conversations"]):
            role = {"human": "user", "gpt": "assistant", "function": "tool"}.get(
                str(item.get("from", "")), str(item.get("from", ""))
            )
            messages.append(Message(role, str(item.get("value", ""))))
        output.append(Conversation("toolace", f"toolace:{index}", tuple(messages)))
    return output


def load_retool(path: str | Path, maximum_rows: int | None = None) -> list[Conversation]:
    frame = pd.read_parquet(path)
    if maximum_rows is not None:
        frame = frame.iloc[:maximum_rows]
    output = []
    for index, row in frame.iterrows():
        messages = []
        for item in _records(row["messages"]):
            content = item.get("content")
            if content is None and item.get("tool_calls"):
                content = json.dumps(item["tool_calls"], ensure_ascii=False, sort_keys=True)
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"role", "content"} and value is not None
            }
            messages.append(Message(str(item.get("role", "")), str(content or ""), metadata))
        output.append(
            Conversation(
                "retool",
                f"retool:{index}",
                tuple(messages),
                metadata={"tools": _records(row.get("tools"))},
            )
        )
    return output


def load_glaive(path: str | Path, maximum_rows: int | None = None) -> list[Conversation]:
    frame = pd.read_parquet(path)
    if maximum_rows is not None:
        frame = frame.iloc[:maximum_rows]
    output = []
    splitter = re.compile(r"(?:^|\n\s*\n?)\s*(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
    for index, row in frame.iterrows():
        system = str(row.get("system", "")).strip()
        if system.startswith("SYSTEM:"):
            system = system[len("SYSTEM:") :].strip()
        messages = [Message("system", system)] if system else []
        parts = splitter.split(str(row.get("chat", "")))
        for role, content in zip(parts[1::2], parts[2::2], strict=True):
            normalized_role = {
                "USER": "user",
                "ASSISTANT": "assistant",
                "FUNCTION RESPONSE": "tool",
            }[role]
            cleaned = re.sub(r"\s*<\|endoftext\|>\s*$", "", content.strip())
            messages.append(Message(normalized_role, cleaned))
        original_index = row.get("original_row_index", index)
        output.append(Conversation("glaive", f"glaive:{int(original_index)}", tuple(messages)))
    return output


def load_corpora(
    raw_dir: str | Path,
    names: Iterable[str],
    maximum_rows: int | None = None,
) -> list[Conversation]:
    raw_dir = Path(raw_dir)
    output: list[Conversation] = []
    for name in names:
        if name == "toolace":
            output.extend(load_toolace(raw_dir / "toolace.parquet", maximum_rows))
        elif name == "retool":
            output.extend(load_retool(raw_dir / "retool.parquet", maximum_rows))
        elif name == "glaive":
            output.extend(
                load_glaive(
                    raw_dir / "glaive_function_calling_v2" / "train-0000.parquet",
                    maximum_rows,
                )
            )
        else:
            raise ValueError(f"unknown corpus: {name}")
    return output


def load_v1_edges(path: str | Path, kinds: set[str] | None = None) -> dict[str, list[SupportEdge]]:
    output: dict[str, list[SupportEdge]] = {}
    with Path(path).open() as handle:
        for line in handle:
            payload = json.loads(line)
            if kinds is not None and payload["kind"] not in kinds:
                continue
            edge = SupportEdge(**payload)
            output.setdefault(edge.row_id, []).append(edge)
    return output
