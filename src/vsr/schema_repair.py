from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass

from .render import locate_value, render_qwen
from .types import Conversation, Message, RenderedConversation, SupportEdge


CALL_NAME = re.compile(r"(?:^|\[|,\s*)([^()[\],]+?)\s*\(")
ARGUMENT = re.compile(r"(?:^|[(,\s])([A-Za-z_][A-Za-z0-9_]*)\s*=")


@dataclass(frozen=True)
class SchemaBundle:
    prefix: str
    schemas: tuple[dict, ...]
    suffix: str


@dataclass(frozen=True)
class ActionRepair:
    row_id: str
    target_message: int
    mode: str
    budget: int
    success: bool
    reason: str
    length: int
    schema_tokens: int
    supervised_tokens: int
    invoked_tools: tuple[str, ...]
    original_indices: tuple[int, ...] = ()
    labels: tuple[int, ...] = ()


def extract_inline_schema(system_text: str) -> SchemaBundle | None:
    marker = "Here is a list of functions in JSON format that you can invoke:"
    marker_index = system_text.find(marker)
    if marker_index < 0:
        return None
    start = system_text.find("[", marker_index + len(marker))
    if start < 0:
        return None
    try:
        schemas, consumed = json.JSONDecoder().raw_decode(system_text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(schemas, list) or not all(isinstance(item, dict) for item in schemas):
        return None
    return SchemaBundle(system_text[:start], tuple(schemas), system_text[start + consumed :])


def _schema_name(schema: dict) -> str:
    function = schema.get("function")
    if isinstance(function, dict):
        return str(function.get("name", ""))
    return str(schema.get("name", ""))


def _schema_function(schema: dict) -> dict:
    function = schema.get("function")
    return function if isinstance(function, dict) else schema


def invocation_requirements(message: Message) -> dict[str, set[str]]:
    requirements: dict[str, set[str]] = {}
    calls = message.metadata.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function", call)
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip()
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            keys = set(map(str, arguments)) if isinstance(arguments, dict) else set()
            if name:
                requirements.setdefault(name, set()).update(keys)
    if requirements:
        return requirements
    for name in CALL_NAME.findall(message.content):
        cleaned = name.strip().strip("'").strip('"')
        if cleaned:
            requirements.setdefault(cleaned, set()).update(ARGUMENT.findall(message.content))
    return requirements


def filter_schemas(
    schemas: list[dict] | tuple[dict, ...],
    requirements: dict[str, set[str]],
    parameter_slice: bool,
) -> list[dict]:
    requested = {name.casefold(): parameters for name, parameters in requirements.items()}
    selected = []
    for raw in schemas:
        name = _schema_name(raw)
        if name.casefold() not in requested:
            continue
        schema = copy.deepcopy(raw)
        if parameter_slice:
            function = _schema_function(schema)
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                used = requested[name.casefold()]
                properties = parameters.get("properties")
                if isinstance(properties, dict):
                    parameters["properties"] = {
                        key: value for key, value in properties.items() if key in used
                    }
                required = parameters.get("required")
                if isinstance(required, list):
                    parameters["required"] = [key for key in required if key in used]
        selected.append(schema)
    return selected


def _modified_conversation(
    conversation: Conversation,
    requirements: dict[str, set[str]],
    mode: str,
) -> tuple[Conversation | None, str]:
    if mode == "whole_schema":
        return conversation, ""
    parameter_slice = mode == "parameter_slice"
    if mode not in {"invoked_tool", "parameter_slice"}:
        return None, "unknown repair mode"
    system_index = next(
        (index for index, message in enumerate(conversation.messages) if message.role == "system"),
        None,
    )
    if system_index is not None:
        system = conversation.messages[system_index]
        bundle = extract_inline_schema(system.content)
        if bundle is not None:
            selected = filter_schemas(bundle.schemas, requirements, parameter_slice)
            if len(selected) != len(requirements):
                return None, "one or more invoked tools not found in schema list"
            replacement = bundle.prefix + json.dumps(selected, ensure_ascii=False, separators=(",", ":")) + bundle.suffix
            messages = list(conversation.messages)
            messages[system_index] = Message("system", replacement, system.metadata)
            return Conversation(conversation.corpus, conversation.row_id, tuple(messages), conversation.metadata), ""
    tools = conversation.metadata.get("tools")
    if not isinstance(tools, list):
        return None, "inline or structured tool declarations unavailable"
    selected = filter_schemas(tools, requirements, parameter_slice)
    if len(selected) != len(requirements):
        return None, "one or more invoked tools not found in structured declarations"
    metadata = dict(conversation.metadata)
    metadata["tools"] = selected
    return Conversation(conversation.corpus, conversation.row_id, conversation.messages, metadata), ""


def _schema_indices(conversation: Conversation, rendered: RenderedConversation) -> set[int]:
    system = next((span for span in rendered.message_spans if span.role == "system"), None)
    if system is not None:
        return set(range(system.block_start, system.block_end))
    first_message = min((span.block_start for span in rendered.message_spans), default=0)
    return set(range(first_message))


def repair_action_target(
    conversation: Conversation,
    tokenizer,
    edges: list[SupportEdge],
    target_message: int,
    budget: int,
    mode: str,
) -> ActionRepair:
    target_edges = [edge for edge in edges if edge.target_message == target_message]
    requirements = invocation_requirements(conversation.messages[target_message])
    if not requirements:
        return ActionRepair(conversation.row_id, target_message, mode, budget, False, "invoked tool not parsed", 0, 0, 0, ())
    modified, reason = _modified_conversation(conversation, requirements, mode)
    if modified is None:
        return ActionRepair(conversation.row_id, target_message, mode, budget, False, reason, 0, 0, 0, tuple(requirements))
    rendered = render_qwen(modified, tokenizer)
    schema = _schema_indices(modified, rendered)
    target_span = rendered.message_spans[target_message]
    selected = set(schema) | set(range(target_span.block_start, target_span.block_end))
    for edge in target_edges:
        message = modified.messages[edge.source_message]
        hits = locate_value(rendered, edge.source_message, edge.source_value, message.content)
        if not hits:
            return ActionRepair(conversation.row_id, target_message, mode, budget, False, "recorded user constraint not located", 0, len(schema), 0, tuple(requirements))
        hit = max(hits, key=lambda item: item[0])
        selected |= set(range(*hit))
        source_span = rendered.message_spans[edge.source_message]
        selected |= set(range(source_span.block_start, source_span.content_start))
        selected |= set(range(source_span.content_end, source_span.block_end))
    if len(selected) > budget:
        return ActionRepair(conversation.row_id, target_message, mode, budget, False, "closure exceeds budget", len(selected), len(schema), 0, tuple(requirements))
    indices = tuple(sorted(selected))
    labels = tuple(
        rendered.labels[index]
        if target_span.block_start <= index < target_span.block_end
        else -100
        for index in indices
    )
    supervised = sum(label != -100 for label in labels)
    if not supervised:
        return ActionRepair(conversation.row_id, target_message, mode, budget, False, "target has no assistant labels", len(indices), len(schema), 0, tuple(requirements))
    return ActionRepair(
        conversation.row_id,
        target_message,
        mode,
        budget,
        True,
        "",
        len(indices),
        len(schema),
        supervised,
        tuple(requirements),
        indices,
        labels,
    )
