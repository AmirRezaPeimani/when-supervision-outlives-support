from __future__ import annotations

from pathlib import Path

from .types import Conversation, MessageSpan, RenderedConversation


def load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)


def _qwen_parts(role: str, content: str) -> tuple[str, str, str]:
    if role == "tool":
        return "<|im_start|>user\n<tool_response>\n", content, "\n</tool_response><|im_end|>\n"
    return f"<|im_start|>{role}\n", content, "<|im_end|>\n"


def _manual_render(conversation: Conversation, tokenizer) -> RenderedConversation:
    """Fallback used by unit-test tokenizers without chat-template support."""
    text_parts: list[str] = []
    char_spans: list[tuple[int, str, int, int, int, int]] = []
    cursor = 0
    for index, message in enumerate(conversation.messages):
        prefix, content, suffix = _qwen_parts(message.role, message.content)
        block_start = cursor
        content_start = block_start + len(prefix)
        content_end = content_start + len(content)
        block_end = content_end + len(suffix)
        text_parts.extend((prefix, content, suffix))
        char_spans.append((index, message.role, block_start, content_start, content_end, block_end))
        cursor = block_end

    text = "".join(text_parts)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    token_offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    labels = [-100] * len(input_ids)
    spans = []
    content_offsets = []

    def overlapping(start: int, end: int) -> list[int]:
        return [i for i, (left, right) in enumerate(token_offsets) if right > start and left < end]

    for index, role, block_char_start, content_char_start, content_char_end, block_char_end in char_spans:
        block_tokens = overlapping(block_char_start, block_char_end)
        body_tokens = overlapping(content_char_start, content_char_end)
        if not block_tokens:
            raise RuntimeError(f"empty rendered message block: {conversation.row_id}/{index}")
        block_start = min(block_tokens)
        block_end = max(block_tokens) + 1
        if body_tokens:
            content_start = min(body_tokens)
            content_end = max(body_tokens) + 1
            relative = tuple(
                (
                    max(token_offsets[i][0] - content_char_start, 0),
                    min(token_offsets[i][1] - content_char_start, content_char_end - content_char_start),
                )
                for i in range(content_start, content_end)
            )
        else:
            content_start = content_end = block_end
            relative = ()
        spans.append(MessageSpan(index, role, block_start, content_start, content_end, block_end))
        content_offsets.append(relative)
        if role == "assistant":
            for token_index in range(content_start, block_end):
                labels[token_index] = input_ids[token_index]
    return RenderedConversation(
        tuple(input_ids), tuple(labels), tuple(spans), "qwen25_current", tuple(content_offsets)
    )


def _training_template(tokenizer):
    from trl.chat_template_utils import get_training_chat_template, has_generation_markers

    if has_generation_markers(tokenizer.chat_template):
        return None
    return get_training_chat_template(tokenizer)


def render_qwen(conversation: Conversation, tokenizer) -> RenderedConversation:
    """Render with the exact current TRL training template, mask, and tokenizer offsets."""
    if not hasattr(tokenizer, "apply_chat_template"):
        return _manual_render(conversation, tokenizer)
    template = _training_template(tokenizer)
    messages = []
    for message in conversation.messages:
        payload = {"role": message.role, "content": message.content}
        payload.update(message.metadata)
        messages.append(payload)
    tools = conversation.metadata.get("tools") or None
    common = {"chat_template": template, "tools": tools}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **common
    )
    native = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **common,
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    native_ids = list(native["input_ids"])
    if input_ids != native_ids:
        raise RuntimeError(f"native text/token rendering mismatch: {conversation.row_id}")
    assistant_mask = list(native["assistant_masks"])
    if len(assistant_mask) != len(input_ids):
        raise RuntimeError(f"assistant mask length mismatch: {conversation.row_id}")
    labels = [token if mask else -100 for token, mask in zip(input_ids, assistant_mask, strict=True)]
    token_offsets = [tuple(pair) for pair in encoded["offset_mapping"]]

    char_spans = []
    search_cursor = 0
    for index, message in enumerate(conversation.messages):
        content = message.content
        if content:
            content_start = text.find(content, search_cursor)
            if content_start < 0:
                raise RuntimeError(f"message content not found in native rendering: {conversation.row_id}/{index}")
            content_end = content_start + len(content)
        else:
            content_start = search_cursor
            content_end = search_cursor
        block_start = text.rfind("<|im_start|>", search_cursor, content_start + 1)
        if block_start < 0:
            block_start = search_cursor
        marker = text.find("<|im_end|>", content_end)
        if marker < 0:
            block_end = content_end
        else:
            block_end = marker + len("<|im_end|>")
            if block_end < len(text) and text[block_end] == "\n":
                block_end += 1
        if block_end < content_end:
            raise RuntimeError(f"invalid native message bounds: {conversation.row_id}/{index}")
        char_spans.append((index, message.role, block_start, content_start, content_end, block_end))
        search_cursor = block_end

    def overlapping(start: int, end: int) -> list[int]:
        return [i for i, (left, right) in enumerate(token_offsets) if right > start and left < end]

    spans = []
    content_offsets = []
    for index, role, block_char_start, content_char_start, content_char_end, block_char_end in char_spans:
        block_tokens = overlapping(block_char_start, block_char_end)
        body_tokens = overlapping(content_char_start, content_char_end)
        if not block_tokens:
            raise RuntimeError(f"empty native message block: {conversation.row_id}/{index}")
        block_start = min(block_tokens)
        block_end = max(block_tokens) + 1
        if body_tokens:
            content_start = min(body_tokens)
            content_end = max(body_tokens) + 1
            relative = tuple(
                (
                    max(token_offsets[i][0] - content_char_start, 0),
                    min(token_offsets[i][1] - content_char_start, content_char_end - content_char_start),
                )
                for i in range(content_start, content_end)
            )
        else:
            content_start = content_end = block_end
            relative = ()
        spans.append(MessageSpan(index, role, block_start, content_start, content_end, block_end))
        content_offsets.append(relative)
    return RenderedConversation(
        tuple(input_ids), tuple(labels), tuple(spans), "qwen25_current", tuple(content_offsets)
    )


def locate_value(
    rendered: RenderedConversation,
    message_index: int,
    value: str,
    content: str,
    case_sensitive: bool = False,
) -> list[tuple[int, int]]:
    offsets = rendered.content_offsets[message_index]
    if not offsets or not value:
        return []
    span = rendered.message_spans[message_index]
    hits = []
    search_content = content if case_sensitive else content.casefold()
    search_value = value if case_sensitive else value.casefold()
    cursor = 0
    while True:
        cursor = search_content.find(search_value, cursor)
        if cursor < 0:
            break
        end_char = cursor + len(value)
        relative_indices = [
            index for index, (left, right) in enumerate(offsets)
            if right > cursor and left < end_char
        ]
        if relative_indices:
            hits.append(
                (
                    span.content_start + min(relative_indices),
                    span.content_start + max(relative_indices) + 1,
                )
            )
        cursor = end_char
    return hits
