from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset

from .render import render_qwen
from .types import Conversation, Message


def load_records(path: str | Path, maximum: int | None = None) -> list[dict]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    return records if maximum is None else records[:maximum]


def as_conversation(record: dict, include_target: bool = True) -> Conversation:
    messages = record["messages"] if include_target else record["messages"][:-1]
    return Conversation(
        "synthetic",
        record["record_id"],
        tuple(
            Message(
                message["role"],
                message.get("content", ""),
                {key: value for key, value in message.items() if key not in {"role", "content"}},
            )
            for message in messages
        ),
        {"tools": record["tools"]},
    )


def tokenize_training_records(records: list[dict], tokenizer, max_length: int) -> Dataset:
    rows = []
    for record in records:
        rendered = render_qwen(as_conversation(record), tokenizer)
        target = rendered.message_spans[-1]
        input_ids = list(rendered.input_ids[:max_length])
        labels = [
            rendered.labels[index]
            if target.block_start <= index < target.block_end
            else -100
            for index in range(min(len(rendered.input_ids), max_length))
        ]
        if not any(label != -100 for label in labels):
            raise ValueError(f"final target is outside max_length: {record['record_id']}")
        rows.append(
            {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
        )
    return Dataset.from_list(rows)


class CausalPaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full((len(features), maximum), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(features), maximum), dtype=torch.long)
        labels = torch.full((len(features), maximum), -100, dtype=torch.long)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention[row, :length] = 1
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def generation_prompt(record: dict, tokenizer) -> torch.Tensor:
    conversation = as_conversation(record, include_target=False)
    messages = []
    for message in conversation.messages:
        payload = {"role": message.role, "content": message.content}
        payload.update(message.metadata)
        messages.append(payload)
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=conversation.metadata["tools"],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Transformers 5 returns a BatchEncoding here, while earlier releases returned a tensor.
    # Keep downstream evaluation version-independent by exposing only the input-id tensor.
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded
