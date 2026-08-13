#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trl
from trl.chat_template_utils import get_training_chat_template, has_generation_markers

from vsr.data import load_corpora
from vsr.policies import materialize_packed
from vsr.render import load_tokenizer, render_qwen
from vsr.types import MessageSpan, RenderedConversation


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_ids_and_labels(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    template = None
    if not has_generation_markers(tokenizer.chat_template):
        template = get_training_chat_template(tokenizer)
    processed = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        chat_template=template,
    )
    input_ids = list(processed["input_ids"])
    assistant_mask = list(processed["assistant_masks"])
    return input_ids, [token if mask else -100 for token, mask in zip(input_ids, assistant_mask, strict=True)]


def packing_fixture() -> list[RenderedConversation]:
    rows = []
    for row_index, length in enumerate((5, 11, 3, 8)):
        ids = tuple(row_index * 100 + index for index in range(length))
        labels = tuple(value if index >= length // 2 else -100 for index, value in enumerate(ids))
        rows.append(
            RenderedConversation(
                ids,
                labels,
                (MessageSpan(0, "assistant", 0, 0, length, length),),
                "fixture",
                (((0, length),),),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-corpus", type=int, default=200)
    parser.add_argument("--output", default="analysis/conformance/current_trl.json")
    args = parser.parse_args()
    tokenizer = load_tokenizer(ROOT / "models/qwen2.5-0.5b-instruct")
    conversations = load_corpora(ROOT / "data/raw", ["toolace", "retool"], args.rows_per_corpus)
    mismatches = []
    label_exact = 0
    id_exact = 0
    for conversation in conversations:
        manual = render_qwen(conversation, tokenizer)
        messages = []
        for message in conversation.messages:
            payload = {"role": message.role, "content": message.content}
            payload.update(message.metadata)
            messages.append(payload)
        template = None
        if not has_generation_markers(tokenizer.chat_template):
            template = get_training_chat_template(tokenizer)
        native = tokenizer.apply_chat_template(
            messages,
            tools=conversation.metadata.get("tools") or None,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            chat_template=template,
        )
        native_ids = list(native["input_ids"])
        native_labels = [
            token if mask else -100
            for token, mask in zip(native_ids, native["assistant_masks"], strict=True)
        ]
        ids_match = native_ids == list(manual.input_ids)
        labels_match = native_labels == list(manual.labels)
        id_exact += int(ids_match)
        label_exact += int(labels_match)
        if (not ids_match or not labels_match) and len(mismatches) < 20:
            mismatches.append(
                {
                    "row_id": conversation.row_id,
                    "manual_tokens": len(manual.input_ids),
                    "native_tokens": len(native_ids),
                    "ids_match": ids_match,
                    "labels_match": labels_match,
                    "manual_supervised": sum(value != -100 for value in manual.labels),
                    "native_supervised": sum(value != -100 for value in native_labels),
                }
            )

    packing = {}
    fixtures = packing_fixture()
    for strategy in ("bfd", "bfd_split", "wrapped"):
        view = materialize_packed(fixtures, 6, strategy)
        packing[strategy] = {
            "generated_examples": view.generated_examples,
            "attention_groups": view.attention_groups,
            "retained_tokens": view.retained_original_tokens,
            "effective_supervised_tokens": view.effective_supervised_tokens,
            "packed_lengths": list(view.packed_lengths),
            "intervals_cover_all_retained_tokens": sum(
                interval.end - interval.start
                for intervals in view.row_intervals.values()
                for interval in intervals
            )
            == view.retained_original_tokens,
        }

    package_root = Path(trl.__file__).resolve().parent
    sources = [
        package_root / "data_utils.py",
        package_root / "trainer/sft_trainer.py",
        package_root / "trainer/sft_config.py",
    ]
    report = {
        "trl_version": trl.__version__,
        "rows": len(conversations),
        "input_id_exact": id_exact,
        "input_id_exact_fraction": id_exact / len(conversations),
        "label_exact": label_exact,
        "label_exact_fraction": label_exact / len(conversations),
        "mismatches": mismatches,
        "packing_provenance_checks": packing,
        "source_hashes": {str(path.relative_to(package_root)): sha256(path) for path in sources},
        "numpy_version": np.__version__,
    }
    if id_exact != len(conversations) or label_exact != len(conversations):
        raise AssertionError(json.dumps(report, indent=2))
    if not all(item["intervals_cover_all_retained_tokens"] for item in packing.values()):
        raise AssertionError("packing provenance intervals are incomplete")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
