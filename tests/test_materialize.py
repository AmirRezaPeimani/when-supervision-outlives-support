import pytest

from vsr.materialize import (
    MaterializedExample,
    build_target_closures,
    materialize_grouped,
    materialize_single_target,
    validate_serialized_materialized_example,
)
from vsr.render import _manual_render
from vsr.types import Conversation, Message, SupportEdge


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_grouped_closure_combines_compatible_targets_and_masks_other_content():
    conversation = Conversation(
        "fixture",
        "fixture:0",
        (
            Message("user", "find both"),
            Message("tool", "alpha=A1 beta=B2"),
            Message("assistant", "first A1"),
            Message("assistant", "second B2"),
        ),
    )
    rendered = _manual_render(conversation, CharacterTokenizer())
    edges = [
        SupportEdge("e1", "fixture:0", 1, 2, "A1", "A1", "tool_to_assistant"),
        SupportEdge("e2", "fixture:0", 1, 3, "B2", "B2", "tool_to_assistant"),
    ]
    closures = build_target_closures(conversation, rendered, edges)
    minimum = len(set().union(*(closure.indices for closure in closures)))
    grouped = materialize_grouped(conversation, rendered, closures, minimum)
    single = materialize_single_target(conversation, rendered, closures, minimum)
    assert len(grouped) == 1
    assert grouped[0].target_messages == (2, 3)
    assert len(single) == 2
    assert grouped[0].supervised_tokens > 0
    for index, label in zip(grouped[0].original_indices, grouped[0].labels, strict=True):
        in_target = any(
            rendered.message_spans[target].block_start <= index < rendered.message_spans[target].block_end
            for target in (2, 3)
        )
        assert (label != -100) <= in_target


def test_materializer_never_emits_a_partial_target_closure():
    conversation = Conversation(
        "fixture",
        "fixture:partial",
        (Message("tool", "identifier=ZX9"), Message("assistant", "VALUE=ZX9")),
    )
    rendered = _manual_render(conversation, CharacterTokenizer())
    edge = SupportEdge("edge", conversation.row_id, 0, 1, "ZX9", "ZX9", "tool_to_assistant")
    closure = build_target_closures(conversation, rendered, [edge])[0]
    assert materialize_single_target(conversation, rendered, [closure], len(closure.indices) - 1) == []
    example = materialize_single_target(conversation, rendered, [closure], len(closure.indices))[0]
    target = rendered.message_spans[1]
    assert set(range(target.block_start, target.block_end)) <= set(example.original_indices)


def test_serialized_materialization_validator_accepts_causally_visible_source():
    conversation = Conversation(
        "fixture",
        "fixture:serialized-valid",
        (Message("tool", "identifier=ZX9"), Message("assistant", "VALUE=ZX9")),
    )
    rendered = _manual_render(conversation, CharacterTokenizer())
    edge = SupportEdge("edge-valid", conversation.row_id, 0, 1, "ZX9", "ZX9", "tool_to_assistant")
    closure = build_target_closures(conversation, rendered, [edge])[0]
    example = materialize_single_target(conversation, rendered, [closure], len(closure.indices))[0]
    validate_serialized_materialized_example(conversation, rendered, [edge], example)


def test_serialized_materialization_validator_rejects_source_after_target():
    conversation = Conversation(
        "fixture",
        "fixture:serialized-after",
        (Message("tool", "identifier=ZX9"), Message("assistant", "VALUE=ZX9")),
    )
    rendered = _manual_render(conversation, CharacterTokenizer())
    edge = SupportEdge("edge-after", conversation.row_id, 0, 1, "ZX9", "ZX9", "tool_to_assistant")
    closure = build_target_closures(conversation, rendered, [edge])[0]
    target = rendered.message_spans[1]
    target_indices = tuple(range(target.block_start, target.block_end))
    source_indices = tuple(sorted(closure.source_indices))
    example = MaterializedExample(
        conversation.row_id,
        (1,),
        target_indices + source_indices,
        tuple(rendered.labels[index] for index in target_indices) + (-100,) * len(source_indices),
    )
    with pytest.raises(ValueError, match="causally visible"):
        validate_serialized_materialized_example(conversation, rendered, [edge], example)
