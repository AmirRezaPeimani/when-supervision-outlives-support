from vsr.audit import aggregate_units, audit_edges_for_view
from vsr.policies import _view_from_groups, materialize_nonpacked
from vsr.render import locate_value, render_qwen
from vsr.types import Conversation, Message, SupportEdge


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        output = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


def fixture():
    conversation = Conversation(
        "fixture",
        "fixture:0",
        (
            Message("user", "retrieve"),
            Message("tool", '{"id":"ZX-4917"}' + " trace" * 20),
            Message("assistant", "The identifier is ZX-4917."),
        ),
    )
    edge = SupportEdge("e", conversation.row_id, 1, 2, "ZX-4917", "ZX-4917", "tool_to_assistant")
    return conversation, edge


def test_locate_value_uses_whole_render_offsets():
    conversation, _ = fixture()
    rendered = render_qwen(conversation, CharacterTokenizer())
    assert locate_value(rendered, 2, "ZX-4917", conversation.messages[2].content)


def test_keep_end_can_leave_trained_target_unsupported():
    conversation, edge = fixture()
    rendered = render_qwen(conversation, CharacterTokenizer())
    target = rendered.message_spans[2]
    budget = target.block_end - target.block_start + 8
    view = materialize_nonpacked([rendered], budget, "keep_end")
    records = audit_edges_for_view([conversation], [rendered], {conversation.row_id: [edge]}, view)
    assert records[0]["target_trained"]
    assert records[0]["unsupported_target"]
    units = aggregate_units(records)
    assert units[0]["unsupported_target"]


def test_source_before_target_in_same_attention_group_is_reachable():
    conversation, edge = fixture()
    rendered = render_qwen(conversation, CharacterTokenizer())
    view = materialize_nonpacked([rendered], len(rendered.input_ids), "keep_start")

    record = audit_edges_for_view(
        [conversation], [rendered], {conversation.row_id: [edge]}, view
    )[0]

    assert record["target_trained"]
    assert record["source_reachable"]
    assert not record["unsupported_target"]


def test_source_after_target_in_same_attention_group_is_not_reachable():
    conversation = Conversation(
        "fixture",
        "fixture:after",
        (
            Message("user", "retrieve"),
            Message("assistant", "The identifier is ZX-4917."),
            Message("tool", '{"id":"ZX-4917"}'),
        ),
    )
    edge = SupportEdge(
        "after", conversation.row_id, 2, 1, "ZX-4917", "ZX-4917", "tool_to_assistant"
    )
    rendered = render_qwen(conversation, CharacterTokenizer())
    view = materialize_nonpacked([rendered], len(rendered.input_ids), "keep_start")

    record = audit_edges_for_view(
        [conversation], [rendered], {conversation.row_id: [edge]}, view
    )[0]

    assert record["target_trained"]
    assert not record["source_reachable"]
    assert record["unsupported_target"]


def test_source_before_target_in_isolated_attention_group_is_not_reachable():
    conversation, edge = fixture()
    rendered = render_qwen(conversation, CharacterTokenizer())
    source = rendered.message_spans[edge.source_message]
    target = rendered.message_spans[edge.target_message]
    source_indices = list(range(source.block_start, source.block_end))
    target_indices = list(range(target.block_start, target.block_end))
    groups = [
        (0, [0] * len(source_indices), source_indices, [rendered.labels[i] for i in source_indices]),
        (1, [0] * len(target_indices), target_indices, [rendered.labels[i] for i in target_indices]),
    ]
    view = _view_from_groups(
        "isolated", 4096, groups, generated_examples=2, packed_lengths=[len(source_indices), len(target_indices)]
    )

    record = audit_edges_for_view(
        [conversation], [rendered], {conversation.row_id: [edge]}, view
    )[0]

    assert record["target_trained"]
    assert not record["source_reachable"]
    assert record["unsupported_target"]


def test_equal_text_with_wrong_row_provenance_is_not_reachable():
    first, edge = fixture()
    second = Conversation("fixture", "fixture:1", first.messages)
    rendered_first = render_qwen(first, CharacterTokenizer())
    rendered_second = render_qwen(second, CharacterTokenizer())
    source = rendered_second.message_spans[edge.source_message]
    target = rendered_first.message_spans[edge.target_message]
    source_indices = list(range(source.block_start, source.block_end))
    target_indices = list(range(target.block_start, target.block_end))
    rows = [1] * len(source_indices) + [0] * len(target_indices)
    indices = source_indices + target_indices
    labels = [rendered_second.labels[i] for i in source_indices] + [
        rendered_first.labels[i] for i in target_indices
    ]
    view = _view_from_groups(
        "wrong_row",
        4096,
        [(0, rows, indices, labels)],
        generated_examples=1,
        packed_lengths=[len(rows)],
    )

    record = audit_edges_for_view(
        [first, second],
        [rendered_first, rendered_second],
        {first.row_id: [edge]},
        view,
    )[0]

    assert record["target_trained"]
    assert not record["source_reachable"]
    assert record["unsupported_target"]


def test_case_sensitive_location_rejects_case_only_match():
    conversation, _ = fixture()
    rendered = render_qwen(conversation, CharacterTokenizer())

    assert locate_value(rendered, 2, "zx-4917", conversation.messages[2].content)
    assert not locate_value(
        rendered,
        2,
        "zx-4917",
        conversation.messages[2].content,
        case_sensitive=True,
    )
