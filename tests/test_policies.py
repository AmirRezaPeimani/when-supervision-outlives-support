from vsr.policies import groups_covering_span, materialize_nonpacked, materialize_packed
from vsr.types import MessageSpan, RenderedConversation


def rendered(length: int, supervised_start: int) -> RenderedConversation:
    ids = tuple(range(length))
    labels = tuple(value if index >= supervised_start else -100 for index, value in enumerate(ids))
    return RenderedConversation(
        ids,
        labels,
        (MessageSpan(0, "assistant", 0, supervised_start, length, length),),
        "fixture",
        (((0, length),),),
    )


def test_keep_start_drops_fully_masked_row_like_current_trl():
    view = materialize_nonpacked([rendered(12, 9)], 6, "keep_start")
    assert view.generated_examples == 0
    assert not view.rows_with_supervision


def test_keep_end_retains_supervised_suffix():
    view = materialize_nonpacked([rendered(12, 9)], 6, "keep_end")
    assert view.generated_examples == 1
    assert view.effective_supervised_tokens == 3
    assert groups_covering_span(view, 0, (9, 12))


def test_bfd_truncates_overflow_without_splitting():
    view = materialize_packed([rendered(12, 9)], 6, "bfd")
    assert view.retained_original_tokens == 6
    assert not groups_covering_span(view, 0, (9, 12))


def test_bfd_split_preserves_tokens_but_separates_attention_groups():
    view = materialize_packed([rendered(12, 9)], 6, "bfd_split")
    assert view.retained_original_tokens == 12
    left = groups_covering_span(view, 0, (1, 3))
    right = groups_covering_span(view, 0, (9, 12))
    assert left and right and left.isdisjoint(right)


def test_wrapped_mid_sequence_chunking():
    view = materialize_packed([rendered(12, 9)], 6, "wrapped")
    assert view.retained_original_tokens == 12
    assert groups_covering_span(view, 0, (9, 12))
    assert not groups_covering_span(view, 0, (4, 8))

