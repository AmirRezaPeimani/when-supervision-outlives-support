from vsr.schema_repair import (
    _modified_conversation,
    extract_inline_schema,
    filter_schemas,
    invocation_requirements,
)
from vsr.types import Conversation, Message


def test_inline_schema_extraction_and_invoked_filtering():
    text = (
        "Rules. Here is a list of functions in JSON format that you can invoke:\n"
        '[{"name":"alpha","parameters":{"type":"object","properties":{"x":{"type":"string"},"y":{"type":"integer"}},"required":["x","y"]}},'
        '{"name":"beta","parameters":{"type":"object","properties":{},"required":[]}}] Tail.'
    )
    bundle = extract_inline_schema(text)
    assert bundle is not None
    requirements = invocation_requirements(Message("assistant", '[alpha(x="v")]'))
    assert requirements == {"alpha": {"x"}}
    invoked = filter_schemas(bundle.schemas, requirements, parameter_slice=False)
    sliced = filter_schemas(bundle.schemas, requirements, parameter_slice=True)
    assert [item["name"] for item in invoked] == ["alpha"]
    assert list(sliced[0]["parameters"]["properties"]) == ["x"]
    assert sliced[0]["parameters"]["required"] == ["x"]


def test_structured_tools_are_used_when_system_has_no_inline_schema():
    conversation = Conversation(
        "fixture",
        "fixture:1",
        (Message("system", "be helpful"), Message("assistant", "call")),
        metadata={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "alpha",
                        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                    },
                }
            ]
        },
    )
    modified, reason = _modified_conversation(conversation, {"alpha": {"x"}}, "invoked_tool")
    assert reason == ""
    assert modified is not None
    assert len(modified.metadata["tools"]) == 1
