from vsr.data import load_glaive
from vsr.support import discover_observation_edges


def test_glaive_fixture_parser_and_oracle(tmp_path):
    import pandas as pd

    path = tmp_path / "glaive.parquet"
    pd.DataFrame(
        {
            "system": ["SYSTEM: use tools"],
            "chat": [
                'USER: lookup x\n\nASSISTANT: call() <|endoftext|>\n\nFUNCTION RESPONSE: {"code":"A7Z91"}\n\nASSISTANT: The code is A7Z91 <|endoftext|>'
            ],
        }
    ).to_parquet(path)
    conversation = load_glaive(path)[0]
    assert [message.role for message in conversation.messages] == ["system", "user", "assistant", "tool", "assistant"]
    edges = discover_observation_edges(conversation)
    assert len(edges) == 1
    assert edges[0].source_value == "A7Z91"
