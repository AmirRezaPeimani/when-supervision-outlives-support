# Data provenance

The main corpus audit uses three public Apache-2.0 datasets. Machine-readable identifiers, revisions, URLs, byte counts, and SHA-256 digests are in `provenance/datasets.json`.

| Corpus | Public identifier | Revision | Rows used |
|---|---|---|---:|
| ToolACE | `Team-ACE/ToolACE` | `6bda777c88d21e5a204703c1ee45597a8fa4f734` | 11,300 |
| ReTool | `swordfaith/ReTool-SFT-multi-turn` | `13eb7a396284caa114d677af3d071864c27ba5cc` | 2,000 |
| Glaive-FC | `glaiveai/glaive-function-calling-v2` | `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` | 5,000 sampled from 39,591 eligible rows |

ToolACE and ReTool use their complete pinned Parquet releases. Their frozen primary support edges are included at `outputs/raw/audit_support_edges.jsonl` (SHA-256 `638615ea59466ae4e5401c5887d8ed56fdcb8f22200332124ba3d7717b85096a`).

The Glaive sample is selected by the lowest SHA-256 ranks of `20260808:row_id` among rows containing at least one precision-first observation-to-assistant edge. The included processed subset has SHA-256 `06c3adf24f8979c1e9f475754ab958a1c377504aa14590ee9f71a81ad98b8cd5`; its 15,541 support edges have SHA-256 `d84c29433cac4e70102ca2cd8eaaa3f01461e7748d766db4a426a662f547c562`.

The controlled-model records in `data/model_study/` are deterministically generated synthetic examples. Their split and corruption invariants are executable in `tests/test_model_study.py`; their hashes are recorded in `configs/model_study_revision2_frozen_manifest.json`.

No benchmark test data is used for training. The controlled train, development, clean test, and conflict sets use disjoint identifiers and target values.
