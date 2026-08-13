# Reproducibility notes

The exact lightweight verification path is:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/verify_artifact.py
```

The first command validates implementation invariants. The second validates the included frozen results against the paper-facing counts and rates.

Full ToolACE and ReTool reruns require their public Parquet files and the pinned Qwen tokenizer:

```bash
.venv/bin/python scripts/download_data.py toolace retool
.venv/bin/python scripts/download_tokenizer.py
```

Each main runner refuses to overwrite a nonempty output directory. Use a new output path for reproduction. Scientific configurations preserve the final policies, budgets, seeds, oracle case rule, model revision, and training settings; only repository-relative acquisition paths differ from the original machine-local layout.

The controlled-model study is optional for artifact smoke testing. Its deterministic input records, training/evaluation scripts, aggregate outputs, and frozen hash manifest are included. Reproducing the full matrix requires the complete Qwen2.5-0.5B-Instruct checkpoint, while corpus auditing requires only the tokenizer files downloaded by `scripts/download_tokenizer.py`.
