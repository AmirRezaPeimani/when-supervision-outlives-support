# Training on Broken Contexts: Source Loss and Repair in Tool-Augmented LLM Post-Training

Research repository for **“Training on Broken Contexts: Source Loss and Repair in Tool-Augmented LLM Post-Training.”** The project tracks rendered tokens through sequence construction to test whether matched assistant targets remain with their recorded tool observations. It includes grouped training-example reconstruction, tool-call schema reduction, packing-conformance checks, and a controlled source-removal experiment.

## Key results

At a 512-token budget, BFD-split leaves 353/913 ToolACE targets separated from their recorded sources (38.66%), 315/1,606 ReTool targets separated from their recorded sources (19.61%), and 431/6,698 Glaive-FC targets separated from their recorded sources (6.43%). ToolACE grouped reconstruction retains 875/918 matched targets without duplicating source tokens. For ToolACE tool calls, retaining a referenced parameters preserves 73.15% of targets, compared with 28.60% when retaining the full schema.

For BFD and BFD-split, each emitted `seq_lengths` fragment is treated as a separate causal-attention group; wrapped packing treats each fixed-length packed block as one attention group.

## Repository structure

- `src/vsr/`: rendering, provenance, policy simulation, auditing, training-example reconstruction, and schema reduction.
- `scripts/`: public data/model acquisition, analyses, figure/table generation, conformance probes, and controlled-model utilities.
- `configs/`: frozen scientific configurations with repository-relative paths.
- `tests/`: focused regression tests, including causal source-before-target validation.
- `analysis/` and `outputs/`: compact frozen evidence for the reported corpus, sensitivity, reconstruction, schema-coverage, conformance, and model-study results.
- `data/`: the released Glaive-FC evaluation subset and deterministic controlled-study records; ToolACE and ReTool are downloaded separately.
- `figures/` and `tables/`: final paper-facing outputs.
- `paper/`: the current manuscript, bibliography, and required Springer files; assets are in the repository figure and table directories.
- `provenance/`: exact public identifiers, revisions, hashes, and license information.

## Installation

Python 3.13 is the verified environment; Python 3.10 or newer is supported.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
```

Verify the checked-in files with:

```bash
shasum -a 256 -c ARTIFACT_MANIFEST.sha256
```

## Data and model acquisition

The study uses the public datasets `Team-ACE/ToolACE`, `swordfaith/ReTool-SFT-multi-turn`, and `glaiveai/glaive-function-calling-v2`, plus `Qwen/Qwen2.5-0.5B-Instruct`. Exact revisions and SHA-256 digests are recorded in `provenance/` and summarized in `DATA.md`.

```bash
.venv/bin/python scripts/download_data.py toolace retool
.venv/bin/python scripts/download_tokenizer.py
```

The included Glaive-FC subset can be reconstructed from its public release with:

```bash
.venv/bin/python scripts/download_data.py glaive
PYTHONPATH=src .venv/bin/python scripts/prepare_glaive.py
```

No private credentials are required.

## Principal reproduction commands

ToolACE and ReTool corpus audit:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase1.py \
  --config configs/primary_audit.json \
  --output-dir outputs/raw/primary_audit
PYTHONPATH=src .venv/bin/python scripts/analyze_phase1.py \
  --input-dir outputs/raw/primary_audit \
  --output-dir analysis/corpus_audit/toolace_retool_reproduced
```

Glaive-FC corpus audit:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase1_glaive.py \
  --config configs/glaive.json \
  --output-dir outputs/raw/glaive_audit
PYTHONPATH=src .venv/bin/python scripts/analyze_phase1.py \
  --input-dir outputs/raw/glaive_audit \
  --output-dir analysis/corpus_audit/glaive_reproduced
```

Grouped reconstruction and schema coverage:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_materialization.py \
  --config configs/materialization.json \
  --output-dir outputs/materialization_reproduced
PYTHONPATH=src .venv/bin/python scripts/run_action_repair.py \
  --config configs/action_repair.json \
  --output-dir outputs/schema_repair_reproduced
```

Lightweight verification of the frozen headline outputs:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_artifact.py
```

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The focused suite covers exact provenance, attention-group isolation, causal source-before-target ordering, complete-target reconstruction, schema validity, split separation, and the documented TRL 1.9.2 wrapped behavior.

## Compute and runtime

The focused tests and frozen-output verification are lightweight CPU jobs. Full corpus rendering and bootstrap analyses are CPU-only but can take several hours. The optional controlled-model matrix trains LoRA adapters for Qwen2.5-0.5B-Instruct and was run with float16 on Apple MPS. No paid compute is required for the corpus audits or reconstruction analyses.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The paper
citation should be used when the manuscript is available; the software entry
may be used for the code artifact.

## License and data notes

The repository's original code and documentation are licensed under the
Apache License 2.0. Third-party datasets and models remain governed by their
upstream terms. The repository does not redistribute ToolACE, ReTool, or model
weights. See [`LICENSES.md`](LICENSES.md) and [`DATA.md`](DATA.md) for details.
