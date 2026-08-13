#!/usr/bin/env python3
"""Run the sealed corruption matrix after the prospective development gate passes.

This driver is intentionally conservative: it refuses to inspect test/conflict splits unless a
completed development manifest records the frozen gate, runs every missing training condition
before any sealed evaluation, and never overwrites a nonempty output directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORRUPTIONS = (0, 25, 50, 100)
SEEDS = (20260808, 20260809, 20260810)


def completed(path: Path) -> bool:
    if not (path / "manifest.json").exists():
        return False
    return json.loads((path / "manifest.json").read_text()).get("status") == "complete"


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-manifest",
        default="outputs/model_study/eval_dev_corruption0_revision2_seed20260808/manifest.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/model_study.json").read_text())
    gate_path = ROOT / args.gate_manifest
    if not gate_path.exists():
        raise FileNotFoundError(f"development gate manifest missing: {gate_path}")
    gate = json.loads(gate_path.read_text())
    threshold = float(config["learnability_gate_exact"])
    if gate.get("split") != "dev" or gate.get("status") != "complete":
        raise RuntimeError("gate manifest is not a completed development evaluation")
    if float(gate.get("exact", -1)) < threshold:
        raise RuntimeError(f"development exact {gate.get('exact')} is below frozen gate {threshold}")

    # Finish every training run before the first sealed evaluation.
    for corruption in CORRUPTIONS:
        for seed in SEEDS:
            output = ROOT / f"outputs/model_study/corruption_{corruption}_revision2_seed{seed}"
            if completed(output):
                print(f"SKIP completed training {output.relative_to(ROOT)}", flush=True)
                continue
            if output.exists() and any(output.rglob("*")):
                raise RuntimeError(f"incomplete nonempty training output requires review: {output}")
            run(
                [
                    sys.executable,
                    "scripts/train_model_study.py",
                    "--corruption",
                    str(corruption),
                    "--seed",
                    str(seed),
                    "--recipe-revision",
                    "2",
                    "--mps-fp16",
                    "--output-dir",
                    str(output.relative_to(ROOT)),
                ]
            )

    for corruption in CORRUPTIONS:
        for seed in SEEDS:
            adapter = ROOT / f"outputs/model_study/corruption_{corruption}_revision2_seed{seed}/adapter"
            for split in ("test", "conflict"):
                output = ROOT / f"outputs/model_study/eval_{split}_corruption{corruption}_seed{seed}"
                if completed(output):
                    print(f"SKIP completed evaluation {output.relative_to(ROOT)}", flush=True)
                    continue
                if output.exists() and any(output.rglob("*")):
                    raise RuntimeError(f"incomplete nonempty evaluation output requires review: {output}")
                run(
                    [
                        sys.executable,
                        "scripts/evaluate_model_study.py",
                        "--adapter",
                        str(adapter),
                        "--split",
                        split,
                        "--batch-size",
                        str(args.batch_size),
                        "--mps-fp16",
                        "--output-dir",
                        str(output.relative_to(ROOT)),
                    ]
                )

    if not args.skip_base:
        for split in ("test", "conflict"):
            output = ROOT / f"outputs/model_study/eval_{split}_base"
            if completed(output):
                print(f"SKIP completed base evaluation {output.relative_to(ROOT)}", flush=True)
                continue
            if output.exists() and any(output.rglob("*")):
                raise RuntimeError(f"incomplete nonempty base output requires review: {output}")
            run(
                [
                    sys.executable,
                    "scripts/evaluate_model_study.py",
                    "--base-only",
                    "--split",
                    split,
                    "--batch-size",
                    str(args.batch_size),
                    "--mps-fp16",
                    "--output-dir",
                    str(output.relative_to(ROOT)),
                ]
            )


if __name__ == "__main__":
    main()
