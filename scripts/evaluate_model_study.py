#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from vsr.model_study import as_conversation, generation_prompt, load_records
from vsr.render import locate_value, render_qwen


ROOT = Path(__file__).resolve().parents[1]
VALUE = re.compile(r"VALUE=([A-Z0-9-]+)")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--split", required=True, choices=["dev", "test", "conflict"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--mps-fp16", action="store_true")
    args = parser.parse_args()
    if args.base_only == bool(args.adapter):
        parser.error("provide exactly one of --adapter or --base-only")
    config = json.loads((ROOT / "configs/model_study.json").read_text())
    split_path = ROOT / f"data/model_study/{args.split}.jsonl"
    records = load_records(split_path, args.maximum_records)
    output = ROOT / args.output_dir
    if output.exists() and any(output.rglob("*")):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_path = (ROOT / config["model_path"]).resolve()
    use_cuda = torch.cuda.is_available() and not args.use_cpu
    use_mps = torch.backends.mps.is_available() and not args.use_cpu and not use_cuda
    compute_dtype = torch.float16 if use_cuda or (use_mps and args.mps_fp16) else torch.float32
    tokenizer_source = model_path if args.base_only else args.adapter
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=compute_dtype
    )
    if not args.base_only:
        model = PeftModel.from_pretrained(model, args.adapter)
    device = torch.device("cpu") if args.use_cpu else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)
    model.eval()
    started = time.time()
    rows = []
    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start : batch_start + args.batch_size]
        prompts = [generation_prompt(record, tokenizer)[0] for record in batch_records]
        maximum_prompt = max(len(prompt) for prompt in prompts)
        prompt_ids = torch.full(
            (len(prompts), maximum_prompt), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        prompt_attention = torch.zeros_like(prompt_ids)
        for row_index, prompt in enumerate(prompts):
            prompt_ids[row_index, -len(prompt) :] = prompt.to(device)
            prompt_attention[row_index, -len(prompt) :] = 1
        with torch.inference_mode():
            generated = model.generate(
                prompt_ids,
                attention_mask=prompt_attention,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generations = [
            tokenizer.decode(row[maximum_prompt:], skip_special_tokens=True).strip()
            for row in generated
        ]

        rendered_batch = [render_qwen(as_conversation(record), tokenizer) for record in batch_records]
        maximum_full = max(len(rendered.input_ids) for rendered in rendered_batch)
        full_ids = torch.full(
            (len(rendered_batch), maximum_full),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        full_attention = torch.zeros_like(full_ids)
        for row_index, rendered in enumerate(rendered_batch):
            length = len(rendered.input_ids)
            full_ids[row_index, :length] = torch.tensor(
                rendered.input_ids, dtype=torch.long, device=device
            )
            full_attention[row_index, :length] = 1
        with torch.inference_mode():
            logits = model(input_ids=full_ids, attention_mask=full_attention).logits[:, :-1].float()
        log_probabilities = torch.log_softmax(logits, dim=-1)

        for row_index, (record, rendered, text) in enumerate(
            zip(batch_records, rendered_batch, generations)
        ):
            match = VALUE.search(text)
            prediction = match.group(1) if match else ""
            token_ids = full_ids[row_index]
            target_span = rendered.message_spans[-1]
            whole_positions = [
                position
                for position in range(max(1, target_span.block_start), target_span.block_end)
                if rendered.labels[position] != -100
            ]
            value_spans = locate_value(
                rendered,
                len(record["messages"]) - 1,
                record["target"],
                record["messages"][-1]["content"],
            )
            value_positions = sorted(
                {
                    position
                    for start, end in value_spans
                    for position in range(max(1, start), end)
                }
            )
            if not whole_positions or not value_positions:
                raise RuntimeError(f"target token localization failed: {record['record_id']}")

            def mean_nll(positions: list[int]) -> float:
                values = [
                    -log_probabilities[row_index, position - 1, token_ids[position]]
                    for position in positions
                ]
                return float(torch.stack(values).mean().item())

            rows.append(
                {
                    "record_id": record["record_id"],
                    "family": record["family"],
                    "target": record["target"],
                    "stale_value": record.get("stale_value"),
                    "prediction": prediction,
                    "raw_generation": text,
                    "exact": prediction == record["target"],
                    "copied_stale": bool(record.get("stale_value"))
                    and prediction == record["stale_value"],
                    "value_token_nll": mean_nll(value_positions),
                    "whole_target_nll": mean_nll(whole_positions),
                    "value_tokens": len(value_positions),
                    "whole_target_tokens": len(whole_positions),
                }
            )
        completed = min(batch_start + len(batch_records), len(records))
        if completed % 40 == 0 or completed == len(records):
            print(f"evaluated {completed}/{len(records)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "predictions.parquet", index=False)
    summary = frame.groupby("family", as_index=False).agg(
        records=("exact", "size"),
        exact=("exact", "mean"),
        copied_stale=("copied_stale", "mean"),
        value_token_nll=("value_token_nll", "mean"),
        whole_target_nll=("whole_target_nll", "mean"),
    )
    summary.loc[len(summary)] = [
        "ALL", len(frame), frame.exact.mean(), frame.copied_stale.mean(),
        frame.value_token_nll.mean(), frame.whole_target_nll.mean(),
    ]
    summary.to_csv(output / "summary.csv", index=False)
    manifest = {
        "status": "complete",
        "split": args.split,
        "split_sha256": digest(split_path),
        "records": len(frame),
        "base_only": args.base_only,
        "adapter": None if args.base_only else str(Path(args.adapter).resolve()),
        "batch_size": args.batch_size,
        "mps_fp16": bool(use_mps and args.mps_fp16),
        "exact": float(frame.exact.mean()),
        "copied_stale": float(frame.copied_stale.mean()),
        "value_token_nll": float(frame.value_token_nll.mean()),
        "whole_target_nll": float(frame.whole_target_nll.mean()),
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "seconds": time.time() - started,
        "paid_compute_cad": 0.0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
