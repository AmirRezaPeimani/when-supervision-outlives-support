#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from vsr.model_study import CausalPaddingCollator, load_records, tokenize_training_records


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption", type=int, required=True, choices=[0, 25, 50, 100])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-train-records", type=int)
    parser.add_argument("--maximum-dev-records", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--mps-fp16", action="store_true")
    parser.add_argument("--recipe-revision", type=int, default=0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config_path = ROOT / "configs/model_study.json"
    config = json.loads(config_path.read_text())
    seed = args.seed if args.seed is not None else config["seed"]
    batch_size = args.batch_size or config["batch_size"]
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps or config["gradient_accumulation_steps"]
    )
    model_path = (ROOT / config["model_path"]).resolve()
    train_path = ROOT / f"data/model_study/train_corruption_{args.corruption}.jsonl"
    dev_path = ROOT / "data/model_study/dev.jsonl"
    output = ROOT / args.output_dir
    if output.exists() and any(output.rglob("*")):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_records = load_records(train_path, args.maximum_train_records)
    dev_records = load_records(dev_path, args.maximum_dev_records)
    train_dataset = tokenize_training_records(train_records, tokenizer, config["max_length"])
    dev_dataset = tokenize_training_records(dev_records, tokenizer, config["max_length"])
    use_cuda = torch.cuda.is_available() and not args.use_cpu
    use_mps = torch.backends.mps.is_available() and not args.use_cpu and not use_cuda
    compute_dtype = torch.float16 if use_cuda or (use_mps and args.mps_fp16) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=compute_dtype
    )
    lora = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs or config["epochs"],
        max_steps=args.max_steps,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # The scientific gate uses exact generation in evaluate_model_study.py. A second
        # teacher-forced dev pass inside Trainer cannot affect weights and adds substantial MPS
        # runtime, so all post-gate matrix runs omit it.
        eval_strategy="no",
        save_strategy="epoch" if args.max_steps < 0 else "no",
        save_total_limit=1,
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        use_cpu=args.use_cpu,
        fp16=use_cuda,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=CausalPaddingCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train()
    adapter = output / "adapter"
    trainer.model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    manifest = {
        "status": "complete",
        "corruption_percent": args.corruption,
        "recipe_revision": args.recipe_revision,
        "optimization_seed": seed,
        "epochs": args.epochs or config["epochs"],
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "trainer_internal_dev_evaluation": False,
        "mps_fp16": bool(use_mps and args.mps_fp16),
        "train_file_sha256": digest(train_path),
        "dev_file_sha256": digest(dev_path),
        "config_sha256": digest(config_path),
        "trainable_parameters": trainable,
        "total_parameters_with_adapter": total,
        "training_metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "seconds": time.time() - started,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "device": str(trainer.args.device),
        "compute_dtype": str(compute_dtype),
        "paid_compute_cad": 0.0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
