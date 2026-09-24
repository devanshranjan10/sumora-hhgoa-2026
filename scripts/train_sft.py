#!/usr/bin/env python3
"""Sumora QLoRA SFT (spec 6b, tier 2).

Archived Qwen2.5-3B-Instruct experiment (NF4-quantized, bf16 LoRA) trained
on teacher traces. The deployed decision model is gradient boosting; the
later Qwen3-14B experiment is documented in eval/sft/NOTES.md.
"""
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen3-14B"
RUNS = Path("/mnt/sih26-train/sumora/runs")
TRACES = Path("/mnt/sih26-train/sumora/traces/traces_norm.jsonl")
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def load_traces() -> Dataset:
    rows = []
    for line in TRACES.read_text().splitlines():
        rec = json.loads(line)
        user = rec["messages"][0]["content"]
        asst = rec["messages"][1]["content"]
        rows.append({"prompt": user, "completion": asst})
    print(f"traces: {len(rows)}", flush=True)
    return Dataset.from_list(rows)


def main() -> int:
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"
    tok.pad_token = tok.eos_token

    ds = load_traces()
    split = ds.train_test_split(test_size=0.15, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=quant, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=str(RUNS / "sft"),
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=20,
        logging_steps=5,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        max_length=8192,
        completion_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        seed=42,
        dataset_num_proc=2,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("EVAL_METRICS", json.dumps(metrics), flush=True)
    trainer.save_model(str(RUNS / "sft" / "final_adapter"))
    tok.save_pretrained(str(RUNS / "sft" / "final_adapter"))
    (RUNS / "sft" / "final_adapter" / "EVAL_METRICS.json").write_text(json.dumps(metrics, indent=2))
    print("SFT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
