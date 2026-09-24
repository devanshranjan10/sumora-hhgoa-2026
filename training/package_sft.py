#!/usr/bin/env python3
"""Package the fine-tuned artifact for the 16GB Mac M1 (spec section 6b).

1. Merge the LoRA adapter into the base model (bf16).
2. Save the merged model + tokenizer to artifacts/merged/.
3. Export a q4_k_m GGUF (llama.cpp quantization ready) if llama-quantize
   is available; otherwise leave the merged bf16 model as the artifact.

A q4_k_m 14B GGUF is ~9GB - fits the 16GB M1 constraint.
"""
import subprocess
from pathlib import Path

ADAPTER = Path("/mnt/sih26-train/sumora/runs/sft/final_adapter")
BASE = "Qwen/Qwen3-14B"
OUT = Path("/mnt/sih26-train/sumora/artifacts")


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    print("loading base + adapter for merge...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    merged = model.merge_and_unload()
    merged_dir = OUT / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.save_pretrained(merged_dir)
    print(f"merged model saved to {merged_dir}", flush=True)

    # GGUF conversion + quantization if the llama.cpp tools are present.
    convert = "/mnt/sih26-train/sumora/llama.cpp/convert_hf_to_gguf.py"
    quant = "/mnt/sih26-train/sumora/llama.cpp/build/bin/llama-quantize"
    if Path(convert).exists() and Path(quant).exists():
        f16 = OUT / "sumora-sft-f16.gguf"
        q4 = OUT / "sumora-sft-q4_k_m.gguf"
        subprocess.run([sys.executable, convert, str(merged_dir), "--outfile", str(f16)],
                       check=True)
        subprocess.run([quant, str(f16), str(q4), "q4_k_m"], check=True)
        print(f"GGUF written: {q4}", flush=True)
    else:
        print("llama.cpp tools not present - merged bf16 model is the artifact", flush=True)
    (OUT / "PACKAGED.txt").write_text("merge complete - see merged/ and *.gguf\n")
    print("PACKAGE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
