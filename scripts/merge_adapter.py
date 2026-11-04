"""Merge a LoRA adapter into base model weights and save as a new full model.

Usage:
  python3 scripts/merge_adapter.py \
    --base Qwen/Qwen3-VL-8B-Instruct \
    --adapter outputs/prism_seed42/ckpt_step000100/student \
    --out outputs/prism_merged
"""
import argparse
import torch
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    print(f"[merge] loading base: {args.base}")
    base = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation=args.attn,
    )
    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=False)

    print(f"[merge] loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter,
                                       adapter_name="merged", is_trainable=False)
    model.set_adapter("merged")

    print(f"[merge] merging into base weights ...")
    merged = model.merge_and_unload()  # returns the underlying base with LoRA weights merged in

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving to {out_path}")
    merged.save_pretrained(str(out_path), safe_serialization=True)
    proc.save_pretrained(str(out_path))
    print(f"[merge] done")


if __name__ == "__main__":
    main()
