"""在 val set 上评估一个 LoRA adapter 的 graph 生成质量。

用法：
  python3 sdpo_vlm/tools/eval_val.py \
      --base Qwen/Qwen3-VL-8B-Instruct \
      --adapter outputs/prism_seed42/ckpt_step000100/student \
      --data data/oraclegraph \
      --val-split 0.98 --max-samples 100 \
      --out outputs/eval/prism.json \
      --device cuda:0 \
      --max-new-tokens 3072

支持 --adapter '' 跑 base model（不挂 adapter，作为 baseline of baselines）。
输出 JSON 包含逐样本 reward / per_type_pr / parsed，并打印 macro 汇总。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdpo_vlm.data.dataset import IsoGraphDataset
from sdpo_vlm.data.prompt import build_student_messages
from sdpo_vlm.feedback.metrics import compute_reward
from sdpo_vlm.tools.score_predictions import score_records


def load_model(base_path: str, adapter_path: Optional[str], device: str,
               attn_impl: str = "sdpa",
               device_map: Optional[str] = None,
               max_memory: Optional[Dict] = None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs = dict(dtype=torch.bfloat16, trust_remote_code=False,
                  attn_implementation=attn_impl)
    if device_map is not None:
        kwargs["device_map"] = device_map
        if max_memory:
            kwargs["max_memory"] = max_memory
    base = AutoModelForImageTextToText.from_pretrained(base_path, **kwargs)
    if device_map is None:
        base = base.to(device)
    base.eval()
    proc = AutoProcessor.from_pretrained(base_path, trust_remote_code=False)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_path,
                                          adapter_name="evalrun", is_trainable=False)
        model.set_adapter("evalrun")
        model.eval()
    else:
        model = base
    return model, proc


@torch.no_grad()
def generate_one(model, proc, image_path: str, device: str,
                 max_new_tokens: int, temperature: float, top_p: float,
                 repetition_penalty: float = 1.0,
                 no_repeat_ngram_size: int = 0):
    msgs = build_student_messages(image_path)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = proc(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    gen_kwargs = dict(
        do_sample=(temperature > 0),
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=False,
        use_cache=True,
        repetition_penalty=repetition_penalty,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    out = model.generate(**inputs, **gen_kwargs)
    gen = out.sequences[0, input_len:].tolist()
    eos = proc.tokenizer.eos_token_id
    if eos is not None and eos in gen:
        gen = gen[:gen.index(eos)]
    return proc.tokenizer.decode(gen, skip_special_tokens=True), len(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="",
                    help="LoRA adapter path (空 = base model)")
    ap.add_argument("--data", required=True,
                    help="dataset root (含 images/ 和 json/)")
    ap.add_argument("--val-split", type=float, default=0.98,
                    help="train ratio；val = 1 - this")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42,
                    help="Dataset split seed; match data.split_seed, not train.seed")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default=None,
                    help="None | 'balanced'，多卡跑大 max_new_tokens")
    ap.add_argument("--max-memory", default=None,
                    help="JSON dict, e.g. '{\"0\":\"85GiB\",\"1\":\"85GiB\"}'")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="温和惩罚重复 token，防止 greedy 陷入循环")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=20,
                    help="禁止 N-gram 重复（0=关闭，20 = 防长循环且不影响节点描述）")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run",
                    help="标识此次评估的 label，写入 JSON")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="only evaluate ds[start_idx:end_idx] for shard-parallel eval")
    ap.add_argument("--end-idx", type=int, default=-1,
                    help="exclusive end; -1 = len(ds)")
    args = ap.parse_args()

    max_memory = json.loads(args.max_memory) if args.max_memory else None
    if max_memory:
        max_memory = {int(k) if k.isdigit() else k: v for k, v in max_memory.items()}

    print(f"[eval] label={args.label}", flush=True)
    print(f"[eval] base={args.base}", flush=True)
    print(f"[eval] adapter={args.adapter or '(none, base model)'}", flush=True)
    print(f"[eval] data={args.data}", flush=True)
    print(f"[eval] val_split={args.val_split} max_samples={args.max_samples} seed={args.seed}",
          flush=True)
    print(f"[eval] device={args.device} device_map={args.device_map} "
          f"max_memory={max_memory}", flush=True)

    # ---- 加载 val ----
    ds = IsoGraphDataset(
        roots=[args.data],
        split="val",
        train_val_split=args.val_split,
        max_samples=args.max_samples,
        keep_full_polygon=False,
        seed=args.seed,
    )
    print(f"[eval] val samples = {len(ds)}", flush=True)

    # ---- 加载模型 ----
    print(f"[eval] loading model ...", flush=True)
    model, proc = load_model(
        args.base, args.adapter or None, args.device,
        attn_impl=args.attn, device_map=args.device_map, max_memory=max_memory,
    )
    print(f"[eval] model ready", flush=True)

    # ---- 跑 ----
    results = []
    rewards = []
    macro_ps = []
    macro_rs = []
    parsed_ok = 0
    n_truncated = 0
    resp_lens = []
    t0 = time.time()
    start_idx = max(0, args.start_idx)
    end_idx = len(ds) if args.end_idx < 0 else min(len(ds), args.end_idx)
    if start_idx >= end_idx:
        raise ValueError("The requested evaluation shard is empty")
    print(f"[eval] shard range=[{start_idx}, {end_idx}) of {len(ds)}", flush=True)
    for i in range(start_idx, end_idx):
        item = ds[i]
        try:
            pred, gen_len = generate_one(
                model, proc, item["image_path"], args.device,
                args.max_new_tokens, args.temperature, args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[eval][{i}] OOM skip", flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[eval][{i}] generate FAIL: {e}", flush=True)
            continue

        try:
            res = compute_reward(item["gold_short"], pred)
        except Exception as e:
            print(f"[eval][{i}] compute_reward FAIL: {type(e).__name__}: {e}  (counted as unparsed, reward=0)", flush=True)
            res = {
                "reward": 0.0, "macro_p": 0.0, "macro_r": 0.0,
                "parsed": False, "per_type_pr": {},
            }
        rewards.append(res["reward"])
        macro_ps.append(res["macro_p"])
        macro_rs.append(res["macro_r"])
        resp_lens.append(gen_len)
        if res["parsed"]:
            parsed_ok += 1
        if gen_len >= args.max_new_tokens - 2:
            n_truncated += 1
        results.append({
            "image_id": item["image_id"],
            "reward": res["reward"],
            "macro_p": res["macro_p"],
            "macro_r": res["macro_r"],
            "parsed": res["parsed"],
            "resp_len": gen_len,
            "per_type_pr": {k: list(v) for k, v in res["per_type_pr"].items()},
            "pred": pred,
            "gold_short": item["gold_short"],
        })
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"[eval][{i+1}/{len(ds)}] reward_mean={statistics.mean(rewards):.4f} "
                  f"parsed%={100*parsed_ok/len(rewards):.0f} "
                  f"resp_len_mean={statistics.mean(resp_lens):.0f} "
                  f"elapsed={elapsed:.0f}s", flush=True)
            # Incremental checkpoint: dump partial results every 5 samples so
            # a crash halfway through can be recovered from the .partial.json
            try:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                with open(args.out + ".partial.json", "w") as _f:
                    json.dump({
                        "partial": True,
                        "n": len(rewards),
                        "running_reward_mean": statistics.mean(rewards),
                        "results": results,
                    }, _f, ensure_ascii=False)
            except Exception as _e:
                print(f"[eval] partial dump warn: {_e}", flush=True)

    # ---- 汇总 ----
    summary = {
        "label": args.label,
        "n": len(rewards),
        "expected_n": end_idx - start_idx,
        "complete": len(rewards) == end_idx - start_idx,
        "skipped_n": end_idx - start_idx - len(rewards),
        "reward_mean": statistics.mean(rewards) if rewards else 0,
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0,
        "macro_p_mean": statistics.mean(macro_ps) if macro_ps else 0,
        "macro_r_mean": statistics.mean(macro_rs) if macro_rs else 0,
        "parsed_pct": 100 * parsed_ok / len(rewards) if rewards else 0,
        "truncated_pct": 100 * n_truncated / len(rewards) if rewards else 0,
        "resp_len_mean": statistics.mean(resp_lens) if resp_lens else 0,
        "resp_len_p50": sorted(resp_lens)[len(resp_lens)//2] if resp_lens else 0,
        "resp_len_p90": sorted(resp_lens)[int(len(resp_lens)*0.9)] if resp_lens else 0,
        "max_reward": max(rewards) if rewards else 0,
        "args": vars(args),
    }

    # 节点/边 micro F1（汇总所有样本的 tp/fp/fn）
    bucket_tot: Dict[str, List[int]] = {}
    for r in results:
        for k, (p, rr, tp, fp, fn) in r["per_type_pr"].items():
            if k not in bucket_tot:
                bucket_tot[k] = [0, 0, 0]
            bucket_tot[k][0] += tp
            bucket_tot[k][1] += fp
            bucket_tot[k][2] += fn
    micro = {}
    for k, (tp, fp, fn) in bucket_tot.items():
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        micro[k] = {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}
    summary["micro_per_type"] = micro
    if results:
        summary["structural"] = score_records(results)["summary"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n=== EVAL SUMMARY ({args.label}) ===")
    print(f"  n={summary['n']}")
    print(f"  reward     mean={summary['reward_mean']:.4f}  std={summary['reward_std']:.4f}  max={summary['max_reward']:.4f}")
    print(f"  macro P/R  P={summary['macro_p_mean']:.4f}  R={summary['macro_r_mean']:.4f}")
    print(f"  parsed%    {summary['parsed_pct']:.1f}")
    print(f"  truncated% {summary['truncated_pct']:.1f}")
    print(f"  resp_len   mean={summary['resp_len_mean']:.0f}  p50={summary['resp_len_p50']}  p90={summary['resp_len_p90']}")
    print(f"  micro per type:")
    for k in sorted(micro):
        m = micro[k]
        print(f"    {k:14s}  P={m['P']:.3f}  R={m['R']:.3f}  F1={m['F1']:.3f}  "
              f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']})")
    print(f"\n[eval] saved → {args.out}")
    if not summary["complete"]:
        raise SystemExit("Evaluation incomplete: rerun the failed pages before reporting metrics")


if __name__ == "__main__":
    main()
