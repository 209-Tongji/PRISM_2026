"""SFT baseline trainer：只在 (image, gold_json) 上做监督微调，无 rollout 无 RL。
用作 PRISM/SDPO/DPO/GRPO 的监督训练对照。
复用 load_training_model（但 teacher adapter 忽略不用）。
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdpo_vlm.data.dataset import IsoGraphDataset, simple_collate
from sdpo_vlm.data.prompt import build_student_messages, find_response_start
from sdpo_vlm.data.schema import short_to_json_str
from sdpo_vlm.trainer.trainer import (
    load_config, load_training_model, prepare_forward_inputs,
    _make_lr_lambda, _repeat_iter,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg["train"].get("seed", 42))
    torch.manual_seed(cfg["train"].get("seed", 42))

    out_dir = Path(cfg["train"]["output_dir"]) / cfg["train"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # ---- data ----
    ds_train = IsoGraphDataset(
        roots=cfg["data"]["root"],
        split="train",
        train_val_split=cfg["data"]["train_val_split"],
        max_samples=cfg["data"].get("max_train_samples"),
        keep_full_polygon=cfg["data"].get("keep_full_polygon", False),
        seed=cfg["data"].get("split_seed", cfg["train"].get("seed", 42)),
    )
    print(f"[data] train samples = {len(ds_train)}", flush=True)
    dl = DataLoader(ds_train, batch_size=1, shuffle=True,
                     collate_fn=simple_collate, num_workers=0)

    # ---- model ----
    model, processor = load_training_model(cfg, device=args.train_device)
    model.set_adapter("student")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {trainable/1e6:.2f}M / {total/1e9:.2f}B "
          f"on {args.train_device}", flush=True)

    # ---- optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, _make_lr_lambda(
            cfg["train"].get("lr_schedule", "cosine"),
            cfg["train"]["warmup_steps"],
            cfg["train"]["max_steps"],
        ),
    )

    # ---- main loop ----
    start_step = int(cfg["train"].get("start_step", 0))
    step = start_step
    max_steps = cfg["train"]["max_steps"]
    log_interval = cfg["train"]["log_interval"]
    save_interval = cfg["train"]["save_interval"]
    grad_clip = cfg["train"]["grad_clip"]
    # Fast-forward LR scheduler so it continues from the correct point after resume
    for _ in range(start_step):
        sched.step()
    if start_step > 0:
        print(f"[resume] starting at step={start_step}, "
              f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

    data_iter = _repeat_iter(dl)
    optim.zero_grad()
    t0 = time.time()
    running = {"loss": 0.0, "tok_cnt": 0, "n": 0}

    while step < max_steps:
        batch = next(data_iter)[0]
        image_path = batch["image_path"]
        gold_short = batch["gold_short"]
        gold_json = short_to_json_str(gold_short)

        student_msgs = build_student_messages(image_path)

        try:
            inputs, ids, resp_len = prepare_forward_inputs(
                processor, student_msgs, gold_json, image_path,
                args.train_device,
            )
            out = model(**inputs, use_cache=False)
            logits = out.logits  # [1, L, V]

            # 只对 response 段做 CE
            start = find_response_start(processor, ids)
            # logits[t] → 预测 ids[t+1]
            # response tokens 位置 = [start, start+resp_len)
            # 对应 logits index = [start-1, start-1+resp_len)
            target_ids = torch.tensor(ids[start:start+resp_len], dtype=torch.long,
                                       device=logits.device)
            resp_logits = logits[0, start-1:start-1+resp_len, :]
            R = min(resp_logits.shape[0], target_ids.shape[0])
            loss = F.cross_entropy(resp_logits[:R].float(), target_ids[:R])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optim.step(); sched.step(); optim.zero_grad()

            step += 1
            running["loss"] += loss.item()
            running["tok_cnt"] += R
            running["n"] += 1

            if step % log_interval == 0:
                elapsed = time.time() - t0
                avg_loss = running["loss"] / running["n"]
                avg_tok = running["tok_cnt"] / running["n"]
                # 兼容 plot_curves.py：以 reward=exp(-loss) 当作粗略度量（不是真 reward）
                # 不，为了避免混淆，reward 字段就写 0
                print(f"step={step} | elapsed={elapsed:.1f}s | loss={avg_loss:.4f} "
                      f"| reward=0.0000 | resp_len={avg_tok:.1f} | "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
                running = {"loss": 0.0, "tok_cnt": 0, "n": 0}

            if step % save_interval == 0:
                save_path = out_dir / f"ckpt_step{step:06d}"
                model.save_pretrained(str(save_path))
                print(f"[save] {save_path}", flush=True)

        except torch.cuda.OutOfMemoryError:
            # IMPORTANT: free tensors from the failed iteration BEFORE continuing,
            # otherwise backward() graph + activations stay alive and the next
            # iteration OOMs on the same batch forever (death loop at step 784).
            try:
                del out, logits, resp_logits, target_ids, loss
            except Exception:
                pass
            try:
                del inputs, ids
            except Exception:
                pass
            optim.zero_grad(set_to_none=True)
            import gc; gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[OOM] skip step {step}, batch={image_path}", flush=True)
            continue
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"[fwd] FAIL skip: {exc}", flush=True)
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

    print("[done] training complete.")


if __name__ == "__main__":
    main()
