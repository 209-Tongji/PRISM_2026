"""统一 baseline trainer：支持 algo=dpo|grpo（PPO+组内归一化）。

与 SDPO trainer 的区别：
- 只挂一个 LoRA adapter（student），ref / old policy 用 base model（关 adapter）
- DPO：每 step 取 G 个 rollout，按 reward 分 winner/loser，做 pair loss
- GRPO：组内 reward → 组归一化 advantage → token-level PPO clipped CE
- 所有日志格式与 SDPO trainer 兼容（step= ... loss= ... reward= ... resp_len= ... lr= ...），
  这样可以共用 plot_curves.py
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from sdpo_vlm.feedback.template import build_for_rollout
from sdpo_vlm.rollout.base import RolloutBackend
from sdpo_vlm.rollout.hf_backend import HFRolloutBackend
from sdpo_vlm.trainer.align import (
    compute_current_logprobs, extract_response_ids, slice_response_logits,
)
from sdpo_vlm.trainer.baseline_losses import (
    dpo_pair_loss, dpo_pair_loss_simpo, dpo_pair_loss_sft_aux,
    grpo_normalise_advantages, ppo_token_loss, reinforce_loss,
    reinforce_loss_length_norm,
)
from sdpo_vlm.trainer.callbacks import AdapterSyncCallback, ConsoleLogger
from sdpo_vlm.trainer.trainer import (
    _constant_lambda, _cosine_lambda, _make_lr_lambda, _repeat_iter, load_config,
    prepare_forward_inputs,
)


def _torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_baseline_model(cfg: Dict, device: str):
    """与 SDPO trainer 相比少挂一个 teacher adapter，仅 student。
    支持 attn_implementation / device_map / max_memory（与 SDPO trainer 同语义）。"""
    from transformers import AutoProcessor
    from peft import LoraConfig, PeftModel, get_peft_model

    model_path = cfg["model"]["base_path"]
    dtype = _torch_dtype(cfg["model"]["dtype"])
    trust_remote_code = cfg["model"].get("trust_remote_code", True)
    attn_impl = cfg["model"].get("attn_implementation", "sdpa")
    device_map = cfg["model"].get("device_map", None)

    from_pretrained_kwargs = dict(
        dtype=dtype, trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
    )
    if device_map is not None:
        from_pretrained_kwargs["device_map"] = device_map
        max_mem = cfg["model"].get("max_memory")
        if max_mem:
            from_pretrained_kwargs["max_memory"] = {
                int(k) if isinstance(k, str) and k.isdigit() else k: v
                for k, v in max_mem.items()
            }

    try:
        from transformers import AutoModelForImageTextToText
        base = AutoModelForImageTextToText.from_pretrained(
            model_path, **from_pretrained_kwargs,
        )
    except Exception:
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            model_path, **from_pretrained_kwargs,
        )

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code,
    )

    if cfg["lora"].get("freeze_visual", True):
        for n, p in base.named_parameters():
            if any(k in n for k in ["visual.", "vision_", "image_encoder",
                                     "visual_model", "merger"]):
                p.requires_grad_(False)

    lc = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], bias="none",
        task_type="CAUSAL_LM",
    )
    resume_path = cfg["model"].get("resume_adapter")
    if resume_path:
        model = PeftModel.from_pretrained(
            base, resume_path, adapter_name="student", is_trainable=True,
        )
        print(f"[resume_adapter] loaded student adapter from {resume_path}", flush=True)
    else:
        model = get_peft_model(base, peft_cfg, adapter_name="student")
    model.set_adapter("student")
    for name, p in model.named_parameters():
        if ".student." in name and "lora_" in name:
            p.requires_grad_(True)

    if device_map is None:
        model = model.to(device)

    if cfg["deepspeed"].get("gradient_checkpointing", True):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
        model.gradient_checkpointing_enable()
    return model, processor


def _initial_adapter_sync(model, cb: AdapterSyncCallback):
    path = Path(cb.out_dir) / "adapter_init"
    path.mkdir(parents=True, exist_ok=True)
    prev = getattr(model, "active_adapter", None)
    try:
        model.set_adapter("student")
        model.save_pretrained(str(path), safe_serialization=True)
    finally:
        if prev is not None:
            try: model.set_adapter(prev)
            except Exception: pass
    actual = str(path)
    sub = path / "student"
    if sub.is_dir() and (sub / "adapter_config.json").exists():
        actual = str(sub)
    try:
        cb.rollout_backend.sync_adapter(actual)
        print(f"[adapter_sync] initial adapter → rollout at {actual}")
    except Exception as exc:
        print(f"[adapter_sync] initial sync failed: {exc}")


def _student_logp_with_grad(model, processor, messages, response_text,
                            image_path, device):
    """单次前向，返回 student 当前 logprob 序列 [R]（带梯度），以及 R。"""
    inputs, ids, resp_len = prepare_forward_inputs(
        processor, messages, response_text, image_path, device,
    )
    out = model(**inputs, use_cache=False)
    logits = out.logits
    resp_logits = slice_response_logits(logits, ids, processor, resp_len)  # [R, V]
    resp_ids = extract_response_ids(ids, processor, resp_len)
    R = min(resp_logits.shape[0], len(resp_ids))
    resp_logits = resp_logits[:R]
    ids_t = torch.tensor(resp_ids[:R], dtype=torch.long, device=resp_logits.device)
    logp_full = F.log_softmax(resp_logits.float(), dim=-1)
    logp = logp_full.gather(1, ids_t.unsqueeze(1)).squeeze(1)   # [R]
    return logp, R


@torch.no_grad()
def _ref_logp_no_grad(model, processor, messages, response_text,
                      image_path, device):
    """关掉 LoRA adapter（disable adapter）= ref policy = base model。"""
    # PEFT: enable/disable adapter
    with model.disable_adapter():
        inputs, ids, resp_len = prepare_forward_inputs(
            processor, messages, response_text, image_path, device,
        )
        out = model(**inputs, use_cache=False)
        logits = out.logits
        resp_logits = slice_response_logits(logits, ids, processor, resp_len)
        resp_ids = extract_response_ids(ids, processor, resp_len)
        R = min(resp_logits.shape[0], len(resp_ids))
        resp_logits = resp_logits[:R]
        ids_t = torch.tensor(resp_ids[:R], dtype=torch.long,
                             device=resp_logits.device)
        logp_full = F.log_softmax(resp_logits.float(), dim=-1)
        logp = logp_full.gather(1, ids_t.unsqueeze(1)).squeeze(1)
    return logp, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    algo = cfg["train"].get("algo", "grpo")
    assert algo in ("dpo", "grpo", "dpo_simpo", "dpo_sftaux", "grpo_ln"), \
        f"unsupported algo: {algo}"
    print(f"[trainer_baseline] algo={algo}", flush=True)

    random.seed(cfg["train"].get("seed", 42))
    torch.manual_seed(cfg["train"].get("seed", 42))

    out_dir = Path(cfg["train"]["output_dir"]) / cfg["train"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    ds_train = IsoGraphDataset(
        roots=cfg["data"]["root"],
        split="train",
        train_val_split=cfg["data"]["train_val_split"],
        max_samples=cfg["data"].get("max_train_samples"),
        keep_full_polygon=cfg["data"].get("keep_full_polygon", False),
        seed=cfg["data"].get(
            "split_seed", cfg["train"].get("seed", 42)
        ),
    )
    print(f"[data] train samples = {len(ds_train)}", flush=True)
    dl = DataLoader(ds_train, batch_size=1, shuffle=True,
                     collate_fn=simple_collate, num_workers=0)

    model, processor = load_baseline_model(cfg, device=args.train_device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {trainable/1e6:.2f}M / {total/1e9:.2f}B "
          f"on {args.train_device}", flush=True)

    backend_name = cfg["rollout"]["backend"]
    if backend_name != "hf":
        raise NotImplementedError("baseline trainer 目前只支持 hf rollout backend")
    rollout = HFRolloutBackend(
        model_path=cfg["model"]["base_path"],
        device=cfg["rollout"]["hf"]["device"],
        dtype=cfg["model"]["dtype"],
        max_batch=cfg["rollout"]["hf"].get("max_batch", 2),
    )
    print(f"[rollout] HF backend on {cfg['rollout']['hf']['device']}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, _make_lr_lambda(
            cfg["train"].get("lr_schedule", "cosine"),
            cfg["train"]["warmup_steps"],
            cfg["train"]["max_steps"],
        ),
    )

    adapter_cb = AdapterSyncCallback(
        rollout_backend=rollout,
        out_dir=str(out_dir / "adapters"),
        n_sync=cfg["train"]["n_sync"],
        adapter_name="student",
    )
    console = ConsoleLogger()

    step = 0
    grad_accum = cfg["train"]["grad_accum"]
    max_steps = cfg["train"]["max_steps"]
    log_interval = cfg["train"]["log_interval"]
    G = cfg["rollout"]["group_size"]
    dpo_beta = cfg["train"].get("dpo_beta", 0.1)
    ppo_clip = cfg["train"].get("ppo_clip_eps", 0.2)
    # SimPO / SFT-aux / length-norm GRPO extras:
    simpo_beta = cfg["train"].get("simpo_beta", 2.5)
    simpo_gamma = cfg["train"].get("simpo_gamma", 0.5)
    sftaux_alpha = cfg["train"].get("sftaux_alpha", 0.1)
    grpo_len_penalty = cfg["train"].get("grpo_length_penalty", 0.05)
    grpo_len_ref = cfg["train"].get("grpo_length_ref", 1500.0)

    data_iter = _repeat_iter(dl)
    micro = 0
    optim.zero_grad()
    running: Dict[str, float] = {"loss": 0.0, "reward": 0.0, "resp_len": 0.0,
                                  "n": 0}
    _initial_adapter_sync(model, adapter_cb)
    t0 = time.time()

    while step < max_steps:
        batch = next(data_iter)[0]
        image_path = batch["image_path"]
        gold_short = batch["gold_short"]
        scenario = batch["scenario"]

        student_msgs = build_student_messages(image_path)
        try:
            rollouts = rollout.sample(
                image_path=image_path, messages=student_msgs, n=G,
                max_new_tokens=cfg["rollout"]["max_new_tokens"],
                temperature=cfg["rollout"]["temperature"],
                top_p=cfg["rollout"]["top_p"],
            )
        except Exception as exc:
            print(f"[rollout] FAIL skip: {exc}", flush=True)
            continue

        per_rollout = []
        for r in rollouts:
            fb = build_for_rollout(
                image_path=image_path,
                rollout={"response_text": r.response_text, "reward": 0.0},
                gold_short=gold_short, scenario=scenario,
                mode=cfg["feedback"]["mode"],
                group_rollouts=None,
                success_threshold=cfg["feedback"]["success_threshold"],
                max_items=cfg["feedback"]["max_diff_items"],
                max_chars_per_item=cfg["feedback"]["max_chars_per_item"],
            )
            per_rollout.append({
                "rollout": r,
                "reward": fb["feedback_reward"],
                "feedback_text": fb["feedback_text"],
            })

        # 过滤空 response
        per_rollout = [x for x in per_rollout if x["rollout"].response_text.strip()]
        if len(per_rollout) < 2:
            continue

        step_loss: Optional[torch.Tensor] = None
        step_rewards: List[float] = [x["reward"] for x in per_rollout]
        step_resp_lens: List[int] = []

        try:
            if algo == "dpo":
                # winner = highest reward, loser = lowest（必须不相等才学）
                hi = max(per_rollout, key=lambda x: x["reward"])
                lo = min(per_rollout, key=lambda x: x["reward"])
                if hi["reward"] - lo["reward"] < 1e-6:
                    # 没有偏好信号：跳过该步
                    continue
                # winner / loser logprob：student & ref（ref = base model = disable adapter）
                model.set_adapter("student")
                logp_w, Rw = _student_logp_with_grad(
                    model, processor, student_msgs,
                    hi["rollout"].response_text, image_path, args.train_device,
                )
                logp_l, Rl = _student_logp_with_grad(
                    model, processor, student_msgs,
                    lo["rollout"].response_text, image_path, args.train_device,
                )
                ref_w, _ = _ref_logp_no_grad(
                    model, processor, student_msgs,
                    hi["rollout"].response_text, image_path, args.train_device,
                )
                ref_l, _ = _ref_logp_no_grad(
                    model, processor, student_msgs,
                    lo["rollout"].response_text, image_path, args.train_device,
                )
                step_loss = dpo_pair_loss(logp_w, logp_l, ref_w, ref_l, beta=dpo_beta)
                step_resp_lens = [Rw, Rl]

            elif algo == "grpo":
                # 组内归一化
                norm_advs = grpo_normalise_advantages(step_rewards)
                losses = []
                for item, A in zip(per_rollout, norm_advs):
                    if abs(A) < 1e-9:
                        # 全组等 reward → 没有信号
                        continue
                    model.set_adapter("student")
                    logp, R = _student_logp_with_grad(
                        model, processor, student_msgs,
                        item["rollout"].response_text, image_path, args.train_device,
                    )
                    # HF backend 不返回 rollout 时 logprobs，退化到 REINFORCE：
                    # L = -mean(logπ_cur(y_t)) * A
                    losses.append(reinforce_loss(logp, float(A)))
                    step_resp_lens.append(R)
                if not losses:
                    continue
                step_loss = torch.stack(losses).mean()

            elif algo == "dpo_simpo":
                # SimPO: length-normalized, reference-free
                hi = max(per_rollout, key=lambda x: x["reward"])
                lo = min(per_rollout, key=lambda x: x["reward"])
                if hi["reward"] - lo["reward"] < 1e-6:
                    continue
                model.set_adapter("student")
                logp_w, Rw = _student_logp_with_grad(
                    model, processor, student_msgs,
                    hi["rollout"].response_text, image_path, args.train_device,
                )
                logp_l, Rl = _student_logp_with_grad(
                    model, processor, student_msgs,
                    lo["rollout"].response_text, image_path, args.train_device,
                )
                step_loss = dpo_pair_loss_simpo(
                    logp_w, logp_l, beta=simpo_beta, gamma=simpo_gamma,
                )
                step_resp_lens = [Rw, Rl]

            elif algo == "dpo_sftaux":
                # DPO + alpha * -logπ(y_gold)
                hi = max(per_rollout, key=lambda x: x["reward"])
                lo = min(per_rollout, key=lambda x: x["reward"])
                if hi["reward"] - lo["reward"] < 1e-6:
                    continue
                model.set_adapter("student")
                logp_w, Rw = _student_logp_with_grad(
                    model, processor, student_msgs,
                    hi["rollout"].response_text, image_path, args.train_device,
                )
                logp_l, Rl = _student_logp_with_grad(
                    model, processor, student_msgs,
                    lo["rollout"].response_text, image_path, args.train_device,
                )
                ref_w, _ = _ref_logp_no_grad(
                    model, processor, student_msgs,
                    hi["rollout"].response_text, image_path, args.train_device,
                )
                ref_l, _ = _ref_logp_no_grad(
                    model, processor, student_msgs,
                    lo["rollout"].response_text, image_path, args.train_device,
                )
                # SFT aux: compute logπ(y_gold) under student adapter
                try:
                    gold_text = short_to_json_str(gold_short)
                    sft_logp, Rg = _student_logp_with_grad(
                        model, processor, student_msgs,
                        gold_text, image_path, args.train_device,
                    )
                except Exception as exc:
                    print(f"[sftaux] gold logp FAIL: {exc}; fallback to 0", flush=True)
                    sft_logp = logp_w.new_zeros(0)
                    Rg = 0
                step_loss = dpo_pair_loss_sft_aux(
                    logp_w, logp_l, ref_w, ref_l, sft_logp,
                    beta=dpo_beta, alpha=sftaux_alpha,
                )
                step_resp_lens = [Rw, Rl, Rg]

            elif algo == "grpo_ln":
                # GRPO with length-penalized advantage
                norm_advs = grpo_normalise_advantages(step_rewards)
                losses = []
                for item, A in zip(per_rollout, norm_advs):
                    if abs(A) < 1e-9:
                        continue
                    model.set_adapter("student")
                    logp, R = _student_logp_with_grad(
                        model, processor, student_msgs,
                        item["rollout"].response_text, image_path, args.train_device,
                    )
                    losses.append(reinforce_loss_length_norm(
                        logp, float(A),
                        length_penalty=grpo_len_penalty,
                        len_ref=grpo_len_ref,
                    ))
                    step_resp_lens.append(R)
                if not losses:
                    continue
                step_loss = torch.stack(losses).mean()
        except torch.cuda.OutOfMemoryError:
            print("[OOM] skip step", flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"[forward] FAIL skip: {exc}", flush=True)
            continue

        if step_loss is None:
            continue

        loss_norm = step_loss / grad_accum
        loss_norm.backward()
        micro += 1

        running["loss"] += step_loss.item()
        running["reward"] += sum(step_rewards) / len(step_rewards)
        running["resp_len"] += sum(step_resp_lens) / max(1, len(step_resp_lens))
        running["n"] += 1

        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params,
                                           cfg["train"]["grad_clip"])
            optim.step(); sched.step(); optim.zero_grad()
            step += 1
            if step % log_interval == 0 and running["n"] > 0:
                el = time.time() - t0
                m = {
                    "loss": running["loss"] / running["n"],
                    "reward": running["reward"] / running["n"],
                    "resp_len": running["resp_len"] / running["n"],
                    "lr": sched.get_last_lr()[0],
                }
                print(f"step={step} | elapsed={el:.1f}s | loss={m['loss']:.4f} "
                      f"| reward={m['reward']:.4f} | resp_len={m['resp_len']:.4f} "
                      f"| lr={m['lr']:.2e}", flush=True)
                running = {"loss": 0.0, "reward": 0.0, "resp_len": 0.0, "n": 0}

            adapter_cb.maybe_sync(model, step)

            if step % cfg["train"]["save_interval"] == 0:
                save_path = out_dir / f"ckpt_step{step:06d}"
                model.save_pretrained(str(save_path))
                print(f"[save] {save_path}", flush=True)

    print("[done] training complete.")


if __name__ == "__main__":
    main()
