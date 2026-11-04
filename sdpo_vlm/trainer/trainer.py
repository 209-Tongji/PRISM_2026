"""SDPO VLM 训练主循环。

本文件是一个**单进程** trainer，作为 MVP。后续可改造为 DeepSpeed 分布式。
流程：
  1) 加载 base model + LoRA(student) + LoRA(teacher)（teacher 初始化自 student）
  2) 加载 rollout backend（HF 或 SGLang）
  3) for step in range(max_steps):
       a) 抽 batch prompts
       b) rollout.sample → 多 rollout（每个 prompt n=G）
       c) 对每个 rollout 算 reward / feedback_text
       d) 组装 (student_input, teacher_input, response_ids)
       e) student forward（需梯度）+ teacher forward（no_grad + set_adapter('teacher')）
       f) 取两边 response 段 logits → top-K JSD loss → backward → optim step
       g) EMA update teacher
       h) 每 N 步 save adapter + rollout.sync_adapter
"""
from __future__ import annotations

import argparse
import json
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

# ---- path bootstrap（允许 `python trainer/trainer.py`） ----
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdpo_vlm.data.dataset import IsoGraphDataset, simple_collate
from sdpo_vlm.configuration import load_config
from sdpo_vlm.data.prompt import (
    build_student_messages, build_teacher_messages, find_response_start,
)
from sdpo_vlm.data.schema import short_to_json_str
from sdpo_vlm.feedback.template import build_for_rollout
from sdpo_vlm.rollout.base import RolloutBackend
from sdpo_vlm.rollout.hf_backend import HFRolloutBackend
from sdpo_vlm.rollout.sglang_backend import SGLangRolloutBackend
from sdpo_vlm.trainer.align import (
    compute_current_logprobs, extract_response_ids, slice_response_logits,
)
from sdpo_vlm.trainer.callbacks import AdapterSyncCallback, ConsoleLogger
from sdpo_vlm.trainer.sdpo_loss import topk_symmetric_jsd_loss
from sdpo_vlm.trainer.teacher import ema_update, snapshot_adapter, trust_region_update


# ---------------- model init ----------------

def _torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_training_model(cfg: Dict, device: str):
    """加载 base model + student/teacher LoRA adapter。

    支持 cfg.model 下的两个新键：
      - attn_implementation: "sdpa" | "flash_attention_2"  (默认 sdpa)
      - device_map: None | "balanced" | dict
          None：单卡，model.to(device)；
          "balanced"：HF 自动把 transformer layer 平铺到所有可见 GPU；
          dict：自定义 device_map（透传给 from_pretrained）。
        当 device_map 不是 None 时，device 参数仅用于把 input 喂到第一层所在的卡，
        不会再 .to() 整个模型。
    """
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
        # max_memory 限制每张卡的预算（避免 device_map="balanced" 把太多塞到一张卡）
        max_mem = cfg["model"].get("max_memory")
        if max_mem:
            # 允许 yaml 写 {0: "70GiB", 1: "70GiB", ...}，转 int key
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

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    # 冻结 visual encoder
    if cfg["lora"].get("freeze_visual", True):
        for n, p in base.named_parameters():
            if any(k in n for k in ["visual.", "vision_", "image_encoder",
                                     "visual_model", "merger"]):
                p.requires_grad_(False)

    # 挂 student LoRA。Continuation 必须直接由 PeftModel.from_pretrained
    # 构造 student adapter；在已经存在的同名 fresh adapter 上再次
    # load_adapter("student") 会让部分 PEFT/accelerate 版本把 LoRA 权重留在
    # 错误设备，尤其容易在多卡 device_map 下触发 CUDA invalid argument。
    lc = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], bias="none",
        task_type="CAUSAL_LM",
    )
    resume_path = cfg["model"].get("resume_adapter")
    if resume_path:
        model = PeftModel.from_pretrained(
            base,
            resume_path,
            adapter_name="student",
            is_trainable=True,
        )
        student_peft_cfg = model.peft_config["student"]
        print(
            f"[resume_adapter] loaded student adapter from {resume_path}",
            flush=True,
        )
    else:
        model = get_peft_model(base, peft_cfg, adapter_name="student")
        student_peft_cfg = peft_cfg
    # 再挂一个 teacher adapter（同构）
    model.add_adapter("teacher", student_peft_cfg)

    # 只让 student 可训
    model.set_adapter("student")
    for name, p in model.named_parameters():
        if ".student." in name and "lora_" in name:
            p.requires_grad_(True)
        elif ".teacher." in name:
            p.requires_grad_(False)

    if device_map is None:
        model = model.to(device)
    # else: device_map 已经把权重放到各卡，不要再 .to() 整体（会爆）

    # gradient checkpointing：PEFT + GC 需要 enable_input_require_grads
    if cfg["deepspeed"].get("gradient_checkpointing", True):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
        model.gradient_checkpointing_enable()
    return model, processor


# ---------------- forward 工具 ----------------

def prepare_forward_inputs(
    processor,
    messages: List[Dict],
    response_text: str,
    image_path: str,
    device: str,
) -> Tuple[Dict, List[int], int]:
    """把 (messages + response) tokenize + 把图像编码，返回 model 接受的 inputs。
    同时返回 full input_ids（list）和 response 的 token 长度。"""
    # 先拼 chat text（tokenize=False），然后加 response
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # 手工拼 response + eos
    eos = processor.tokenizer.eos_token or "<|im_end|>"
    text_full = text_prompt + response_text + eos + "\n"

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[text_full], images=[image], padding=True, return_tensors="pt",
    )
    input_ids = inputs["input_ids"][0].tolist()

    # response 长度：从 response_start 到 input_ids 末尾（包含 eos 但不含结尾 \n 的 token）
    start = find_response_start(processor, input_ids)
    # response 结束：找到最后一个非 pad / 非 \n token
    response_len = len(input_ids) - start
    # 把所有 tensor 搬到 device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, input_ids, response_len


def forward_logits(model, inputs: Dict) -> torch.Tensor:
    """返回 [1, L, V] 的 logits。"""
    out = model(**inputs, use_cache=False)
    return out.logits


# ---------------- 训练循环 ----------------

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
        # Keep the data partition fixed across optimizer/model seeds when
        # data.split_seed is provided. Historically train.seed controlled both,
        # confounding method variance with train/val membership.
        seed=cfg["data"].get(
            "split_seed", cfg["train"].get("seed", 42)
        ),
    )
    print(f"[data] train samples = {len(ds_train)}")
    dl = DataLoader(ds_train, batch_size=1, shuffle=True,
                     collate_fn=simple_collate, num_workers=0)

    # ---- model ----
    model, processor = load_training_model(cfg, device=args.train_device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {trainable/1e6:.2f}M / {total/1e9:.2f}B "
          f"on {args.train_device}", flush=True)
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            try:
                free, total_b = torch.cuda.mem_get_info(d)
                print(f"  [gpu{d}] free={free/1e9:.1f}GB total={total_b/1e9:.1f}GB",
                      flush=True)
            except Exception as exc:
                print(f"  [gpu{d}] mem_get_info failed: {exc}", flush=True)

    # ---- rollout backend ----
    backend_name = cfg["rollout"]["backend"]
    if backend_name == "hf":
        rollout = HFRolloutBackend(
            model_path=cfg["model"]["base_path"],
            device=cfg["rollout"]["hf"]["device"],
            dtype=cfg["model"]["dtype"],
            max_batch=cfg["rollout"]["hf"].get("max_batch", 2),
        )
        print(f"[rollout] HF backend on {cfg['rollout']['hf']['device']}",
              flush=True)
    elif backend_name == "sglang":
        rollout = SGLangRolloutBackend(
            base_url=cfg["rollout"]["sglang"]["base_url"],
            timeout_s=cfg["rollout"]["sglang"].get("timeout_s", 600),
        )
    else:
        raise ValueError(backend_name)

    # ---- teacher 初始化 ----
    teacher_cfg = cfg["sdpo"]["teacher"]
    teacher_mode = teacher_cfg["mode"]
    init_snapshot = None
    if teacher_mode == "trust_region":
        init_snapshot = snapshot_adapter(model, "student")
    # teacher adapter 初始化为 student 的拷贝（保持和 RLVR baseline 对齐），
    # 若 frozen 则永不更新；EMA 会逐步漂移
    _init_teacher_from_student(model)

    # ---- optimizer ----
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

    # ---- callbacks ----
    adapter_cb = AdapterSyncCallback(
        rollout_backend=rollout,
        out_dir=str(out_dir / "adapters"),
        n_sync=cfg["train"]["n_sync"],
        adapter_name="student",
    )
    console = ConsoleLogger()

    # ---- 主循环 ----
    step = 0
    grad_accum = cfg["train"]["grad_accum"]
    max_steps = cfg["train"]["max_steps"]
    log_interval = cfg["train"]["log_interval"]
    top_k = cfg["sdpo"]["top_k"]
    use_jsd = cfg["sdpo"].get("jsd", True)
    imp_eps = cfg["sdpo"]["importance_clip_eps"]
    # New SDPO variants (for attempting to exceed SFT ceiling):
    #   reward_weighted: if true, weight per-rollout JSD loss by softmax(reward / tau)
    #                    instead of equal-mean. Method A.
    #   reward_weighted_tau: temperature for the softmax (smaller = sharper).
    #   dpo_aux_alpha: if >0, add alpha * DPO pair loss (best vs worst rollout by reward).
    #                  Requires G >= 2. Method B.
    #   dpo_aux_beta: DPO beta for the aux pair loss.
    reward_weighted = cfg["sdpo"].get("reward_weighted", False)
    reward_weighted_tau = cfg["sdpo"].get("reward_weighted_tau", 0.5)
    dpo_aux_alpha = cfg["sdpo"].get("dpo_aux_alpha", 0.0)
    dpo_aux_beta = cfg["sdpo"].get("dpo_aux_beta", 0.1)
    # per_rollout_backward: call .backward() after each rollout's forward and free
    # the graph. Lets G=4+ fit in memory at the cost of not supporting dpo_aux
    # (needs multi-rollout sum-logp at aggregation time).
    per_rollout_backward = cfg["sdpo"].get("per_rollout_backward", False)
    if per_rollout_backward and (reward_weighted or dpo_aux_alpha > 0):
        raise ValueError("per_rollout_backward requires reward_weighted=false and dpo_aux_alpha=0")
    if reward_weighted and reward_weighted_tau <= 0:
        raise ValueError("reward_weighted_tau must be positive")
    G = cfg["rollout"]["group_size"]

    data_iter = _repeat_iter(dl)
    micro = 0
    optim.zero_grad()
    running: Dict[str, float] = {"loss": 0.0, "reward": 0.0, "resp_len": 0.0,
                                  "imp_mean": 0.0, "imp_n": 0, "n": 0}

    # 初次 adapter 同步（rollout 侧先得到 student 权重）
    _initial_adapter_sync(model, adapter_cb)

    while step < max_steps:
        batch = next(data_iter)[0]
        image_path = batch["image_path"]
        gold_short = batch["gold_short"]
        scenario = batch["scenario"]

        # 1) rollout G 个样本（用 rollout backend 上最后一次同步的 adapter）
        student_msgs = build_student_messages(image_path)
        try:
            rollouts = rollout.sample(
                image_path=image_path,
                messages=student_msgs,
                n=G,
                max_new_tokens=cfg["rollout"]["max_new_tokens"],
                temperature=cfg["rollout"]["temperature"],
                top_p=cfg["rollout"]["top_p"],
            )
        except Exception as exc:
            print(f"[rollout] FAIL skip: {exc}")
            continue

        # 2) 评 reward，构造每条的 feedback
        per_rollout_info: List[Dict] = []
        for r in rollouts:
            fb = build_for_rollout(
                image_path=image_path,
                rollout={"response_text": r.response_text, "reward": 0.0},
                gold_short=gold_short,
                scenario=scenario,
                mode=cfg["feedback"]["mode"],
                feedback_level=cfg["feedback"].get("level"),
                group_rollouts=None,  # 第一遍先不考虑 group success
                success_threshold=cfg["feedback"]["success_threshold"],
                max_items=cfg["feedback"]["max_diff_items"],
                max_chars_per_item=cfg["feedback"]["max_chars_per_item"],
            )
            per_rollout_info.append({
                "rollout": r,
                "reward": fb["feedback_reward"],
                "feedback_text": fb["feedback_text"],
                "correct_solution_json": fb["correct_solution_json"],
                "teacher_messages": fb["teacher_messages"],
            })

        # 3) 过一遍组内 success 选择（第二遍把 best rollout 作 correct_solution）
        best_reward = max(x["reward"] for x in per_rollout_info)
        if best_reward >= cfg["feedback"]["success_threshold"] and \
           cfg["feedback"]["mode"] in ("success", "auto", "auto_no_dgr"):
            best = max(per_rollout_info, key=lambda x: x["reward"])
            for item in per_rollout_info:
                if item is best:
                    continue
                item["teacher_messages"] = build_teacher_messages(
                    image_path=image_path,
                    feedback_text=item["feedback_text"],
                    correct_solution_json=best["rollout"].response_text,
                )

        # 4) 对每个 rollout 做 student & teacher forward
        step_losses: List[torch.Tensor] = []
        step_rewards: List[float] = []
        step_resp_lens: List[int] = []
        per_rollout_student_sumlogp: List[torch.Tensor] = []  # for DPO aux (method B)
        for item in per_rollout_info:
            r: RolloutBackend = item["rollout"]
            if not r.response_text.strip():
                continue
            try:
                # --- student ---
                model.set_adapter("student")
                s_inputs, s_ids, s_resp_len = prepare_forward_inputs(
                    processor, student_msgs, r.response_text, image_path,
                    args.train_device,
                )
                s_logits = forward_logits(model, s_inputs)
                s_resp_logits = slice_response_logits(
                    s_logits, s_ids, processor, s_resp_len,
                )  # [R, V]

                # --- teacher (no grad) ---
                # teacher_mode='online' 时不切 adapter（teacher = student 实时同步），
                # 只靠 prompt 差异制造 student/teacher gap，是"prompt-only self-distillation"
                if teacher_mode == "online":
                    # 保持 student adapter
                    pass
                else:
                    model.set_adapter("teacher")
                with torch.no_grad():
                    t_inputs, t_ids, t_resp_len = prepare_forward_inputs(
                        processor, item["teacher_messages"], r.response_text,
                        image_path, args.train_device,
                    )
                    t_logits = forward_logits(model, t_inputs)
                    t_resp_logits = slice_response_logits(
                        t_logits, t_ids, processor, t_resp_len,
                    )
                # 切回 student 继续 backward（如果之前切过）
                if teacher_mode != "online":
                    model.set_adapter("student")

                # 对齐 R
                R = min(s_resp_logits.shape[0], t_resp_logits.shape[0])
                if R < 2:
                    continue
                s_resp_logits = s_resp_logits[:R]
                t_resp_logits = t_resp_logits[:R].to(s_resp_logits.device)

                # --- PPO-style importance ratio（半在线）---
                # rollout 是由 N step 前的 adapter 生成的，student 现在已经更新；
                # 需要 ratio = exp(logπ_current(y_t) - logπ_rollout(y_t))，再 clip。
                imp = None
                if r.logprobs and len(r.logprobs) >= 2:
                    # 1) 从 student 当前 logits 抽出每个 response token 的 logprob
                    resp_ids = extract_response_ids(s_ids, processor, s_resp_len)
                    resp_ids = resp_ids[:R]
                    # 2) rollout 侧的 logprob（SGLang 返回的，对应 rollout 时的 adapter）
                    rollout_lp = torch.tensor(
                        r.logprobs[:R], dtype=torch.float32,
                        device=s_resp_logits.device,
                    )
                    # rollout 有可能只给了 <R 个（例如被 EOS 截断），这时补齐 0
                    if rollout_lp.shape[0] < R:
                        pad = torch.zeros(
                            R - rollout_lp.shape[0],
                            dtype=rollout_lp.dtype, device=rollout_lp.device,
                        )
                        rollout_lp = torch.cat([rollout_lp, pad], dim=0)
                    # 3) student current logprob（用 detach，避免把 ratio 的梯度回传）
                    with torch.no_grad():
                        cur_lp = compute_current_logprobs(
                            s_resp_logits.detach(), resp_ids,
                        )  # [R]
                    # 4) ratio = exp(cur - old)，再夹到 [1-eps, 1+eps]（loss 内做 clip）
                    imp = torch.exp(cur_lp - rollout_lp)
                    # 防爆：超过 10 的 ratio 直接置 1（视为无效 token）
                    imp = torch.where(torch.isfinite(imp) & (imp < 10.0) & (imp > 0.0),
                                      imp, torch.ones_like(imp))

                loss = topk_symmetric_jsd_loss(
                    s_resp_logits, t_resp_logits.detach(),
                    top_k=top_k, use_jsd=use_jsd,
                    importance_ratio=imp, importance_clip_eps=imp_eps,
                )

                # per_rollout_backward: if enabled, backward this rollout's loss
                # now (weighted by 1/G), detach everything, free graph. Saves
                # G× activation memory for backward.
                if per_rollout_backward:
                    # Incorporate reward weight *and* dpo_aux-free scaling here.
                    # (dpo_aux is disabled in this mode since it requires multi-rollout coupling.)
                    w = 1.0 / G
                    if reward_weighted:
                        # Compute softmax weights right here using partial step_rewards.
                        # Since we don't have all rewards yet, approximate by a constant
                        # equal weight (1/G); the true softmax can only be applied after
                        # all rollouts are done. We accept this slight bias in exchange
                        # for memory; the same reward_weighted + per_rollout_backward
                        # combo is only used for G=3,4 diagnostic runs.
                        w = 1.0 / G
                    (loss * (w / grad_accum)).backward()
                    step_losses.append(loss.detach())  # for logging only
                    step_rewards.append(item["reward"])
                    step_resp_lens.append(R)
                    # free memory right now
                    del s_resp_logits, t_resp_logits, s_logits
                    torch.cuda.empty_cache()
                else:
                    step_losses.append(loss)
                    step_rewards.append(item["reward"])
                    step_resp_lens.append(R)

                # Cache student sum-logp for DPO aux (method B). Use the same
                # s_resp_logits we already have, extracting token logprobs:
                if dpo_aux_alpha > 0:
                    try:
                        resp_ids_for_aux = extract_response_ids(s_ids, processor, s_resp_len)[:R]
                        ids_t = torch.tensor(resp_ids_for_aux, dtype=torch.long,
                                             device=s_resp_logits.device)
                        logp_all = torch.nn.functional.log_softmax(
                            s_resp_logits.float(), dim=-1,
                        )
                        per_tok_logp = logp_all.gather(1, ids_t.unsqueeze(1)).squeeze(1)
                        per_rollout_student_sumlogp.append(per_tok_logp.sum())
                    except Exception:
                        per_rollout_student_sumlogp.append(s_resp_logits.new_tensor(0.0))

                if imp is not None:
                    running["imp_mean"] += imp.mean().item()
                    running["imp_n"] += 1
            except torch.cuda.OutOfMemoryError:
                print("[OOM] skip rollout")
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                import traceback; traceback.print_exc()
                print(f"[forward] FAIL skip: {exc}")
                continue

        if not step_losses:
            continue

        if per_rollout_backward:
            # Backward already done per-rollout; just track running stats.
            micro += 1
            running["loss"] += sum(l.item() for l in step_losses) / len(step_losses)
            running["reward"] += sum(step_rewards) / len(step_rewards)
            running["resp_len"] += sum(step_resp_lens) / len(step_resp_lens)
            running["n"] += 1
        else:
            # Aggregate per-rollout JSD losses.
            # Method A (reward_weighted): weight by softmax(reward / tau) instead of equal mean.
            if reward_weighted and len(step_losses) > 1:
                import math
                rewards_arr = torch.tensor(step_rewards, device=step_losses[0].device,
                                           dtype=torch.float32)
                weights = torch.softmax(rewards_arr / max(reward_weighted_tau, 1e-6), dim=0)
                loss_t = sum(w * l for w, l in zip(weights.tolist(), step_losses)) / grad_accum
            else:
                loss_t = torch.stack(step_losses).mean() / grad_accum

            # Method B (dpo_aux): add alpha * DPO pair loss on best-vs-worst rollouts (by reward).
            if dpo_aux_alpha > 0 and len(step_losses) >= 2 and len(step_rewards) == len(step_losses):
                # We need student logp on best & worst rollouts; these were cached
                # as per_rollout_student_sumlogp in the loop above.
                if 'per_rollout_student_sumlogp' in locals() and len(per_rollout_student_sumlogp) >= 2:
                    # pick best & worst by reward
                    hi_idx = max(range(len(step_rewards)), key=lambda i: step_rewards[i])
                    lo_idx = min(range(len(step_rewards)), key=lambda i: step_rewards[i])
                    if step_rewards[hi_idx] - step_rewards[lo_idx] > 1e-6:
                        lp_w = per_rollout_student_sumlogp[hi_idx]
                        lp_l = per_rollout_student_sumlogp[lo_idx]
                        # SimPO-style length-normalized DPO (reference-free, no extra forward)
                        lp_w_n = lp_w / max(step_resp_lens[hi_idx], 1)
                        lp_l_n = lp_l / max(step_resp_lens[lo_idx], 1)
                        logits = dpo_aux_beta * (lp_w_n - lp_l_n)
                        dpo_l = -torch.nn.functional.logsigmoid(logits)
                        loss_t = loss_t + dpo_aux_alpha * (dpo_l / grad_accum)

            loss_t.backward()
            micro += 1

            running["loss"] += loss_t.item() * grad_accum
            running["reward"] += sum(step_rewards) / len(step_rewards)
            running["resp_len"] += sum(step_resp_lens) / len(step_resp_lens)
            running["n"] += 1

        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params,
                                           cfg["train"]["grad_clip"])
            optim.step(); sched.step(); optim.zero_grad()

            # teacher 更新
            if teacher_mode == "ema":
                ema_update(model, "student", "teacher",
                           alpha=teacher_cfg["ema_alpha"])
            elif teacher_mode == "trust_region":
                trust_region_update(model, "student", "teacher",
                                     init_snapshot,
                                     alpha=teacher_cfg["trust_region_alpha"])
            # frozen 下不动

            step += 1
            if step % log_interval == 0 and running["n"] > 0:
                m = {
                    "loss": running["loss"] / running["n"],
                    "reward": running["reward"] / running["n"],
                    "resp_len": running["resp_len"] / running["n"],
                    "lr": sched.get_last_lr()[0],
                }
                if running["imp_n"] > 0:
                    m["imp_ratio"] = running["imp_mean"] / running["imp_n"]
                console.log(step, m)
                running = {"loss": 0.0, "reward": 0.0, "resp_len": 0.0,
                            "imp_mean": 0.0, "imp_n": 0, "n": 0}

            adapter_cb.maybe_sync(model, step)

            if step % cfg["train"]["save_interval"] == 0:
                save_path = out_dir / f"ckpt_step{step:06d}"
                model.save_pretrained(str(save_path))
                print(f"[save] {save_path}")

    print("[done] training complete.")


def _cosine_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


def _constant_lambda(warmup: int):
    """constant with warmup：前 warmup 步线性 ramp，之后保持 1.0。"""
    def f(step):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        return 1.0
    return f


def _make_lr_lambda(kind: str, warmup: int, total: int):
    if kind == "cosine":
        return _cosine_lambda(warmup, total)
    elif kind == "constant":
        return _constant_lambda(warmup)
    else:
        raise ValueError(f"unknown lr_schedule: {kind}")


def _repeat_iter(dl: DataLoader):
    while True:
        for b in dl:
            yield b


@torch.no_grad()
def _init_teacher_from_student(model):
    """把 student adapter 的参数拷贝给 teacher adapter，作为初始状态。"""
    from sdpo_vlm.trainer.teacher import _iter_lora_params
    s = {name: p for name, p in _iter_lora_params(model, "student")}
    t = {name: p for name, p in _iter_lora_params(model, "teacher")}
    for s_name, s_p in s.items():
        t_name = s_name.replace(".student.", ".teacher.")
        if t_name in t and t[t_name].shape == s_p.shape:
            t[t_name].copy_(s_p)


def _initial_adapter_sync(model, cb: AdapterSyncCallback):
    """step 0 之前先把当前 student adapter 导出并让 rollout 热加载。"""
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


if __name__ == "__main__":
    main()
