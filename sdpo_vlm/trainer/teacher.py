"""Teacher 维护策略：EMA / trust-region / frozen。

实现思路：**同一个 base model**，挂两个 LoRA adapter：
  - name='student'   : 可训（optimizer 只更新它）
  - name='teacher'   : 不训，按配置维护

forward 时：
  model.set_adapter('student') → student forward
  model.set_adapter('teacher') → teacher forward (no_grad)

EMA：每个 step 把 student adapter 的参数按 α 融合到 teacher adapter。
trust-region：teacher = (1-α)*init + α*student（init 在 step 0 保存一份）。
frozen：teacher 不变，永远等于初始化时的状态。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def _iter_lora_params(model, adapter_name: str):
    """遍历指定 adapter 的 LoRA 参数（peft 内部的 lora_A / lora_B / lora_embedding_A/B）。"""
    for name, p in model.named_parameters():
        # PEFT 的参数名约定：lora_A.{adapter}.xxx / lora_B.{adapter}.xxx 等
        if f".{adapter_name}." in name and ("lora_" in name):
            yield name, p


@torch.no_grad()
def snapshot_adapter(model, adapter_name: str) -> Dict[str, torch.Tensor]:
    """把某 adapter 的参数 clone 成 dict，供 trust-region init 用。"""
    snap: Dict[str, torch.Tensor] = {}
    for name, p in _iter_lora_params(model, adapter_name):
        snap[name] = p.detach().clone()
    return snap


@torch.no_grad()
def ema_update(model, student_name: str, teacher_name: str, alpha: float = 0.995):
    """θ_teacher ← α*θ_teacher + (1-α)*θ_student，对每一对同名 lora 参数。

    两个 adapter 在 PEFT 下有完全对应的参数名，只是中间的 adapter_name 不同。
    我们做一次字符串替换来配对。
    """
    student_params = {name: p for name, p in _iter_lora_params(model, student_name)}
    teacher_params = {name: p for name, p in _iter_lora_params(model, teacher_name)}
    matched = 0
    for s_name, s_p in student_params.items():
        t_name = s_name.replace(f".{student_name}.", f".{teacher_name}.")
        t_p = teacher_params.get(t_name)
        if t_p is None or t_p.shape != s_p.shape:
            continue
        t_p.mul_(alpha).add_(s_p, alpha=(1.0 - alpha))
        matched += 1
    return matched


@torch.no_grad()
def trust_region_update(
    model, student_name: str, teacher_name: str,
    init_snapshot: Dict[str, torch.Tensor], alpha: float = 0.01,
):
    """θ_teacher = (1-α)*θ_init + α*θ_student。"""
    student_params = {name: p for name, p in _iter_lora_params(model, student_name)}
    teacher_params = {name: p for name, p in _iter_lora_params(model, teacher_name)}
    for s_name, s_p in student_params.items():
        t_name = s_name.replace(f".{student_name}.", f".{teacher_name}.")
        init_name = s_name  # init snapshot 用 student 名
        t_p = teacher_params.get(t_name)
        init_p = init_snapshot.get(init_name)
        if t_p is None or init_p is None or t_p.shape != s_p.shape:
            continue
        t_p.copy_((1.0 - alpha) * init_p + alpha * s_p)
