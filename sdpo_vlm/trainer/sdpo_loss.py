"""SDPO loss：top-K 对称 JSD，带 tail 项；支持 PPO 式 importance clip。

输入形状：
  student_logits: [R, V]  需梯度
  teacher_logits: [R, V]  stopgrad（no_grad 外部已保证）

步骤：
  1. 在 student 侧取 top-K token 集合 S_t（K = top_k）
  2. 把 student/teacher 在这 K 个 id 上的 logprob 取出，并计算 tail
     p_tail = log(1 - sum_{i∈S} p_i) = logsumexp(outside logits) 再和 logsumexp(all)
     对应求差；实现上我们直接在 softmax 概率空间做：
       p_top = softmax(student)[:, S]
       p_tail = 1 - sum(p_top)
     同理 q_top / q_tail
  3. m = 0.5 * (p + q) 在 S ∪ {tail} 维度上
  4. JSD_t = 0.5 * KL(p || m) + 0.5 * KL(q || m)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def topk_symmetric_jsd_loss(
    student_logits: torch.Tensor,   # [R, V]
    teacher_logits: torch.Tensor,   # [R, V]
    *,
    top_k: int = 100,
    mask: Optional[torch.Tensor] = None,   # [R]，1=计入 loss
    importance_ratio: Optional[torch.Tensor] = None,  # [R]，PPO clip 权重
    importance_clip_eps: float = 0.2,
    reduction: str = "mean",
    use_jsd: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """返回标量 loss。

    若 use_jsd=False，退化到 KL(student || teacher)（仅 top-K+tail）。
    """
    assert student_logits.shape == teacher_logits.shape
    R, V = student_logits.shape
    if top_k > V:
        top_k = V

    # 1. 在 student 侧求 top-K ids（每个 t 独立）
    _, topk_ids = torch.topk(student_logits, k=top_k, dim=-1)        # [R, K]

    # 2. 各自的全概率
    p = F.softmax(student_logits.float(), dim=-1)                     # [R, V]
    q = F.softmax(teacher_logits.float(), dim=-1)                     # [R, V]

    # 3. gather top-K 概率
    p_top = torch.gather(p, 1, topk_ids)                              # [R, K]
    q_top = torch.gather(q, 1, topk_ids)                              # [R, K]

    # 4. tail
    p_tail = (1.0 - p_top.sum(dim=-1, keepdim=True)).clamp_min(eps)   # [R, 1]
    q_tail = (1.0 - q_top.sum(dim=-1, keepdim=True)).clamp_min(eps)   # [R, 1]

    p_cat = torch.cat([p_top, p_tail], dim=-1).clamp_min(eps)          # [R, K+1]
    q_cat = torch.cat([q_top, q_tail], dim=-1).clamp_min(eps)          # [R, K+1]

    if use_jsd:
        m = 0.5 * (p_cat + q_cat)
        # JSD = 0.5*KL(p||m) + 0.5*KL(q||m) = 0.5*sum(p*log(p/m)) + 0.5*sum(q*log(q/m))
        kl_pm = (p_cat * (p_cat.log() - m.log())).sum(dim=-1)          # [R]
        kl_qm = (q_cat * (q_cat.log() - m.log())).sum(dim=-1)
        per_tok = 0.5 * (kl_pm + kl_qm)
    else:
        # 纯 KL(student || teacher)
        per_tok = (p_cat * (p_cat.log() - q_cat.log())).sum(dim=-1)    # [R]

    if mask is None:
        mask = per_tok.new_ones(R)

    # PPO-style importance clip（用于半在线：rollout 是旧 adapter）
    if importance_ratio is not None:
        clipped = importance_ratio.clamp(
            1.0 - importance_clip_eps, 1.0 + importance_clip_eps
        )
        per_tok = per_tok * clipped

    num = (per_tok * mask).sum()
    den = mask.sum().clamp_min(1.0)

    if reduction == "mean":
        return num / den
    elif reduction == "sum":
        return num
    else:
        raise ValueError(reduction)
