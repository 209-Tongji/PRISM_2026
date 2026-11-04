"""与 SDPO 对照的 baseline 算法 loss：

1. DPO（Direct Preference Optimization）：
   给一对 (winner, loser) response（在线 rollout 内 reward 排序后取 best/worst），
   用 reference model（baseline trainer 中关闭 LoRA 的 base model）算
     L = -log sigmoid(beta * (logπ(yw|x) - logπ_ref(yw|x)
                              - logπ(yl|x) + logπ_ref(yl|x)))

2. PPO / RLVR-style policy gradient：
   对每个 rollout token 用 (reward - baseline) 做加权交叉熵，
   并加上 importance ratio clip：
     ratio = exp(logπ_cur(y_t) - logπ_old(y_t))
     L = -mean( min(ratio * A, clip(ratio,1-e,1+e) * A) )
   其中 A = reward - group_mean_reward（GRPO 形式）

这两个仅做对照实验用，不引入 self-teacher / DGR。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def dpo_pair_loss(
    student_logp_w: torch.Tensor,    # [Rw]   winner 在 student 下的 token logprob
    student_logp_l: torch.Tensor,    # [Rl]   loser
    ref_logp_w: torch.Tensor,        # [Rw]
    ref_logp_l: torch.Tensor,        # [Rl]
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """标准 DPO loss（log-sum 序列级，不考虑 length normalisation）。"""
    s_w = student_logp_w.sum()
    s_l = student_logp_l.sum()
    r_w = ref_logp_w.sum()
    r_l = ref_logp_l.sum()
    logits = beta * ((s_w - r_w) - (s_l - r_l))
    return -F.logsigmoid(logits)


def ppo_token_loss(
    student_logp: torch.Tensor,      # [R]   当前 student logπ(y_t|...)
    rollout_logp: torch.Tensor,      # [R]   rollout 时的 logπ_old(y_t|...)，detached
    advantage: float,                # 标量 A_i = reward_i - baseline
    *,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """单 rollout 的 PPO clipped surrogate (token mean)。

    L = -mean_t min(ratio * A, clip(ratio,1-e,1+e) * A)
    advantage 是标量，broadcast 到 token 维度。
    """
    ratio = torch.exp(student_logp - rollout_logp.detach())
    A = student_logp.new_tensor(advantage)
    a = ratio * A
    b = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * A
    per_tok = -torch.minimum(a, b)
    return per_tok.mean()


def reinforce_loss(
    student_logp: torch.Tensor,      # [R]
    advantage: float,
) -> torch.Tensor:
    """REINFORCE / GRPO（无 clip 简化版）：
    L = -mean(logπ(y_t)) * A
    HF backend 不返回 rollout 时的 logprobs，所以严格 PPO ratio 算不出来；
    退化到 REINFORCE 是最干净的对照实现。
    """
    A = student_logp.new_tensor(advantage)
    return -(student_logp.mean() * A)


def grpo_normalise_advantages(rewards: List[float],
                              eps: float = 1e-6) -> List[float]:
    """GRPO 风格：组内 reward 减均值除标准差。"""
    if not rewards:
        return []
    import statistics
    if len(rewards) == 1:
        return [0.0]
    m = sum(rewards) / len(rewards)
    s = statistics.pstdev(rewards) + eps
    return [(r - m) / s for r in rewards]


# ============================================================
# Fixed-DPO variants (to prevent length collapse in long-form generation)
# ============================================================

def dpo_pair_loss_simpo(
    student_logp_w: torch.Tensor,    # [Rw]
    student_logp_l: torch.Tensor,    # [Rl]
    *,
    beta: float = 2.5,
    gamma: float = 0.5,
) -> torch.Tensor:
    """SimPO loss (Meng et al. 2024): reference-free, length-normalized.

    L = -log σ( β * (logπ(yw)/|yw| - logπ(yl)/|yl| - γ) )

    Fixes DPO length collapse by normalizing logπ by token count,
    making the loss insensitive to sequence length. Works without a
    reference model, which also removes the pathology where long
    responses over-dominate the reference-policy term.

    Recommended defaults from SimPO paper: β=2.5, γ=0.5 (reward margin).
    """
    s_w = student_logp_w.sum() / max(student_logp_w.numel(), 1)
    s_l = student_logp_l.sum() / max(student_logp_l.numel(), 1)
    logits = beta * (s_w - s_l - gamma)
    return -F.logsigmoid(logits)


def dpo_pair_loss_sft_aux(
    student_logp_w: torch.Tensor,
    student_logp_l: torch.Tensor,
    ref_logp_w: torch.Tensor,
    ref_logp_l: torch.Tensor,
    sft_logp: torch.Tensor,           # [Rg]  logπ(y_gold)
    *,
    beta: float = 0.1,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Standard DPO + SFT auxiliary loss on gold response (Pang et al. 2024).

    L = L_dpo + α * (-mean(logπ(y_gold)))

    The SFT aux term pins the policy to the gold distribution, preventing
    schema drift while DPO optimizes preferences. α=0.1 is the standard
    mixing coefficient used in Iterative RPO and R-DPO.
    """
    dpo = dpo_pair_loss(student_logp_w, student_logp_l, ref_logp_w, ref_logp_l, beta=beta)
    sft = -sft_logp.mean() if sft_logp.numel() > 0 else sft_logp.new_tensor(0.0)
    return dpo + alpha * sft


def reinforce_loss_length_norm(
    student_logp: torch.Tensor,      # [R]
    advantage: float,
    *,
    length_penalty: float = 0.0,     # optional: subtract λ * |y| / len_ref from A
    len_ref: float = 1500.0,
) -> torch.Tensor:
    """REINFORCE/GRPO with length-normalized advantage.

    Motivation: plain REINFORCE with -mean(logπ) * A is length-invariant in
    gradient magnitude, but the reward signal often correlates with length
    (short responses get 0 for not-parsed, long responses get graded). This
    leads the policy to collapse to short outputs. Adding a mild length
    penalty (λ ~ 0.05) keeps the policy honest.
    """
    A_val = advantage
    if length_penalty > 0 and len_ref > 0:
        # penalize deviation from len_ref, encouraging neither collapse nor explosion
        dev = (student_logp.numel() - len_ref) / len_ref
        A_val = advantage - length_penalty * dev
    A = student_logp.new_tensor(A_val)
    return -(student_logp.mean() * A)
