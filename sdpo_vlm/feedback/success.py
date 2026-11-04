"""Mode C: 组内挑 best rollout 作为 self-teacher 的 correct_solution。"""
from __future__ import annotations

from typing import Dict, List, Optional


def pick_group_best(
    rollouts: List[Dict],
    *,
    threshold: float = 0.8,
) -> Optional[Dict]:
    """rollouts: 每条形如 {"response_text": str, "reward": float, ...}。
    若有 reward>threshold 的 rollout，返回其中 reward 最高者；否则 None。
    """
    eligible = [r for r in rollouts if r.get("reward", 0.0) >= threshold]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r["reward"])
