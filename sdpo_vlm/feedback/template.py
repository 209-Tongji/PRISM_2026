"""拼 self-teacher prompt，并合并 auto 模式逻辑：
- 若组内有 rollout 的 reward>=threshold → 用它当 correct_solution；否则用 gold_short（若允许）或不给
- feedback_text 总是加（来自 diff.build_feedback 或 diff_levels.build_feedback_at_level）
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..data.prompt import build_teacher_messages
from ..data.schema import short_to_json_str
from .diff import build_feedback
from .diff_levels import build_feedback_at_level
from .success import pick_group_best


def build_for_rollout(
    *,
    image_path: str,
    rollout: Dict,               # {"response_text", "reward", ...}
    gold_short: Dict,
    scenario: Optional[List[str]] = None,
    mode: str = "auto",          # diff | dgr | success | auto | auto_no_dgr | level
    feedback_level: Optional[str] = None,   # L0/L1/L2/L3/L4 — mode='level' 时启用
    group_rollouts: Optional[List[Dict]] = None,
    success_threshold: float = 0.8,
    max_items: int = 20,
    max_chars_per_item: int = 80,
    allow_gold_as_solution: bool = True,
) -> Dict:
    """返回 {"teacher_messages":..., "feedback_text":..., "correct_solution_json": str|None}"""
    # 1. 反馈文本
    if mode == "level" and feedback_level:
        fb = build_feedback_at_level(
            gold_short=gold_short,
            pred_str=rollout["response_text"],
            level=feedback_level,
            scenario=scenario,
            max_examples=max_items if max_items > 0 else 5,
        )
    else:
        fb = build_feedback(
            gold_short=gold_short,
            pred_str=rollout["response_text"],
            scenario=scenario,
            max_items=max_items,
            max_chars_per_item=max_chars_per_item,
        )
    feedback_text = (
        ""
        if mode == "auto_no_dgr"
        else fb["feedback_text"]
    )

    # 2. 选 correct_solution
    correct_solution_json: Optional[str] = None
    # L4 = full + correct_solution
    if mode == "level" and feedback_level == "L4":
        correct_solution_json = short_to_json_str(gold_short)
    elif mode in ("success", "auto", "auto_no_dgr") and group_rollouts:
        best = pick_group_best(group_rollouts, threshold=success_threshold)
        if best is not None and best is not rollout:
            correct_solution_json = best["response_text"]

    if correct_solution_json is None and mode in (
        "auto", "auto_no_dgr"
    ) and allow_gold_as_solution:
        correct_solution_json = short_to_json_str(gold_short)

    if mode == "diff":
        correct_solution_json = None
    elif mode == "dgr":
        correct_solution_json = None
    elif mode == "success":
        pass
    elif mode == "level" and feedback_level in ("L0", "L1", "L2", "L3"):
        # L0-L3 都不给 correct_solution，纯反馈对比
        correct_solution_json = None

    # 3. 拼 teacher prompt
    teacher_messages = build_teacher_messages(
        image_path=image_path,
        feedback_text=feedback_text,
        correct_solution_json=correct_solution_json,
    )

    return {
        "teacher_messages": teacher_messages,
        "feedback_text": feedback_text,
        "correct_solution_json": correct_solution_json,
        "feedback_reward": fb["reward"],
        "feedback_parsed": fb["parsed"],
        "feedback_per_type_pr": fb.get("per_type_pr", {}),
    }
