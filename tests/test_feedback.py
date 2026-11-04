"""Tests for the matched no-DGR ablation mode."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdpo_vlm.feedback.template import build_for_rollout


def test_auto_no_dgr_keeps_gold_solution_but_removes_diagnostic_text() -> None:
    gold = {
        "n": [{"id": "m1", "t": "MT", "box": [0, 0, 10, 10]}],
        "e": [],
    }
    result = build_for_rollout(
        image_path="/tmp/fake.jpg",
        rollout={"response_text": json.dumps({"n": [], "e": []}),
                 "reward": 0.0},
        gold_short=gold,
        mode="auto_no_dgr",
        group_rollouts=None,
        success_threshold=0.5,
        max_items=30,
        max_chars_per_item=200,
    )
    assert result["feedback_text"] == ""
    assert json.loads(result["correct_solution_json"]) == gold
    teacher_text = result["teacher_messages"][1]["content"][1]["text"]
    assert "【参考正确解】" in teacher_text
    assert "【反馈：你先前尝试的评估】" not in teacher_text
    assert "请参考上述正确解重新回答原问题" in teacher_text


def test_auto_no_dgr_preserves_group_success_solution_selection() -> None:
    gold = {
        "n": [{"id": "m1", "t": "MT", "box": [0, 0, 10, 10]}],
        "e": [],
    }
    weak = {
        "response_text": json.dumps({"n": [], "e": []}),
        "reward": 0.1,
    }
    strong = {
        "response_text": json.dumps(gold),
        "reward": 0.9,
    }
    result = build_for_rollout(
        image_path="/tmp/fake.jpg",
        rollout=weak,
        gold_short=gold,
        mode="auto_no_dgr",
        group_rollouts=[weak, strong],
        success_threshold=0.5,
        max_items=30,
        max_chars_per_item=200,
    )
    assert result["feedback_text"] == ""
    assert result["correct_solution_json"] == strong["response_text"]
    teacher_text = result["teacher_messages"][1]["content"][1]["text"]
    assert "【参考正确解】" in teacher_text
    assert "【反馈：你先前尝试的评估】" not in teacher_text
