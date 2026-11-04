"""Run a CPU example of schema conversion, DGR, and the distillation loss."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from sdpo_vlm.data.schema import roundtrip_diff, to_short
from sdpo_vlm.feedback.template import build_for_rollout
from sdpo_vlm.tools.score_predictions import score_records
from sdpo_vlm.trainer.sdpo_loss import topk_symmetric_jsd_loss


def main():
    document = json.loads((ROOT / "examples/oraclegraph/json/toy_page.json").read_text())
    assert not any(roundtrip_diff(document["oracle_graph"]))
    gold = to_short(document["oracle_graph"])
    weak = {"n": gold["n"][:1], "e": []}
    feedback = build_for_rollout(
        image_path=str(ROOT / "examples/oraclegraph/images/toy_page.jpg"),
        rollout={"response_text": json.dumps(weak), "reward": 0.0},
        gold_short=gold,
        mode="auto",
    )
    print(feedback["feedback_text"])
    assert feedback["correct_solution_json"] is not None

    torch.manual_seed(42)
    student = torch.randn(8, 128, requires_grad=True)
    teacher = torch.randn(8, 128)
    loss = topk_symmetric_jsd_loss(student, teacher, top_k=100)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    print(f"Distillation loss: {loss.item():.6f}; gradients finite")
    records = json.loads((ROOT / "examples/predictions.json").read_text())["results"]
    print(json.dumps(score_records(records)["summary"], indent=2))
    print("CPU smoke test passed (no model download or GPU required).")


if __name__ == "__main__":
    main()
