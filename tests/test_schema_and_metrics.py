import json
from pathlib import Path

import pytest

from sdpo_vlm.data.dataset import IsoGraphDataset
from sdpo_vlm.data.schema import roundtrip_diff, to_short
from sdpo_vlm.feedback.metrics import compute_reward
from sdpo_vlm.tools.score_predictions import score_records


ROOT = Path(__file__).resolve().parents[1]


def test_example_roundtrip_and_dataset_loading():
    root = ROOT / "examples/oraclegraph"
    doc = json.loads((root / "json/toy_page.json").read_text())
    assert not any(roundtrip_diff(doc["oracle_graph"]))
    dataset = IsoGraphDataset([str(root)], split="val", train_val_split=0)
    assert len(dataset) == 1
    assert dataset[0]["gold_short"] == to_short(doc["oracle_graph"])
    assert Path(dataset[0]["image_path"]).is_file()


def test_missing_dataset_fails_instead_of_empty_training(tmp_path):
    with pytest.raises(ValueError, match="No image/JSON pairs"):
        IsoGraphDataset([str(tmp_path)])


def test_perfect_and_invalid_predictions():
    gold = {
        "n": [{"id": "m1", "t": "MT", "box": [0, 0, 10, 10], "txt": "example"}],
        "e": [],
    }
    assert compute_reward(gold, json.dumps(gold))["reward"] == 1
    assert compute_reward(gold, "{invalid")["reward"] == 0
    result = score_records([{"image_id": "page", "gold_short": gold, "pred": "{invalid"}])
    assert result["summary"]["structural_micro_f1"] == 0
    assert result["results"][0]["node_fn"] == 1


def test_pooled_micro_f1_counts_large_failed_page():
    one = {"id": "m1", "t": "MT", "box": [0, 0, 10, 10]}
    gold_small = {"n": [one], "e": []}
    gold_large = {"n": [dict(one, id=f"m{i}") for i in range(99)], "e": []}
    result = score_records([
        {"image_id": "small", "gold_short": gold_small, "pred": json.dumps(gold_small)},
        {"image_id": "large", "gold_short": gold_large, "pred": '{"n":[],"e":[]}'},
    ])
    assert result["summary"]["structural_micro_f1"] == pytest.approx(2 / 101)
    assert result["summary"]["structural_micro_f1"] != pytest.approx(0.5)


def test_edge_matching_uses_node_assignment_and_exact_types():
    gold = {"n": [
        {"id": "a", "t": "MT", "box": [0, 0, 10, 10]},
        {"id": "b", "t": "IA", "box": [20, 0, 30, 10]},
    ], "e": [{"s": "b", "d": "a", "t": "A"}]}
    pred = {"n": [dict(gold["n"][0], id="x"), dict(gold["n"][1], id="y")],
            "e": [{"s": "y", "d": "x", "t": "A"}]}
    record = {"image_id": "page", "gold_short": gold, "pred": json.dumps(pred)}
    assert score_records([record])["summary"]["structural_micro_f1"] == 1
    pred["e"][0]["t"] = "R"
    record["pred"] = json.dumps(pred)
    row = score_records([record])["results"][0]
    assert (row["edge_tp"], row["edge_fp"], row["edge_fn"]) == (0, 1, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        score_records([record, record])
