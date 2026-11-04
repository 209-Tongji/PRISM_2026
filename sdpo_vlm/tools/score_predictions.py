"""Score saved predictions using the pooled structural evaluation protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdpo_vlm.feedback.structural import (
    _parse_prediction,
    aggregate_rows,
    strict_page_score,
)


def score_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No evaluation records to score")
    ids = [record["image_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image_id values; merge shards without overlap")
    scored = []
    for record in records:
        pred, parsed = _parse_prediction(record["pred"])
        row = strict_page_score(record["gold_short"], pred, parsed)
        scored.append({"image_id": record["image_id"], **row})
    summary = {
        "n": len(scored),
        "structural_micro_f1": aggregate_rows(scored, "strict_micro_f1"),
        "node_precision": aggregate_rows(scored, "node_precision"),
        "node_recall": aggregate_rows(scored, "node_recall"),
        "edge_precision": aggregate_rows(scored, "edge_precision"),
        "edge_recall": aggregate_rows(scored, "edge_recall"),
        "parsed_pct": 100 * sum(row["parsed"] for row in scored) / len(scored),
    }
    return {"summary": summary, "results": scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("partial"):
        raise ValueError("Partial predictions cannot be reported as a complete evaluation")
    if isinstance(payload, dict) and payload.get("summary", {}).get("complete") is False:
        raise ValueError("Evaluation contains failed or missing pages; rerun them first")
    records = payload["results"] if isinstance(payload, dict) else payload
    output = score_records(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
