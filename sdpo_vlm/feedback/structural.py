"""Exact-type/IoU structural scoring with counts pooled across pages.

The matching and aggregation functions are extracted from the audited PRISM
structural evaluation. The training reward is in feedback.metrics.
"""
from __future__ import annotations

import json
import statistics
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _bbox(values) -> List[float]:
    try:
        xs = [float(v) for v in values]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0, 0.0]
    if len(xs) == 4:
        return xs
    if len(xs) >= 8:
        return [min(xs[0::2]), min(xs[1::2]), max(xs[0::2]), max(xs[1::2])]
    return [0.0, 0.0, 0.0, 0.0]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = _bbox(a)
    bx1, by1, bx2, by2 = _bbox(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _parse_prediction(raw: str) -> Tuple[Dict, bool]:
    try:
        pred = json.loads(raw)
        if not isinstance(pred, dict):
            return {"n": [], "e": []}, False
        nodes = pred.get("n", [])
        edges = pred.get("e", [])
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return {"n": [], "e": []}, False
        return {"n": nodes, "e": edges}, True
    except Exception:
        return {"n": [], "e": []}, False


def strict_page_score(gold: Dict, pred: Dict, parsed: bool) -> Dict[str, float]:
    gold_nodes = [
        n for n in gold.get("n", [])
        if isinstance(n, dict) and "t" in n and "box" in n
    ]
    pred_nodes = [
        n for n in pred.get("n", [])
        if isinstance(n, dict) and "t" in n and "box" in n
    ]

    gold_to_pred: Dict[str, str] = {}
    node_pairs: List[Tuple[int, int]] = []
    node_types = {n["t"] for n in gold_nodes} | {n["t"] for n in pred_nodes}
    for node_type in node_types:
        gold_idx = [i for i, n in enumerate(gold_nodes) if n["t"] == node_type]
        pred_idx = [i for i, n in enumerate(pred_nodes) if n["t"] == node_type]
        if not gold_idx or not pred_idx:
            continue
        cost = np.ones((len(gold_idx), len(pred_idx)), dtype=np.float64)
        for i, gi in enumerate(gold_idx):
            for j, pj in enumerate(pred_idx):
                cost[i, j] = 1.0 - _iou(
                    gold_nodes[gi]["box"], pred_nodes[pj]["box"]
                )
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            overlap = 1.0 - cost[row, col]
            if overlap < 0.5:
                continue
            gi = gold_idx[row]
            pj = pred_idx[col]
            node_pairs.append((gi, pj))
            gold_to_pred[gold_nodes[gi].get("id")] = pred_nodes[pj].get("id")

    node_tp = len(node_pairs)
    node_fp = len(pred_nodes) - node_tp
    node_fn = len(gold_nodes) - node_tp

    mapped_gold_edges = []
    for edge in gold.get("e", []):
        if not isinstance(edge, dict):
            continue
        source = gold_to_pred.get(edge.get("s"))
        target = gold_to_pred.get(edge.get("d"))
        if source is not None and target is not None:
            mapped_gold_edges.append((source, target, edge.get("t")))
    pred_edges = {
        (edge.get("s"), edge.get("d"), edge.get("t"))
        for edge in pred.get("e", [])
        if isinstance(edge, dict)
    }
    mapped_gold_set = set(mapped_gold_edges)
    edge_tp = len(mapped_gold_set & pred_edges)
    edge_fp = len(pred_edges - mapped_gold_set)
    edge_fn = len(gold.get("e", [])) - edge_tp

    node_p, node_r, node_f1 = _prf(node_tp, node_fp, node_fn)
    edge_p, edge_r, edge_f1 = _prf(edge_tp, edge_fp, edge_fn)
    micro_p, micro_r, micro_f1 = _prf(
        node_tp + edge_tp, node_fp + edge_fp, node_fn + edge_fn
    )
    return {
        "parsed": float(parsed),
        "node_tp": node_tp,
        "node_fp": node_fp,
        "node_fn": node_fn,
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "node_precision": node_p,
        "node_recall": node_r,
        "node_f1": node_f1,
        "edge_precision": edge_p,
        "edge_recall": edge_r,
        "edge_f1": edge_f1,
        "strict_micro_precision": micro_p,
        "strict_micro_recall": micro_r,
        "strict_micro_f1": micro_f1,
        "gold_nodes": len(gold_nodes),
        "gold_edges": len(gold.get("e", [])),
    }


def aggregate_rows(rows: Iterable[Dict], metric: str) -> float:
    rows = list(rows)
    if metric == "strict_micro_f1":
        tp = sum(row["node_tp"] + row["edge_tp"] for row in rows)
        fp = sum(row["node_fp"] + row["edge_fp"] for row in rows)
        fn = sum(row["node_fn"] + row["edge_fn"] for row in rows)
        return _prf(tp, fp, fn)[2]
    if metric == "node_precision":
        return _prf(
            sum(row["node_tp"] for row in rows),
            sum(row["node_fp"] for row in rows),
            sum(row["node_fn"] for row in rows),
        )[0]
    if metric == "node_recall":
        return _prf(
            sum(row["node_tp"] for row in rows),
            sum(row["node_fp"] for row in rows),
            sum(row["node_fn"] for row in rows),
        )[1]
    if metric == "edge_precision":
        return _prf(
            sum(row["edge_tp"] for row in rows),
            sum(row["edge_fp"] for row in rows),
            sum(row["edge_fn"] for row in rows),
        )[0]
    if metric == "edge_recall":
        return _prf(
            sum(row["edge_tp"] for row in rows),
            sum(row["edge_fp"] for row in rows),
            sum(row["edge_fn"] for row in rows),
        )[1]
    return statistics.mean(row[metric] for row in rows)
