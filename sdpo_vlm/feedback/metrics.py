"""Reward 与 per-type P/R 计算。

reward = min(macro_P, macro_R)，macro 在"节点类型桶 + 边类型桶"上求平均。
特殊：若 pred JSON 无法解析 → reward=0；若 gold 某桶为空且 pred 也为空 → 该桶
在 macro 里直接跳过（避免 0/0 占位）。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from .matcher import match_edges, match_nodes


def compute_reward(
    gold_short: Dict,
    pred_str: str,
) -> Dict:
    """
    Returns: {
      "reward": float in [0,1],
      "macro_p": float, "macro_r": float,
      "per_type_pr": {type_label: (p, r, tp, fp, fn)},
      "parsed": bool,
      "pred_short": dict or None,
      "errors": [str],
    }
    """
    errors: List[str] = []
    try:
        pred = json.loads(pred_str)
        if not isinstance(pred, dict) or "n" not in pred or "e" not in pred:
            raise ValueError("missing 'n' or 'e'")
        # 兼容：只有 n/e 即可
        pred_short = {
            "n": pred.get("n", []),
            "e": pred.get("e", []),
        }
        # 容错：每项必须是 dict 且包含必需字段
        pred_short["n"] = [x for x in pred_short["n"]
                           if isinstance(x, dict) and "t" in x and "box" in x
                           and isinstance(x.get("box"), list) and len(x["box"]) >= 4]
        pred_short["e"] = [x for x in pred_short["e"]
                           if isinstance(x, dict) and "s" in x and "d" in x and "t" in x]
        # gold_to_pred 需要 pred 节点有 id；为没有 id 的合成一个
        for k, n in enumerate(pred_short["n"]):
            if "id" not in n:
                n["id"] = f"_anon_{k}"
        parsed = True
    except Exception as exc:
        errors.append(f"无法解析为合法 JSON: {exc}")
        return {
            "reward": 0.0,
            "macro_p": 0.0, "macro_r": 0.0,
            "per_type_pr": {},
            "parsed": False,
            "pred_short": None,
            "errors": errors,
            "node_match": None,
            "edge_match": None,
        }

    nm = match_nodes(gold_short["n"], pred_short["n"])
    em = match_edges(gold_short["e"], pred_short["e"], nm["gold_to_pred"])

    per_type_pr: Dict[str, Tuple[float, float, int, int, int]] = {}

    # 节点桶
    for t, info in nm["by_type"].items():
        tp = len(info["pairs"])
        fn = len(info["unmatched_g"])
        fp = len(info["unmatched_p"])
        p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
        r = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
        per_type_pr[f"node:{t}"] = (p, r, tp, fp, fn)

    # 边桶
    for t, info in em["by_type"].items():
        tp, fp, fn = info["tp"], info["fp"], info["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
        r = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
        per_type_pr[f"edge:{t}"] = (p, r, tp, fp, fn)

    # macro：gold 和 pred 都空的桶不参与
    buckets = []
    for k, (p, r, tp, fp, fn) in per_type_pr.items():
        if tp + fp + fn == 0:
            continue
        buckets.append((p, r))
    if buckets:
        macro_p = sum(b[0] for b in buckets) / len(buckets)
        macro_r = sum(b[1] for b in buckets) / len(buckets)
    else:
        macro_p = macro_r = 0.0

    reward = min(macro_p, macro_r)

    return {
        "reward": reward,
        "macro_p": macro_p, "macro_r": macro_r,
        "per_type_pr": per_type_pr,
        "parsed": True,
        "pred_short": pred_short,
        "errors": errors,
        "node_match": nm,
        "edge_match": em,
    }


def pretty_per_type(per_type_pr: Dict) -> str:
    lines = []
    for k, (p, r, tp, fp, fn) in sorted(per_type_pr.items()):
        lines.append(f"  {k}: P={p:.2f} R={r:.2f} (tp={tp}, fp={fp}, fn={fn})")
    return "\n".join(lines)
