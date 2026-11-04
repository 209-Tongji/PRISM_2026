"""缩写 schema 的节点/边匹配。

给 gold（short）和 pred（short），返回匹配结果。节点按 type 分桶后用 Hungarian
求最小代价匹配；代价 = 0.6*(1-char_overlap) + 0.4*(1-bbox_iou)。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def bbox_iou(a, b) -> float:
    """Robust bbox IoU. Coerces str→int and tolerates polygon (8 pts) by taking min/max."""
    def to_bbox(v):
        if v is None:
            return [0, 0, 0, 0]
        if not isinstance(v, (list, tuple)):
            return [0, 0, 0, 0]
        try:
            xs = [float(x) for x in v]
        except (TypeError, ValueError):
            return [0, 0, 0, 0]
        if len(xs) == 4:
            return [int(xs[0]), int(xs[1]), int(xs[2]), int(xs[3])]
        if len(xs) >= 8:
            # polygon: alternating x,y
            x_vals = xs[0::2]
            y_vals = xs[1::2]
            return [int(min(x_vals)), int(min(y_vals)), int(max(x_vals)), int(max(y_vals))]
        return [0, 0, 0, 0]
    ax1, ay1, ax2, ay2 = to_bbox(a)
    bx1, by1, bx2, by2 = to_bbox(b)
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def char_overlap(a: str, b: str) -> float:
    """按多重集字符交集算重合率。"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    return inter / max(len(a), len(b))


def _cost(gn: Dict, pn: Dict) -> float:
    # 只有同类型节点才会被拿来匹配（外层已分桶）
    txt_a = gn.get("txt", "")
    txt_b = pn.get("txt", "")
    ov = char_overlap(txt_a, txt_b)
    iou = bbox_iou(gn["box"], pn["box"])
    return 0.6 * (1 - ov) + 0.4 * (1 - iou)


def _greedy_match(gold: List[Dict], pred: List[Dict]) -> List[Tuple[int, int]]:
    """scipy 不可用时的 fallback：贪心。每次选当前代价最小的配对。"""
    remaining_g = set(range(len(gold)))
    remaining_p = set(range(len(pred)))
    pairs = []
    while remaining_g and remaining_p:
        best = None
        best_c = 1e9
        for i in remaining_g:
            for j in remaining_p:
                c = _cost(gold[i], pred[j])
                if c < best_c:
                    best_c = c
                    best = (i, j)
        if best is None or best_c >= 1.0:
            break
        pairs.append(best)
        remaining_g.discard(best[0])
        remaining_p.discard(best[1])
    return pairs


def match_nodes(
    gold_nodes: List[Dict],
    pred_nodes: List[Dict],
    *,
    cost_threshold: float = 0.9,
) -> Dict:
    """按 type 分桶后做二部图匹配。

    返回：{
      "by_type": {t: {"pairs": [(gold_idx, pred_idx, cost)], "unmatched_g":[i], "unmatched_p":[j]}},
      "gold_to_pred": {gold_id: pred_id},  # 方便后续边匹配
    }
    cost_threshold: 超过该代价视为不可匹配（即使 Hungarian 给了匹配）。
    """
    # 按类型分桶
    by_type_g: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
    by_type_p: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
    for i, n in enumerate(gold_nodes):
        by_type_g[n["t"]].append((i, n))
    for j, n in enumerate(pred_nodes):
        by_type_p[n["t"]].append((j, n))

    by_type: Dict[str, Dict] = {}
    gold_to_pred: Dict[str, str] = {}

    for t in set(list(by_type_g.keys()) + list(by_type_p.keys())):
        g_items = by_type_g.get(t, [])
        p_items = by_type_p.get(t, [])
        if not g_items and not p_items:
            continue
        pairs: List[Tuple[int, int, float]] = []
        unm_g = [g[0] for g in g_items]
        unm_p = [p[0] for p in p_items]

        if g_items and p_items:
            if HAVE_SCIPY:
                cost_mat = np.zeros((len(g_items), len(p_items)))
                for a, (_, gn) in enumerate(g_items):
                    for b, (_, pn) in enumerate(p_items):
                        cost_mat[a, b] = _cost(gn, pn)
                r, c = linear_sum_assignment(cost_mat)
                for a, b in zip(r, c):
                    if cost_mat[a, b] < cost_threshold:
                        pairs.append((g_items[a][0], p_items[b][0], float(cost_mat[a, b])))
            else:
                gi = [gn for _, gn in g_items]
                pi = [pn for _, pn in p_items]
                raw = _greedy_match(gi, pi)
                for a, b in raw:
                    c = _cost(gi[a], pi[b])
                    if c < cost_threshold:
                        pairs.append((g_items[a][0], p_items[b][0], c))

            # 更新 unmatched
            matched_g = {p[0] for p in pairs}
            matched_p = {p[1] for p in pairs}
            unm_g = [i for i in unm_g if i not in matched_g]
            unm_p = [j for j in unm_p if j not in matched_p]

        by_type[t] = {"pairs": pairs, "unmatched_g": unm_g, "unmatched_p": unm_p}
        for gi, pj, _ in pairs:
            gold_to_pred[gold_nodes[gi]["id"]] = pred_nodes[pj]["id"]

    return {"by_type": by_type, "gold_to_pred": gold_to_pred}


def match_edges(
    gold_edges: List[Dict],
    pred_edges: List[Dict],
    gold_to_pred: Dict[str, str],
) -> Dict:
    """把 gold 的端点通过 gold_to_pred 映射到 pred 空间后，按 (s,d,t) 三元组比对。

    返回：{"by_type": {edge_type: {"tp":n, "fp":n, "fn":n}}}
    """
    # 映射 gold 端点
    gold_set = set()
    gold_by_type: Dict[str, int] = defaultdict(int)
    for e in gold_edges:
        s = gold_to_pred.get(e["s"])
        d = gold_to_pred.get(e["d"])
        if s is None or d is None:
            continue  # gold 端点没被 pred 匹到 → 这条边直接算 FN
        gold_set.add((s, d, e["t"]))
        gold_by_type[e["t"]] += 1
    pred_set = set((e["s"], e["d"], e["t"]) for e in pred_edges
                   if isinstance(e, dict) and "s" in e and "d" in e and "t" in e)

    # 按类型分桶统计
    types = set(e["t"] for e in gold_edges) | set(
        e["t"] for e in pred_edges if isinstance(e, dict) and "t" in e
    )
    per_type: Dict[str, Dict[str, int]] = {}
    for t in types:
        g_t = {trip for trip in gold_set if trip[2] == t}
        p_t = {trip for trip in pred_set if trip[2] == t}
        # 未被 gold_to_pred 覆盖的 gold 边 → FN（原始 gold_edges 里 type=t 的数量
        #  减去 g_t 大小）
        full_g_count = sum(1 for e in gold_edges if e["t"] == t)
        tp = len(g_t & p_t)
        fp = len(p_t - g_t)
        # 漏的：没映射成功的也算 FN
        fn = full_g_count - tp
        per_type[t] = {"tp": tp, "fp": fp, "fn": fn}

    return {"by_type": per_type}
