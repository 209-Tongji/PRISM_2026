"""按"反馈抽象层级"重新设计 DGR：L0–L4 五档，量化 feedback granularity。

L0: pass/fail（1 bit）
L1: error category（"你在边结构上出错" / "你节点输出多了" / "schema 错误" 等）
L2: per-bucket aggregate（每桶 P/R）
L3: actionable instances（漏检节点的 box+txt）
L4: full DGR + correct_solution
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .metrics import compute_reward


_TYPE_ZH = {
    "MT": "正文", "IA": "夹批", "TM": "天头批注", "BM": "地脚批注",
    "SM": "侧批", "CF": "版心文本", "RS": "红章",
}
_EDGE_ZH = {"R": "阅读顺序", "A": "批注指向", "O": "印章覆盖"}


def _fmt_box(b: List[int]) -> str:
    if not b or len(b) < 4: return "[?]"
    return f"[{b[0]},{b[1]}]-[{b[2]},{b[3]}]"


def build_feedback_at_level(
    gold_short: Dict, pred_str: str, *,
    level: str = "L3",          # L0/L1/L2/L3/L4
    scenario: Optional[List[str]] = None,
    max_examples: int = 5,
) -> Dict:
    """统一接口，按 level 控制 feedback 详细程度。"""
    res = compute_reward(gold_short, pred_str)
    if not res["parsed"]:
        # 不论哪一档，无法 parse 都给同一种 minimal feedback
        return {
            "feedback_text": "上一轮未返回合法 JSON，请严格输出符合 schema 的紧凑 JSON。",
            "reward": 0.0,
            "per_type_pr": {},
            "parsed": False,
        }
    reward = res["reward"]
    per_type_pr = res["per_type_pr"]

    if level == "L0":
        # 1 bit
        text = "【诊断报告】上一轮失败。" if reward < 0.5 else "【诊断报告】上一轮成功。"
        return _wrap(text, reward, per_type_pr)

    if level == "L1":
        # 错误类别（4 大类的哪一类有问题）
        worst = _category_diagnosis(per_type_pr)
        text = f"【诊断报告】{worst}"
        return _wrap(text, reward, per_type_pr)

    if level == "L2":
        # per-bucket P/R aggregate（无具体节点）
        lines = [f"【诊断报告】整体：reward={reward:.2f}"]
        for k, (p, r, tp, fp, fn) in sorted(per_type_pr.items()):
            if tp + fp + fn == 0: continue
            zh = _bucket_zh(k)
            lines.append(f"- {zh}：P={p:.2f} R={r:.2f}（漏 {fn} 错 {fp}）")
        return _wrap("\n".join(lines), reward, per_type_pr)

    if level == "L3":
        # actionable instances（具体漏检节点）
        lines = [f"【诊断报告】整体：reward={reward:.2f}"]
        nm = res.get("node_match")
        if nm:
            priority = ["MT", "IA", "TM", "BM", "CF", "RS", "SM"]
            added = 0
            for t in priority:
                if t not in nm["by_type"]: continue
                info = nm["by_type"][t]
                for gi in info["unmatched_g"]:
                    if added >= max_examples: break
                    n = gold_short["n"][gi]
                    zh = _TYPE_ZH.get(t, t)
                    txt = (n.get("txt") or "")[:20]
                    seg = (f"  * 漏检{zh} id={n['id']} 位置={_fmt_box(n['box'])}"
                           + (f" 文字~‘{txt}’" if txt else ""))
                    lines.append(seg)
                    added += 1
                if added >= max_examples: break
        return _wrap("\n".join(lines), reward, per_type_pr)

    if level == "L4":
        # full DGR + 在 template.py 里另外加 correct_solution
        # 这里只生成 full feedback；correct_solution 由 template 注入
        lines = [f"【诊断报告】整体：reward={reward:.2f}"]
        # per-bucket
        for k, (p, r, tp, fp, fn) in sorted(per_type_pr.items()):
            if tp + fp + fn == 0: continue
            zh = _bucket_zh(k)
            lines.append(f"- {zh}：P={p:.2f} R={r:.2f}（漏 {fn} 错 {fp}）")
        # actionable
        nm = res.get("node_match")
        if nm:
            priority = ["MT", "IA", "TM", "BM", "CF", "RS", "SM"]
            added = 0
            for t in priority:
                if t not in nm["by_type"]: continue
                info = nm["by_type"][t]
                for gi in info["unmatched_g"]:
                    if added >= max_examples: break
                    n = gold_short["n"][gi]
                    zh = _TYPE_ZH.get(t, t)
                    txt = (n.get("txt") or "")[:30]
                    seg = (f"  * 漏检{zh} id={n['id']} 位置={_fmt_box(n['box'])}"
                           + (f" 文字~‘{txt}’" if txt else ""))
                    lines.append(seg)
                    added += 1
        return _wrap("\n".join(lines), reward, per_type_pr)

    raise ValueError(f"unknown level: {level}")


def _category_diagnosis(per_type_pr: Dict) -> str:
    """选出表现最差的大类，给 L1 用。判断阈值更符合直觉。"""
    node_buckets = [(k, v) for k, v in per_type_pr.items() if k.startswith("node:")]
    edge_buckets = [(k, v) for k, v in per_type_pr.items() if k.startswith("edge:")]
    nodes_p = _avg_p(node_buckets)
    nodes_r = _avg_r(node_buckets)
    edges_p = _avg_p(edge_buckets)
    edges_r = _avg_r(edge_buckets)

    msgs = []
    # 边
    if edges_r < 0.5 and edge_buckets:
        msgs.append("你在边结构（节点之间的连接关系）上有较多漏检")
    elif edges_p < 0.5 and edge_buckets:
        msgs.append("你在边结构上有较多误报")
    # 节点
    if nodes_r < 0.5 and node_buckets:
        msgs.append("你在节点识别上有较多漏检")
    elif nodes_p < 0.5 and node_buckets:
        msgs.append("你在节点识别上有较多误报")
    if not msgs:
        msgs.append("整体接近正确，仍有少量错误")
    return "；".join(msgs) + "，请仔细检查并改进。"


def _avg_r(buckets):
    if not buckets: return 1.0
    rs = []
    for _, v in buckets:
        _, r, tp, fp, fn = v
        if tp + fp + fn > 0: rs.append(r)
    return sum(rs) / len(rs) if rs else 1.0


def _avg_p(buckets):
    if not buckets: return 1.0
    ps = []
    for _, v in buckets:
        p, _, tp, fp, fn = v
        if tp + fp + fn > 0: ps.append(p)
    return sum(ps) / len(ps) if ps else 1.0


def _bucket_zh(k):
    if k.startswith("node:"):
        return f"节点[{_TYPE_ZH.get(k[5:], k[5:])}]"
    return f"边[{_EDGE_ZH.get(k[5:], k[5:])}]"


def _wrap(text, reward, per_type_pr):
    return {
        "feedback_text": text,
        "reward": reward,
        "per_type_pr": per_type_pr,
        "parsed": True,
    }
