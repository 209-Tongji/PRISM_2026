"""基于 match 结果生成中文 DGR 诊断报告（作为 self-teacher 的 feedback f）。"""
from __future__ import annotations

from typing import Dict, List, Optional

from .metrics import compute_reward


_TYPE_ZH = {
    "MT": "正文",
    "IA": "行间批注（夹批）",
    "TM": "天头批注",
    "BM": "地脚批注",
    "SM": "侧批",
    "CF": "版心文本",
    "RS": "红章",
}
_EDGE_ZH = {
    "R": "阅读顺序 (READS_AFTER)",
    "A": "批注指向 (ANNOTATES)",
    "O": "印章覆盖 (OVERLAPS)",
}


def _fmt_box(b: List[int]) -> str:
    if not b or len(b) < 4:
        return "[?]"
    return f"[{b[0]},{b[1]}]-[{b[2]},{b[3]}]"


def build_feedback(
    gold_short: Dict,
    pred_str: str,
    *,
    scenario: Optional[List[str]] = None,
    max_items: int = 20,
    max_chars_per_item: int = 80,
) -> Dict:
    """给出 (feedback_text, reward, per_type_pr, parsed)。

    feedback_text 是 student 后续被 self-teacher 吃进去的 f。
    """
    res = compute_reward(gold_short, pred_str)
    lines: List[str] = ["【诊断报告】"]

    if not res["parsed"]:
        lines.append("上一轮未返回合法 JSON。请严格输出符合 schema 的紧凑 JSON，"
                     "字段包含 n / e，节点 {id,t,box,txt?}，边 {s,d,t}。不要多余说明。")
        if res["errors"]:
            lines.append("错误提示：" + res["errors"][0][:max_chars_per_item])
        return {
            "feedback_text": "\n".join(lines),
            "reward": 0.0,
            "per_type_pr": {},
            "parsed": False,
        }

    # 总体
    lines.append(
        f"整体：macro_P={res['macro_p']:.2f} macro_R={res['macro_r']:.2f} "
        f"reward={res['reward']:.2f}"
    )

    # 每桶细分
    n_items = 0
    for k, (p, r, tp, fp, fn) in sorted(res["per_type_pr"].items()):
        if tp + fp + fn == 0:
            continue
        if k.startswith("node:"):
            t = k[5:]
            zh = _TYPE_ZH.get(t, t)
            seg = f"- 节点[{zh}]：P={p:.2f} R={r:.2f}，漏检 {fn} 个、误报 {fp} 个。"
        else:
            t = k[5:]
            zh = _EDGE_ZH.get(t, t)
            seg = f"- 边[{zh}]：P={p:.2f} R={r:.2f}，漏 {fn} 条、错 {fp} 条。"
        lines.append(seg[:max_chars_per_item])
        n_items += 1
        if n_items >= max_items:
            break

    # 列举具体漏检的节点（按类型先正文再批注）
    nm = res["node_match"]
    if nm:
        priority = ["MT", "IA", "TM", "BM", "CF", "RS", "SM"]
        miss_added = 0
        for t in priority:
            if t not in nm["by_type"]:
                continue
            info = nm["by_type"][t]
            for gi in info["unmatched_g"]:
                if miss_added >= 5:
                    break
                n = gold_short["n"][gi]
                zh = _TYPE_ZH.get(t, t)
                txt = (n.get("txt") or "")[:20]
                seg = (f"  * 漏检{zh}节点 id={n['id']} 位置={_fmt_box(n['box'])}"
                       + (f" 文字~‘{txt}’" if txt else ""))
                lines.append(seg[:max_chars_per_item + 40])
                miss_added += 1

    # 场景提示
    if scenario:
        zh_map = {
            "DOUBLE_LINE_FLATTENING": "本页出现高密度行间批注，注意区分主文本与夹批",
            "SEAL_TEXT_INTRUSION": "本页有印章遮盖正文，注意补全被覆盖的字符、别漏 OVERLAPS",
            "CROSS_COLUMN_FUSION": "本页有列分隔线视觉缺失，注意仍按列分隔文本",
            "NOTE_MISANCHORING": "本页出现位置异常的批注，注意 ANNOTATES 指向正确",
            "NORMAL": "",
        }
        hints = [zh_map[s] for s in scenario if s in zh_map and zh_map[s]]
        if hints:
            lines.append("场景提示：" + "；".join(hints))

    return {
        "feedback_text": "\n".join(lines),
        "reward": res["reward"],
        "per_type_pr": res["per_type_pr"],
        "parsed": True,
    }
