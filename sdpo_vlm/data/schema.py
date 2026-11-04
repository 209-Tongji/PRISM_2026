"""缩写 schema <-> 完整 OracleGraph 双向转换。

完整 OracleGraph 里包含"内容节点"（有文本或可直接看出）和"结构节点"（外框、
分隔线、鱼尾等页面骨架）。对 VLM 任务，我们只要求模型输出"内容节点 + 边"，
去掉骨架节点可大幅降低 token 预算。

缩写 schema:
  {
    "n": [ {"id": "m1", "t": "MT", "box":[x1,y1,x2,y2], "txt":"..."} ],
    "e": [ {"s":"m1", "d":"m2", "t":"R"} ]
  }

- id：短 node id，训练时直接用 gold 里的 n_main3 → m3（去掉 'n_' 和类型前缀）
- t：类型缩写（见 NODE_TYPE_ABBREV / EDGE_TYPE_ABBREV）
- box：外接矩形 [x1,y1,x2,y2]（从 polygon 的 8 个点求 min/max）
- txt：文本内容（仅内容节点才有）
- s/d/t：边的 source / destination / type

round-trip 约束：对 gold JSON 的"内容子集"做 to_short → to_full 后，
和原始"内容子集"在 { node_id, type, text, box外接矩形, edges（端点都是内容
节点的） } 上必须完全一致。
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

# ---- 节点类型映射 ----

# 模型需要输出的内容节点
CONTENT_NODE_TYPES = [
    "MAIN_TEXT",
    "INTERLINEAR_ANNOTATION",
    "TOP_MARGINALIA",
    "BOTTOM_MARGINALIA",
    "SIDE_MARGINALIA",
    "CENTER_FOLD_TEXT",
    "RED_SEAL_STAMP",
]

# 骨架节点：训练时从缩写 schema 中剔除
STRUCTURE_NODE_TYPES = [
    "OUTER_PAGE_FRAME",
    "COLUMN_SEPARATOR",
    "CENTER_FOLD_COLUMN",
    "FISHTAIL_MARK",
    "FOLD_TRUNK_LINE",
    "PUNCTUATION_MARK",  # generate_json.py 的遗留
]

NODE_TYPE_ABBREV = {
    "MAIN_TEXT":              "MT",
    "INTERLINEAR_ANNOTATION": "IA",
    "TOP_MARGINALIA":         "TM",
    "BOTTOM_MARGINALIA":      "BM",
    "SIDE_MARGINALIA":        "SM",
    "CENTER_FOLD_TEXT":       "CF",
    "RED_SEAL_STAMP":         "RS",
}
NODE_TYPE_FULL = {v: k for k, v in NODE_TYPE_ABBREV.items()}

# ---- 边类型映射 ----

EDGE_TYPE_ABBREV = {
    "READS_AFTER": "R",
    "ANNOTATES":   "A",
    "OVERLAPS":    "O",
}
EDGE_TYPE_FULL = {v: k for k, v in EDGE_TYPE_ABBREV.items()}


# ---- 工具函数 ----

def polygon_to_bbox(polygon: List[int]) -> List[int]:
    """polygon = [x1,y1,x2,y2,x3,y3,x4,y4] → [xmin,ymin,xmax,ymax]"""
    if len(polygon) < 2:
        return [0, 0, 0, 0]
    xs = polygon[0::2]
    ys = polygon[1::2]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def bbox_to_polygon(bbox: List[int]) -> List[int]:
    """[x1,y1,x2,y2] → 8 点顺时针闭合 polygon"""
    x1, y1, x2, y2 = bbox
    return [int(x1), int(y1), int(x2), int(y1),
            int(x2), int(y2), int(x1), int(y2)]


# id 映射：gold 里的 n_main3 / n_interlinear1 / n_seal2 → 短 id
# 为了 round-trip 可逆，我们保留一份"类型前缀 + 序号"的映射：
#   MT → m, IA → ia, TM → tm, BM → bm, SM → sm, CF → cf, RS → s
_PREFIX = {
    "MT": "m",  "IA": "ia", "TM": "tm", "BM": "bm",
    "SM": "sm", "CF": "cf", "RS": "s",
}
_PREFIX_REV = {v: k for k, v in _PREFIX.items()}


def _make_short_id(full_id: str, type_abbrev: str, counter: Dict[str, int]) -> str:
    """为每个内容节点生成稳定的短 id。同时把原 full_id 映射过去用于
    边的端点 rewrite。"""
    prefix = _PREFIX[type_abbrev]
    counter[prefix] = counter.get(prefix, 0) + 1
    return f"{prefix}{counter[prefix]}"


# ---- 主转换 ----

def to_short(oracle_graph: Dict, *, keep_full_polygon: bool = False) -> Dict:
    """完整 OracleGraph → 缩写 schema（只包含内容节点及其之间的边）。

    Args:
      oracle_graph: 形如 {"nodes": [...], "edges": [...]}（gold JSON 里的
        'oracle_graph' 字段）
      keep_full_polygon: 若 True，在 box 字段里保留 8 点 polygon 而非外接矩形
    """
    id_map: Dict[str, str] = {}   # 原 full_id → 新短 id
    short_nodes: List[Dict] = []
    counter: Dict[str, int] = {}

    for n in oracle_graph.get("nodes", []):
        t = n["type"]
        if t not in NODE_TYPE_ABBREV:
            # 跳过骨架节点
            continue
        tab = NODE_TYPE_ABBREV[t]
        sid = _make_short_id(n["node_id"], tab, counter)
        id_map[n["node_id"]] = sid

        if keep_full_polygon:
            box = list(n["polygon"])
        else:
            box = polygon_to_bbox(n["polygon"])

        item = {"id": sid, "t": tab, "box": box}
        # txt 只对有文本的节点有意义
        txt = n.get("text", "")
        if txt:
            item["txt"] = txt
        short_nodes.append(item)

    short_edges: List[Dict] = []
    for e in oracle_graph.get("edges", []):
        s, d, et = e["source"], e["target"], e["type"]
        if s not in id_map or d not in id_map:
            # 端点是骨架节点，drop 掉
            continue
        if et not in EDGE_TYPE_ABBREV:
            continue
        short_edges.append({
            "s": id_map[s],
            "d": id_map[d],
            "t": EDGE_TYPE_ABBREV[et],
        })

    return {"n": short_nodes, "e": short_edges}


def to_full(short: Dict, *, keep_full_polygon: bool = False) -> Dict:
    """缩写 schema → 完整 OracleGraph（只恢复内容节点 + 这些节点间的边）。

    恢复后的 node_id 格式与 gold 对齐：m1 → n_main1, ia3 → n_interlinear3 等。
    """
    full_nodes: List[Dict] = []
    id_map: Dict[str, str] = {}   # 短 id → 恢复后的 full id

    for n in short.get("n", []):
        tab = n["t"]
        if tab not in NODE_TYPE_FULL:
            continue
        full_type = NODE_TYPE_FULL[tab]
        sid = n["id"]
        # 短 id 形如 m3 → 数字部分
        idx = _parse_index(sid)
        full_id = _full_id_from(full_type, idx)
        id_map[sid] = full_id

        box = n.get("box", [0, 0, 0, 0])
        if keep_full_polygon and len(box) == 8:
            polygon = list(box)
        else:
            polygon = bbox_to_polygon(box[:4])

        full_nodes.append({
            "node_id": full_id,
            "polygon": polygon,
            "text": n.get("txt", ""),
            "type": full_type,
            "attributes": {},  # 缩写不保留 attributes
        })

    full_edges: List[Dict] = []
    for e in short.get("e", []):
        s, d, et = e.get("s"), e.get("d"), e.get("t")
        if s not in id_map or d not in id_map:
            continue
        if et not in EDGE_TYPE_FULL:
            continue
        full_edges.append({
            "source": id_map[s],
            "target": id_map[d],
            "type": EDGE_TYPE_FULL[et],
        })

    return {"nodes": full_nodes, "edges": full_edges}


def _parse_index(short_id: str) -> int:
    """从 'm3'、'ia10'、's2' 里取出尾部数字。"""
    for i, c in enumerate(short_id):
        if c.isdigit():
            return int(short_id[i:])
    return 0


def _full_id_from(full_type: str, idx: int) -> str:
    """按 merged_generator.py 的命名约定恢复 full_id。"""
    mapping = {
        "MAIN_TEXT":              "n_main",
        "INTERLINEAR_ANNOTATION": "n_interlinear",
        "TOP_MARGINALIA":         "n_top",
        "BOTTOM_MARGINALIA":      "n_bottom",
        "SIDE_MARGINALIA":        "n_side",
        "CENTER_FOLD_TEXT":       "n_centertext",
        "RED_SEAL_STAMP":         "n_seal",
    }
    return f"{mapping[full_type]}{idx}"


# ---- 用于训练/推理时的字符串 I/O ----

def short_to_json_str(short: Dict) -> str:
    """紧凑 JSON 输出（ASCII=false，保留中文；无多余空格）。"""
    return json.dumps(short, ensure_ascii=False, separators=(",", ":"))


def json_str_to_short(s: str) -> Dict:
    """模型输出的 JSON 字符串 → dict。
    若解析失败抛 ValueError，让上游处理（reward=0 并加 feedback）。"""
    return json.loads(s)


# ---- 自测 ----

def content_subset(oracle_graph: Dict) -> Dict:
    """抽取原图谱里的内容子集（nodes/edges 都只保留能映射到缩写 schema 的）。
    用于 round-trip 对比。"""
    nodes = []
    ids = set()
    for n in oracle_graph.get("nodes", []):
        if n["type"] in NODE_TYPE_ABBREV:
            nodes.append(n)
            ids.add(n["node_id"])
    edges = []
    for e in oracle_graph.get("edges", []):
        if e["type"] in EDGE_TYPE_ABBREV and e["source"] in ids and e["target"] in ids:
            edges.append(e)
    return {"nodes": nodes, "edges": edges}


def _norm_node_for_compare(n: Dict) -> Tuple:
    """归一化节点为可比较的元组（忽略 attributes 和 polygon 精确形状，
    只对外接矩形比较）。"""
    return (n["type"], n.get("text", ""), tuple(polygon_to_bbox(n["polygon"])))


def _norm_edge_for_compare(e: Dict, id_to_type: Dict[str, str],
                            id_to_bbox: Dict[str, Tuple]) -> Tuple:
    """边的归一化：用 (src_type, src_bbox, dst_type, dst_bbox, edge_type) 比较，
    避免因为 node_id 被改写（n_main3 → n_main1）导致误判不等。"""
    s = e["source"]; d = e["target"]
    return (
        id_to_type.get(s, "?"), id_to_bbox.get(s, ()),
        id_to_type.get(d, "?"), id_to_bbox.get(d, ()),
        e["type"],
    )


def roundtrip_diff(oracle_graph: Dict, *, keep_full_polygon: bool = False):
    """对 gold 图谱做 to_short → to_full，返回 (missing_nodes, extra_nodes,
    missing_edges, extra_edges)。round-trip 一致时全部为 []。"""
    subset = content_subset(oracle_graph)
    short = to_short(oracle_graph, keep_full_polygon=keep_full_polygon)
    restored = to_full(short, keep_full_polygon=keep_full_polygon)

    # 节点比较（用归一化元组）
    orig_set = {_norm_node_for_compare(n) for n in subset["nodes"]}
    new_set  = {_norm_node_for_compare(n) for n in restored["nodes"]}
    missing_nodes = sorted(orig_set - new_set)
    extra_nodes   = sorted(new_set - orig_set)

    def _id_to_type_and_bbox(nodes):
        t, b = {}, {}
        for n in nodes:
            t[n["node_id"]] = n["type"]
            b[n["node_id"]] = tuple(polygon_to_bbox(n["polygon"]))
        return t, b

    orig_t, orig_b = _id_to_type_and_bbox(subset["nodes"])
    new_t,  new_b  = _id_to_type_and_bbox(restored["nodes"])

    orig_edges = {_norm_edge_for_compare(e, orig_t, orig_b) for e in subset["edges"]}
    new_edges  = {_norm_edge_for_compare(e, new_t,  new_b)  for e in restored["edges"]}
    missing_edges = sorted(orig_edges - new_edges)
    extra_edges   = sorted(new_edges - orig_edges)

    return missing_nodes, extra_nodes, missing_edges, extra_edges
