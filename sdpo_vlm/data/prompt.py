"""Qwen3-VL 的 chat 模板 + response span 定位。

只做两件事：
1. 构造 messages（system + user(image+text) + 可选 assistant prefill）
2. 对 tokenize 后的 input_ids 定位 response 的起止位置（用于 loss mask 和
   student/teacher logits 对齐）
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---- System / User 文案模板 ----

SYSTEM_TEXT = (
    "你是古籍图像解析助手。给定一页古籍扫描图，你需要输出该页的结构化图谱。"
    "只输出一个合法的 JSON（UTF-8，无多余文字），字段使用下方约定的缩写 schema。"
)

USER_TEXT_TEMPLATE = (
    "请按缩写 schema 输出这页的 OracleGraph。\n"
    "节点类型：MT=MAIN_TEXT, IA=INTERLINEAR_ANNOTATION, TM=TOP_MARGINALIA, "
    "BM=BOTTOM_MARGINALIA, SM=SIDE_MARGINALIA, CF=CENTER_FOLD_TEXT, RS=RED_SEAL_STAMP。\n"
    "边类型：R=READS_AFTER, A=ANNOTATES, O=OVERLAPS。\n"
    "返回字段：{\"n\":[{\"id\",\"t\",\"box\":[x1,y1,x2,y2],\"txt\"}], "
    "\"e\":[{\"s\",\"d\",\"t\"}]}。"
    "box 为外接矩形，txt 仅内容节点保留。"
    "只输出一个 JSON，不要多余文字。"
)

def build_student_messages(image_path: str) -> List[Dict]:
    """普通 student prompt：system + user(image + user_text)。"""
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": USER_TEXT_TEMPLATE},
        ]},
    ]


def build_teacher_messages(
    image_path: str,
    feedback_text: str,
    correct_solution_json: Optional[str] = None,
) -> List[Dict]:
    """self-teacher prompt：在 user 段里塞入 feedback（和可选的 correct_solution）。

    对应论文 Table 2：
      user_text + \n【参考正确解】\n{correct_solution}\n
      【反馈：你先前尝试的评估】\n{feedback_text}\n
      请重新正确地回答原问题。
    """
    parts = [USER_TEXT_TEMPLATE]
    if correct_solution_json:
        parts.append("\n【参考正确解】\n" + correct_solution_json)
    if feedback_text and feedback_text.strip():
        parts.append("\n【反馈：你先前尝试的评估】\n" + feedback_text.strip())
    if feedback_text and feedback_text.strip():
        instruction = "请结合上述反馈重新正确地回答原问题"
    elif correct_solution_json:
        instruction = "请参考上述正确解重新回答原问题"
    else:
        instruction = "请重新正确地回答原问题"
    parts.append(f"\n{instruction}，只输出一个合法 JSON。")
    teacher_user_text = "".join(parts)
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": teacher_user_text},
        ]},
    ]


# ---- Response span 定位 ----

_ASSISTANT_START_LITERAL = "<|im_start|>assistant\n"


def find_response_start(
    processor,
    input_ids: List[int],
) -> int:
    """在 input_ids 里找到最后一个 `<|im_start|>assistant\n` 之后的第一个
    token 索引。返回 int。

    实现：把 `<|im_start|>assistant\n` tokenize 成 id 序列，然后在 input_ids
    里从末尾向前搜这个子序列，返回子序列末尾 +1。
    """
    tokenizer = processor.tokenizer
    marker = tokenizer.encode(_ASSISTANT_START_LITERAL, add_special_tokens=False)
    if not marker:
        raise RuntimeError("无法 tokenize assistant 起始标记")

    if isinstance(input_ids, list):
        ids = input_ids
    else:
        ids = list(input_ids.tolist()) if hasattr(input_ids, "tolist") else list(input_ids)

    m_len = len(marker)
    # 从后向前搜，保证找到真正的 assistant 开头（而不是 prompt 里的字面量）
    for i in range(len(ids) - m_len, -1, -1):
        if ids[i : i + m_len] == marker:
            return i + m_len
    raise RuntimeError(
        f"在 input_ids 中找不到 assistant 起始标记 {marker}"
    )
