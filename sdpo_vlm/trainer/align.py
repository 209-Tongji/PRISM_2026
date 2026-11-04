"""student 和 teacher 的 response token 对齐。

student  input = [sys, user, response]
teacher  input = [sys, user+feedback, response]

两次 forward 后分别定位 response_start，对 response 段 [start-1 : start-1+R]
取 logits（shifted 到 next-token 预测），因 response 相同所以两边 R 一致。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..data.prompt import find_response_start


def slice_response_logits(
    logits: torch.Tensor,        # [1, L, V]
    input_ids: List[int],
    processor,
    response_len: int,
) -> torch.Tensor:
    """返回 [R, V] 的 response 部分 next-token logits（R = response_len）。

    处理：
      - response_start = 第一个 response token 的位置
      - next-token logits 在位置 (response_start - 1)
      - 切出 [response_start-1 : response_start-1 + R]
    """
    start = find_response_start(processor, input_ids)
    end = start + response_len
    # 越界保护
    end = min(end, logits.shape[1])
    out = logits[0, start - 1 : end - 1, :]   # [R', V]
    if out.shape[0] != response_len:
        # response 可能被截断
        pad = response_len - out.shape[0]
        if pad > 0:
            out = torch.cat([out, out.new_zeros(pad, out.shape[-1])], dim=0)
    return out


def extract_response_ids(
    input_ids: List[int],
    processor,
    response_len: int,
) -> List[int]:
    """从完整 input_ids 里切出 response 段的 token id 序列（长度 = response_len）。"""
    start = find_response_start(processor, input_ids)
    end = min(start + response_len, len(input_ids))
    return input_ids[start:end]


def compute_current_logprobs(
    resp_logits: torch.Tensor,    # [R, V]
    response_ids: List[int],      # len = R
) -> torch.Tensor:
    """给一串 response token id，从对应 logits 算出每个 token 在 student
    当前策略下的 logprob（shape=[R]）。"""
    import torch.nn.functional as F
    R = resp_logits.shape[0]
    ids = torch.tensor(response_ids[:R], dtype=torch.long, device=resp_logits.device)
    if ids.shape[0] < R:
        # 不够长，末尾补 0（会和 mask 一起被忽略；上游应传 mask）
        pad = torch.zeros(R - ids.shape[0], dtype=torch.long, device=resp_logits.device)
        ids = torch.cat([ids, pad], dim=0)
    logp = F.log_softmax(resp_logits.float(), dim=-1)  # [R, V]
    return logp.gather(1, ids.unsqueeze(1)).squeeze(1)  # [R]


def assert_response_ids_equal(
    input_ids_student: List[int],
    input_ids_teacher: List[int],
    processor,
    response_len: int,
):
    """确认两边解出来的 response token 一致，避免对齐 bug。"""
    s = find_response_start(processor, input_ids_student)
    t = find_response_start(processor, input_ids_teacher)
    a = input_ids_student[s : s + response_len]
    b = input_ids_teacher[t : t + response_len]
    if a != b:
        raise RuntimeError(
            f"student/teacher response ids 不一致：\n"
            f"  student[{s}:{s+response_len}] = {a[:8]}...\n"
            f"  teacher[{t}:{t+response_len}] = {b[:8]}..."
        )
