"""RolloutBackend 抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RolloutResult:
    response_text: str
    response_ids: List[int]
    logprobs: List[float] = field(default_factory=list)   # 每 token 的 logprob；可空
    finish_reason: str = "stop"
    sampling_meta: Dict = field(default_factory=dict)


class RolloutBackend(ABC):
    @abstractmethod
    def sample(
        self,
        *,
        image_path: str,
        messages: List[Dict],
        n: int,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kw,
    ) -> List[RolloutResult]:
        """对一个 (image, messages) 采 n 个 rollout。"""

    def sync_adapter(self, adapter_path: str) -> None:
        """热加载新 LoRA adapter。默认 no-op，backend 自行实现。"""
        return None

    def close(self) -> None:
        return None
