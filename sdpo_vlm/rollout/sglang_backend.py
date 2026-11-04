"""SGLang HTTP client backend（阶段 2）。

依赖：`pip install "sglang[all]"`；外部用 scripts/sglang_server.sh 起 server。
该类通过 OpenAI-compat /v1/chat/completions 采样，/load_lora_adapter +
/unload_lora_adapter 做 LoRA 热加载。

约定：server 启动时 `--enable-lora --lora-target-modules q,k,v,o,gate,up,down`，
trainer 每 N step 把当前 student LoRA 导出到磁盘，调 `sync_adapter(path)`：
  1) 用版本化命名 `student_v{counter}` POST /load_lora_adapter
  2) 切换 `_active_lora_name` 到新名字（之后的 chat 请求会带 lora_path=新名）
  3) 把上 *上* 一份旧 adapter unload，避免占显存（保留前一份给可能的 in-flight）
"""
from __future__ import annotations

import base64
import json
import os
import threading
from typing import Dict, List, Optional

import requests

from .base import RolloutBackend, RolloutResult


def _encode_image_as_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b = f.read()
    ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


def _messages_for_openai(messages: List[Dict]) -> List[Dict]:
    """把我们的 messages 格式转成 OpenAI Vision API 兼容格式。"""
    out = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        new_content = []
        for c in content:
            if c.get("type") == "text" or "text" in c:
                new_content.append({"type": "text", "text": c["text"]})
            elif c.get("type") == "image" or "image" in c:
                path = c.get("image") or c.get("image_url")
                url = _encode_image_as_data_url(path) if os.path.exists(path) else path
                new_content.append({"type": "image_url", "image_url": {"url": url}})
        out.append({"role": role, "content": new_content})
    return out


class SGLangRolloutBackend(RolloutBackend):
    def __init__(self, base_url: str = "http://127.0.0.1:30000/v1",
                 timeout_s: int = 600,
                 lora_name_prefix: str = "student_v"):
        self.base_url = base_url.rstrip("/")
        # SGLang server 的根（剥掉 /v1）
        self.root_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._lora_name_prefix = lora_name_prefix
        self._sync_counter = 0
        self._active_lora_name: Optional[str] = None
        self._prev_lora_name: Optional[str] = None  # 前一份，留作过渡

    # ---- LoRA 同步 ----

    def sync_adapter(self, adapter_path: str) -> None:
        """把磁盘上的新 adapter load 进 server，并把 active 切换过去。

        旧 adapter 在 *下一次* sync 时才 unload，避免和 in-flight 请求冲突
        （即使 trainer 串行调用，server 内部的 batched scheduler 也可能还没排空）。
        """
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(adapter_path)

        with self._lock:
            self._sync_counter += 1
            new_name = f"{self._lora_name_prefix}{self._sync_counter}"

            # 1) load 新 adapter
            r = requests.post(
                self.root_url + "/load_lora_adapter",
                json={"lora_name": new_name, "lora_path": adapter_path},
                timeout=120,
            )
            r.raise_for_status()
            body = r.json()
            if not body.get("success", False):
                raise RuntimeError(f"load_lora_adapter failed: {body}")

            # 2) 卸载更早一份（保留前一份给 in-flight；只卸前前一份）
            to_unload = self._prev_lora_name
            self._prev_lora_name = self._active_lora_name
            self._active_lora_name = new_name

            if to_unload is not None:
                try:
                    requests.post(
                        self.root_url + "/unload_lora_adapter",
                        json={"lora_name": to_unload}, timeout=60,
                    )
                except Exception as exc:
                    print(f"[sglang_backend] WARN unload {to_unload}: {exc}")

            print(f"[sglang_backend] active LoRA → {new_name} ({adapter_path})")

    # ---- sample ----

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
        payload = {
            "model": kw.get("model", "default"),
            "messages": _messages_for_openai(messages),
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "logprobs": True,
            "top_logprobs": kw.get("top_logprobs", 5),
        }
        # 指定 active LoRA（如果同步过的话）。SGLang 通过 extra_body 传 OpenAI
        # 兼容字段以外的扩展参数。
        if self._active_lora_name is not None:
            payload["extra_body"] = {"lora_path": self._active_lora_name}

        r = requests.post(
            self.base_url + "/chat/completions",
            json=payload, timeout=self.timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        results: List[RolloutResult] = []
        for ch in data["choices"]:
            text = ch["message"]["content"] or ""
            finish = ch.get("finish_reason", "stop")
            lp_info = ch.get("logprobs", {}) or {}
            content_lps = lp_info.get("content") or []
            logprobs = [x.get("logprob", 0.0) for x in content_lps]
            results.append(RolloutResult(
                response_text=text,
                response_ids=[],
                logprobs=logprobs,
                finish_reason=finish,
                sampling_meta={"lora": self._active_lora_name},
            ))
        return results
