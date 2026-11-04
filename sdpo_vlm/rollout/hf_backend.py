"""基于 HuggingFace transformers.generate 的 rollout backend（MVP）。

占用一张独立 GPU，加载 Qwen3-VL + 可热加载 LoRA adapter。
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

import torch
from PIL import Image
from .base import RolloutBackend, RolloutResult


def _load_model(model_path: str, device: str, dtype: torch.dtype):
    """Qwen3-VL 需要用 AutoModelForImageTextToText（新 transformers）。"""
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM

    try:
        m = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True,
            attn_implementation="sdpa",
        )
    except Exception:
        # 兼容旧版 transformers 的 Qwen2-VL / fallback
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True,
            attn_implementation="sdpa",
        )
    m = m.to(device)
    m.eval()
    return m


class HFRolloutBackend(RolloutBackend):
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:7",
        dtype: str = "bf16",
        max_batch: int = 2,
    ):
        from transformers import AutoProcessor

        self.model_path = model_path
        self.device = device
        self.dtype = torch.bfloat16 if dtype == "bf16" else (
            torch.float16 if dtype == "fp16" else torch.float32
        )
        self.max_batch = max_batch
        self._lock = threading.Lock()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model = _load_model(model_path, device, self.dtype)
        self._active_adapter: Optional[str] = None

    # ---- adapter 热加载 ----

    def sync_adapter(self, adapter_path: str) -> None:
        """PEFT: 把新 adapter 加载到 rollout model 上。

        保持接口简单：使用 model.load_adapter(adapter_path, adapter_name='rollout')
        然后 set_adapter('rollout')。旧 adapter 会被替换。
        """
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(adapter_path)

        # 延迟导入 peft（只有 sync 时才需要）
        try:
            from peft import PeftModel
        except ImportError:
            raise RuntimeError("peft 未安装")

        with self._lock:
            if isinstance(self.model, PeftModel):
                # 卸掉当前 adapter，加载新的到相同 name
                try:
                    self.model.delete_adapter("rollout")
                except Exception:
                    pass
                self.model.load_adapter(adapter_path, adapter_name="rollout")
                self.model.set_adapter("rollout")
            else:
                self.model = PeftModel.from_pretrained(
                    self.model, adapter_path,
                    adapter_name="rollout", is_trainable=False,
                )
                self.model.set_adapter("rollout")
            self.model.eval()
            self._active_adapter = adapter_path

    # ---- sample ----

    @torch.no_grad()
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
        # 1. apply_chat_template（文本）
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # 2. 把 image_path 填进来
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
        ).to(self.device)

        # 3. generate n 个（num_return_sequences）
        input_len = inputs["input_ids"].shape[1]
        # MVP：跳过 output_scores（占大量显存）；半在线 importance ratio 暂不用
        outs = self.model.generate(
            **inputs,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        seqs = outs.sequences  # [n, input_len + gen_len]
        results: List[RolloutResult] = []
        for i in range(n):
            gen = seqs[i, input_len:].tolist()
            # 去掉结尾的 pad 之后的 eos（如果有）
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and eos_id in gen:
                eos_idx = gen.index(eos_id)
                gen_ids = gen[:eos_idx]
                finish_reason = "stop"
            else:
                gen_ids = gen
                finish_reason = "length"
            text_out = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            results.append(RolloutResult(
                response_text=text_out,
                response_ids=gen_ids,
                logprobs=[],   # MVP 不算 logprobs；阶段 2 切 SGLang 后从 server 返回
                finish_reason=finish_reason,
                sampling_meta={"temperature": temperature, "top_p": top_p,
                               "adapter": self._active_adapter},
            ))
        return results
