"""训练回调：periodic logging、adapter 同步、checkpoint save。"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class AdapterSyncCallback:
    """每 N step 把 student LoRA 导出到磁盘并通知 rollout backend 热加载。"""

    def __init__(self, rollout_backend, out_dir: str, n_sync: int,
                 adapter_name: str = "student"):
        self.rollout_backend = rollout_backend
        self.out_dir = out_dir
        self.n_sync = n_sync
        self.adapter_name = adapter_name
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    def maybe_sync(self, model, step: int) -> Optional[str]:
        if step == 0 or step % self.n_sync != 0:
            return None
        path = os.path.join(self.out_dir, f"adapter_step{step:06d}")
        # 把 student adapter 单独导出（peft 在多 adapter 场景下默认会保存当前
        # active 的；我们先 set_adapter，再 save_pretrained）
        prev_active = getattr(model, "active_adapter", None)
        try:
            model.set_adapter(self.adapter_name)
            model.save_pretrained(path, safe_serialization=True)
        finally:
            if prev_active is not None:
                try:
                    model.set_adapter(prev_active)
                except Exception:
                    pass
        if self.rollout_backend is not None:
            try:
                # peft 保存出来的 adapter 在子目录 adapter_name 下
                actual_path = path
                sub = os.path.join(path, self.adapter_name)
                if os.path.isdir(sub) and os.path.exists(
                    os.path.join(sub, "adapter_config.json")
                ):
                    actual_path = sub
                self.rollout_backend.sync_adapter(actual_path)
            except Exception as exc:
                print(f"[adapter_sync] WARN rollout sync failed: {exc}")
        return path


class ConsoleLogger:
    def __init__(self):
        self.t0 = time.time()

    def log(self, step: int, metrics: Dict[str, Any]):
        elapsed = time.time() - self.t0
        parts = [f"step={step}", f"elapsed={elapsed:.1f}s"]
        for k, v in metrics.items():
            if isinstance(v, float):
                # 对小量（lr、ratio 等）用科学计数法
                if v != 0 and abs(v) < 1e-3:
                    parts.append(f"{k}={v:.2e}")
                else:
                    parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts), flush=True)
