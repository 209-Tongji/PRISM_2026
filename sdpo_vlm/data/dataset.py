"""IsoGraph 数据集加载器。

从 dataset_lite / dataset_smoke 读 (image.jpg, oracle_graph.json) 对；
产出每条：
  - image_path: str
  - image_id: str
  - gold_short: dict    # 缩写 schema（用于 reward 计算和可选的 correct_solution）
  - gold_full: dict     # 完整 OracleGraph（用于 to_short / 评测）
  - scenario: list[str] # adversarial_scenario
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from .schema import to_short


class IsoGraphDataset(Dataset):
    def __init__(
        self,
        roots: List[str],
        *,
        split: str = "train",
        train_val_split: float = 0.9,
        max_samples: Optional[int] = None,
        keep_full_polygon: bool = False,
        seed: int = 42,
    ):
        self.keep_full_polygon = keep_full_polygon
        items: List[Dict] = []
        for root in roots:
            r = Path(root)
            if not r.exists():
                continue
            img_dir = r / "images"
            jsn_dir = r / "json"
            for jp in sorted(jsn_dir.glob("*.json")):
                ip = img_dir / (jp.stem + ".jpg")
                if not ip.exists():
                    continue
                items.append({"image_path": str(ip), "json_path": str(jp),
                              "image_id": jp.stem})
        rng = random.Random(seed)
        rng.shuffle(items)
        n_train = int(len(items) * train_val_split)
        if split == "train":
            items = items[:n_train]
        elif split == "val":
            items = items[n_train:]
        else:
            raise ValueError(split)
        if max_samples is not None:
            items = items[:max_samples]
        self.items = items
        if not self.items:
            raise ValueError(
                f"No image/JSON pairs in split={split!r} under {roots}. "
                "Expected images/<id>.jpg and json/<id>.json; check the split ratio."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        doc = json.loads(Path(it["json_path"]).read_text(encoding="utf-8"))
        gold_full = doc["oracle_graph"]
        gold_short = to_short(gold_full, keep_full_polygon=self.keep_full_polygon)
        return {
            "image_id": it["image_id"],
            "image_path": it["image_path"],
            "gold_short": gold_short,
            "gold_full": gold_full,
            "scenario": doc.get("adversarial_scenario", ["NORMAL"]),
            "metadata": doc.get("metadata", {}),
        }


def simple_collate(batch: List[Dict]) -> List[Dict]:
    """SDPO 的 collate 不做 padding（rollout/forward 各自处理），直接返回 list。"""
    return list(batch)
