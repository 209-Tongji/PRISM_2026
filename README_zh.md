# PRISM 核心代码

本仓库提供论文 **PRISM: A Reward-Aware Self-Distillation Recipe for Ancient
Chinese Document Structural Parsing with Sub-10B MLLMs** 的核心实现。

PRISM 将古籍页面图像转换为 OracleGraph JSON，结合反馈条件下的自蒸馏教师、
诊断图报告（DGR）、奖励加权聚合和长度归一化的辅助偏好损失。
默认骨干为 `Qwen/Qwen3-VL-8B-Instruct`，采用 LoRA 训练。

## 安装与快速验证

使用 Python 3.10+，先安装适合本机 CUDA 的 PyTorch，然后执行：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip install pytest
python -m pytest -q
python scripts/smoke_test.py
```

最后两条命令在 CPU 上验证核心逻辑，无需下载模型。完整训练需要另行准备模型、
数据和 GPU；该示例不用于复现论文数值。

## 核心实现

| 文件/目录 | 内容 |
| --- | --- |
| `sdpo_vlm/trainer/trainer.py` | PRISM 主循环、奖励权重、辅助偏好损失 |
| `sdpo_vlm/trainer/sdpo_loss.py` | Top-K 加尾部桶的 JSD 损失 |
| `sdpo_vlm/trainer/teacher.py` | EMA、冻结与 trust-region 教师 |
| `sdpo_vlm/feedback/` | 节点/边匹配、DGR、训练奖励和结构评测 |
| `sdpo_vlm/data/` | 图结构转换、提示词和数据加载 |
| `sdpo_vlm/rollout/` | HF / SGLang 采样后端 |
| `configs/` | PRISM、SFT、对照方法和组件消融 |
| `scripts/generate_data.py` | 程序化 OracleGraph 页面生成器 |

## 训练与评测

数据按 `images/<id>.jpg` 与 `json/<id>.json` 配对，JSON 内含 `oracle_graph`。
一个完整格式示例位于 `examples/oraclegraph/`。

```bash
export PRISM_MODEL=Qwen/Qwen3-VL-8B-Instruct
export PRISM_DATA=/path/to/oraclegraph
export PRISM_ROLLOUT_DEVICE=cuda:1

python -m sdpo_vlm.trainer.trainer_sft --config configs/sft.yaml --train-device cuda:0
export PRISM_SFT_ADAPTER=outputs/sft_seed42/ckpt_step001000/student
python -m sdpo_vlm.trainer.trainer --config configs/prism.yaml --train-device cuda:0

python -m sdpo_vlm.tools.eval_val \
  --base "$PRISM_MODEL" \
  --adapter outputs/prism_seed42/ckpt_step000100/student \
  --data "$PRISM_DATA" --val-split 0.98 --seed 42 --max-samples 100 \
  --max-new-tokens 3072 --temperature 0 --repetition-penalty 1.05 \
  --no-repeat-ngram-size 20 --device cuda:0 --attn sdpa \
  --out outputs/eval/prism.json
```

默认训练进程在 `cuda:0`，独立采样模型在 `cuda:1`。可通过 YAML 中的
`model.device_map` 分配模型层；请按图像分辨率、序列长度与显存调整配置。
多随机种子比较时固定 `data.split_seed=42`，仅改变 `train.seed`，并为每个种子
加载对应的 SFT adapter。

训练奖励为类型桶上的 `min(macro_P, macro_R)`；结构评测在节点类型一致、
框 IoU ≥ 0.5 的匹配基础上，将所有页面的节点/边 TP、FP、FN 汇总后计算 micro-F1。
评测文件将二者分开保存。HF 采样后端未记录旧策略 log probability，因此对应的
组相对策略梯度对照采用归一化 REINFORCE。

仓库只包含核心源代码与构造的小样例。主 5K 数据是程序化生成的数据；完整数据、
字体/背景素材、模型权重与训练 adapter 不随源码打包。
完整使用说明见 [英文 README](README.md)、[方法说明](docs/method.md) 和
[数据说明](docs/data.md)。
