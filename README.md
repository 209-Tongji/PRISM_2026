# PRISM

Core implementation of **PRISM: A Reward-Aware Self-Distillation Recipe for
Ancient Chinese Document Structural Parsing with Sub-10B MLLMs**.

PRISM trains a vision-language model to turn a document image into an
OracleGraph JSON graph. Its self-distillation objective combines a hindsight
teacher, a Diagnostic Graph Report (DGR), softmax reward weighting, and a
length-normalized auxiliary preference loss. The reference backbone is
`Qwen/Qwen3-VL-8B-Instruct` with LoRA adapters.

[中文说明](README_zh.md) · [Method and evaluation](docs/method.md) ·
[Data format and generation](docs/data.md)

## Included code

| Location | Purpose |
| --- | --- |
| `sdpo_vlm/trainer/trainer.py` | PRISM/SDPO training loop, reward weighting, auxiliary preference |
| `sdpo_vlm/trainer/sdpo_loss.py` | Top-K plus tail Jensen–Shannon divergence |
| `sdpo_vlm/trainer/teacher.py` | EMA, frozen, and trust-region teacher updates |
| `sdpo_vlm/feedback/` | Node/edge matching, DGR, training reward, structural evaluation |
| `sdpo_vlm/data/` | OracleGraph schema, student/teacher prompts, dataset loader |
| `sdpo_vlm/rollout/` | Hugging Face and optional SGLang rollout backends |
| `sdpo_vlm/trainer/trainer_sft.py` | Supervised initialization |
| `sdpo_vlm/trainer/trainer_baseline.py` | DPO, SimPO, and group-relative policy-gradient controls |
| `sdpo_vlm/tools/` | Generation evaluation and scoring saved predictions |
| `configs/` | Portable PRISM, baseline, and component ablation configurations |
| `scripts/` | Procedural data generator, adapter merge, CPU smoke example |

This repository contains source code and a small, constructed example. Full
datasets, rendering fonts/backgrounds, model weights, and trained adapters are
not bundled. The 5K main dataset used by this implementation is procedurally
generated; see [data provenance](docs/data.md).

## Installation

Use Python 3.10 or newer in a dedicated environment. Install a PyTorch build
compatible with your CUDA runtime, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The release targets the Transformers 4.57 series (`>=4.57.1,<5`). Training uses
a single process; `model.device_map` can distribute model layers across GPUs.
The default configuration places training on `cuda:0` and a separate rollout
model on `cuda:1`. Memory requirements depend on image resolution and response
length, so configure the GPU placement for your hardware. SDPA is the default
attention implementation; FlashAttention is optional.

Run the core checks without downloading a model:

```bash
python -m pip install pytest
python -m pytest -q
python scripts/smoke_test.py
```

The smoke example checks schema conversion, DGR construction, a differentiable
self-distillation loss, and structural scoring on CPU. It does not train the
8B model or reproduce benchmark results.

## Prepare data

Place paired files in the following layout:

```text
data/oraclegraph/
├── images/<image_id>.jpg
└── json/<image_id>.json
```

Each annotation contains an `oracle_graph` object with `nodes` and `edges`.
The loader converts the full graph to the compact training representation.
See `examples/oraclegraph/` for a complete one-page example and
[docs/data.md](docs/data.md) for the schema and generator command.

## Train

Run commands from the repository root. Environment variables override paths;
algorithm parameters live in YAML. Both SFT and PRISM use the same fixed
`data.split_seed=42` and `train_val_split=0.98` by default.

```bash
export PRISM_MODEL=Qwen/Qwen3-VL-8B-Instruct
export PRISM_DATA=/path/to/oraclegraph
export PRISM_OUTPUT=outputs
export PRISM_ROLLOUT_DEVICE=cuda:1

# 1. Supervised initialization: 1,000 updates, LoRA rank 32.
python -m sdpo_vlm.trainer.trainer_sft --config configs/sft.yaml --train-device cuda:0

# 2. Continue with PRISM: 100 updates from the SFT adapter.
export PRISM_SFT_ADAPTER=outputs/sft_seed42/ckpt_step001000/student
python -m sdpo_vlm.trainer.trainer --config configs/prism.yaml --train-device cuda:0
```

For another training seed, change `train.seed` and `train.run_name` in a local
config, train that seed's SFT adapter, and point `PRISM_SFT_ADAPTER` to it.
Keep `data.split_seed` fixed when comparing methods. The audited multi-seed
protocol uses seeds `42, 7, 2024, 123, 3407`; the default config is one run.

Component controls use the same PRISM trainer:

| Config | Change relative to PRISM |
| --- | --- |
| `sdpo_same_teacher.yaml` | Uniform rollout weighting and no auxiliary preference; same teacher input |
| `no_dgr.yaml` | Remove diagnostic text, retain reference-solution selection |
| `no_gold.yaml` | L3 diagnostic instances without a full gold/group solution |
| `no_reward_weighting.yaml` | Uniform rollout weights |
| `no_auxiliary.yaml` | Disable auxiliary preference |
| `frozen_teacher.yaml` | Freeze the initialized teacher |
| `online_teacher.yaml` | Use the current student as the feedback-conditioned teacher |

The separate baseline entry point is:

```bash
python -m sdpo_vlm.trainer.trainer_baseline --config configs/dpo.yaml --train-device cuda:0
# Other configs: configs/simpo.yaml, configs/grpo.yaml
```

The HF backend does not record rollout log probabilities. Consequently, the
group-relative control uses normalized REINFORCE; importance-ratio clipping is
only applied where old-policy log probabilities are available. See
[method details](docs/method.md) for the precise objectives.

## Evaluate

Generate predictions with a fixed split and deterministic decoding:

```bash
python -m sdpo_vlm.tools.eval_val \
  --base "$PRISM_MODEL" \
  --adapter outputs/prism_seed42/ckpt_step000100/student \
  --data "$PRISM_DATA" \
  --val-split 0.98 --seed 42 --max-samples 100 \
  --max-new-tokens 3072 --temperature 0 \
  --repetition-penalty 1.05 --no-repeat-ngram-size 20 \
  --device cuda:0 --attn sdpa --out outputs/eval/prism.json
```

The output preserves predictions, targets, per-page training reward, and a
separate `summary.structural` section. The structural metric matches nodes by
type and bounding-box IoU, matches typed edges through those node assignments,
and pools TP/FP/FN over pages before computing micro-F1. It differs from the
training reward `min(macro_precision, macro_recall)`.

Score saved predictions without loading a model:

```bash
python -m sdpo_vlm.tools.score_predictions \
  --input examples/predictions.json --out outputs/example_scores.json
```

Evaluations with skipped pages are marked incomplete and exit with an error.
Report per-seed pooled structural scores before averaging across training
seeds. [docs/method.md](docs/method.md) describes the evaluation conventions.

## Export an adapter

```bash
python scripts/merge_adapter.py \
  --base "$PRISM_MODEL" \
  --adapter outputs/prism_seed42/ckpt_step000100/student \
  --out outputs/prism_merged
```

## Release notes

The release preserves the PRISM loss and feedback implementation while
extracting it from the research workspace. It retains fixed data splits and
the corrected No-DGR teacher prompt, uses explicit adapter loading, and adds
portable configuration, CPU examples, and evaluation completeness checks.
The original import path `sdpo_vlm` is retained for compatibility.

## License

This repository uses the [Apache License 2.0](LICENSE).
