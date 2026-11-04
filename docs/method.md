# Method and evaluation conventions

## Training objective

For each image, sample a group of responses from the rollout policy. A graph
comparison with the training target produces a reward and DGR. The student
conditions on the image and parsing instruction. The teacher also receives
the DGR and, in `auto` mode, a successful group response or a gold JSON fallback.
The teacher scores the **same response tokens** as the student.

The distillation loss uses the student's top-K vocabulary indices plus one
bucket containing the remaining probability mass. It computes the symmetric
Jensen–Shannon divergence between student and detached teacher distributions
on this shared support. `top_k=100` is the default. This implementation still
materializes vocabulary logits and probabilities; top-K does not remove the
full model forward or its memory requirements.

For rollout rewards `r_i`, PRISM assigns weights
`w_i = softmax(r_i / tau)`, with `tau=0.5`. The weighted mean of the rollout
distillation losses is combined with an auxiliary best-versus-worst loss:

```text
L_aux = -log sigmoid(beta * (mean_token_logp(best) - mean_token_logp(worst)))
L_PRISM = sum_i w_i * L_distill_i + alpha * L_aux
```

Defaults are `alpha=0.2`, `beta=2.5`, rollout group size 2, LoRA rank 32,
learning rate `3e-7`, 10 warmup steps, and 100 PRISM updates following 1,000 SFT
updates. The auxiliary loss is reference-free and length-normalized. The
historical config key is `dpo_aux_alpha`; this term does not use a DPO reference
policy. Equal-reward pairs contribute no auxiliary loss.

The EMA teacher has coefficient `0.999`. Teacher and student adapters share
the frozen backbone, and the teacher parameters are initialized from the
student. Gold/reference solutions are used only in teacher inputs during
training; generation evaluation uses the student prompt.

The HF rollout backend has a separate model copy and refreshes its adapter
every `train.n_sync` updates. It omits old-policy token log probabilities, so
the JSD importance weighting is inactive in that mode. The SGLang client can
provide token log probabilities; when available, detached probability ratios
are clamped to `[1-eps, 1+eps]` before weighting the loss.

## Baselines and ablations

SFT optimizes response-token cross entropy on gold JSON. DPO compares the best
and worst sampled responses using the base model with its LoRA adapter disabled
as the reference. The student can start from the SFT adapter, while this
reference remains the base model. SimPO uses sequence-length-normalized log
probabilities and a margin. The HF group-relative control centers and
standardizes group rewards and applies REINFORCE; it does not perform true
PPO clipping without stored old-policy log probabilities.

`sdpo_same_teacher.yaml` keeps the teacher inputs identical to PRISM and
disables reward weighting and the auxiliary term. It isolates those two loss
components, rather than removing diagnostic supervision. `no_dgr.yaml`
removes diagnostic text while retaining reference selection. `no_gold.yaml`
uses L3 instance feedback without a full reference JSON, and also disables
group-solution prompt selection through that feedback mode.

## Two evaluation quantities

The original training reward in `feedback/metrics.py` is
`min(macro_precision, macro_recall)` over nonempty node/edge type buckets.
Node matching uses a cost combining character overlap and bounding-box IoU.
This is retained to preserve the training algorithm.

The structural scorer in `feedback/structural.py` comes from the audited
structural evaluation:

1. Group nodes by exact type.
2. Compute Hungarian assignments using `1 - bbox_IoU` and retain pairs with
   IoU at least 0.5.
3. Match typed, directed edges through the retained node correspondences.
4. Pool node and edge TP, FP, and FN over all evaluation pages.
5. Compute `micro_F1 = 2*TP / (2*TP + FP + FN)`.

Evaluate each training seed separately and then compute the equal-weight mean
and standard deviation of those pooled scores. Averaging per-page F1 first is
a different statistic. Predicted text is not part of the structural IoU
matching criterion; it is used by the original training reward matcher.

The parser is permissive, consistent with the original scorer; the `parsed`
field indicates accepted graph JSON, not exhaustive schema validation. This
release does not evaluate z-order: the compact training schema omits it.

Use identical validation membership and gold targets across methods. The
portable configs separate `data.split_seed` from `train.seed`. Deterministic
generation defaults are 3,072 new tokens, temperature 0, repetition penalty
1.05, and no-repeat n-gram size 20. The evaluation command reports missing
pages and exits unsuccessfully if generation skipped any requested image.

## Implementation scope

The JSON parsing implementation and the PRISM objective are included. Later
graph-native models, routing experiments, private experiment logs, manuscript
drafts, and review materials are outside this release. The legacy `deepspeed`
config section controls gradient checkpointing only; this trainer does not
initialize a DeepSpeed engine. SFT/PRISM adapter continuation is supported;
saved adapters are not complete optimizer/RNG checkpoints for exact training
resumption.
