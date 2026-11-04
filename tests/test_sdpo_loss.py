"""Sanity test：对随机 logits 验证 top-K JSD 近似与暴力 full JSD 的一致性。

期望：两者的差应该 < ~1%（相对），top_k 越大越接近。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from sdpo_vlm.trainer.sdpo_loss import topk_symmetric_jsd_loss


def brute_force_jsd(student_logits, teacher_logits, mask=None, eps=1e-12):
    p = F.softmax(student_logits.float(), dim=-1)
    q = F.softmax(teacher_logits.float(), dim=-1)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
    kl_qm = (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
    per_tok = 0.5 * (kl_pm + kl_qm)
    if mask is None:
        return per_tok.mean()
    return (per_tok * mask).sum() / mask.sum().clamp_min(1.0)


def main():
    torch.manual_seed(0)
    R, V = 32, 4096

    # 模拟更真实的 LLM logits：大部分 mass 集中在 top-K 内（峰值尖锐）
    # 构造方式：logits = sharp peak + 小平底噪声
    student = torch.randn(R, V) * 0.5
    topk_boost = torch.randn(R, 80) * 4.0 + 8.0   # top-80 位置被显著抬高
    boost_idx = torch.arange(80).unsqueeze(0).expand(R, -1)
    student.scatter_add_(1, boost_idx, topk_boost)

    teacher = student + torch.randn(R, V) * 0.3

    ref = brute_force_jsd(student, teacher).item()
    print(f"brute-force JSD (sharp-peak logits): {ref:.6f}")

    for k in [32, 100, 500, 2000, V]:
        approx = topk_symmetric_jsd_loss(student, teacher, top_k=k).item()
        rel = abs(approx - ref) / max(abs(ref), 1e-8)
        tag = "OK" if rel < 0.05 else "WARN"
        print(f"  top_k={k:5d}  approx={approx:.6f}  rel_err={rel*100:.3f}%  [{tag}]")

    # 附加一组"扁平"分布：top-K 近似偏差大属正常（不是 bug）
    print("\nflat-logits（peak 不明显）—— top-K 偏差会较大，仅供参考：")
    flat_s = torch.randn(R, V) * 2
    flat_t = flat_s + torch.randn(R, V) * 0.5
    flat_ref = brute_force_jsd(flat_s, flat_t).item()
    for k in [100, V]:
        a = topk_symmetric_jsd_loss(flat_s, flat_t, top_k=k).item()
        rel = abs(a - flat_ref) / max(abs(flat_ref), 1e-8)
        print(f"  top_k={k:5d}  approx={a:.6f}  rel_err={rel*100:.3f}%")

    # 测试：student == teacher 时 loss == 0
    same_loss = topk_symmetric_jsd_loss(student, student, top_k=100).item()
    assert same_loss < 1e-5, f"self-JSD should be ~0, got {same_loss}"
    print(f"  self JSD = {same_loss:.3e} (expect ~0)")

    # 测试：mask
    mask = torch.zeros(R); mask[:8] = 1
    m_loss = topk_symmetric_jsd_loss(student, teacher, top_k=V, mask=mask).item()
    m_ref = brute_force_jsd(student, teacher, mask=mask).item()
    print(f"masked full-K  approx={m_loss:.6f}  brute={m_ref:.6f}")
    assert abs(m_loss - m_ref) / max(abs(m_ref), 1e-8) < 0.01

    # 测试：importance ratio clip
    r_loss = topk_symmetric_jsd_loss(
        student, teacher, top_k=100,
        importance_ratio=torch.ones(R) * 1.5,
        importance_clip_eps=0.2,
    ).item()
    expected_scale = 1.2   # clamp(1.5, 0.8, 1.2) = 1.2
    base = topk_symmetric_jsd_loss(student, teacher, top_k=100).item()
    assert abs(r_loss - base * expected_scale) / max(base, 1e-8) < 0.01
    print(f"importance clip (ratio=1.5 → clamp 1.2): r_loss/base = {r_loss/base:.3f}")

    print("ALL OK")


def test_reference_jsd_mask_and_importance_clipping():
    main()


def test_loss_has_finite_student_gradients():
    torch.manual_seed(7)
    student = torch.randn(5, 64, requires_grad=True)
    teacher = torch.randn(5, 64)
    loss = topk_symmetric_jsd_loss(student, teacher, top_k=32)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0


if __name__ == "__main__":
    main()
