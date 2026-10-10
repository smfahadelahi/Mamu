"""Training losses: next-token cross-entropy and logit distillation (KL)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """logits: (B, T, V); targets: (B, T)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Forward KL(teacher || student) with temperature scaling, scaled by T^2.

    If top_k is set, only the teacher's top-k tokens per position are used
    (renormalised). This is far cheaper to store and transfer for a 32k vocab
    than full distributions.
    """
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    if top_k is not None:
        t_vals, idx = t.topk(top_k, dim=-1)
        t_logp = F.log_softmax(t_vals, dim=-1)
        s_logp = F.log_softmax(s, dim=-1).gather(-1, idx)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    else:
        t_logp = F.log_softmax(t, dim=-1)
        s_logp = F.log_softmax(s, dim=-1)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    return kl.mean() * (temperature ** 2)


def distill_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """alpha * CE(hard labels) + (1 - alpha) * KD(teacher soft targets)."""
    hard = ce_loss(student_logits, targets)
    soft = kd_loss(student_logits, teacher_logits, temperature, top_k)
    return alpha * hard + (1.0 - alpha) * soft
