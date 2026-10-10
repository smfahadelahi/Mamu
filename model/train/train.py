"""Pretraining / distillation / QAT loop for MobileLM.

Examples:
  # Pretrain from scratch on prepared shards
  python -m train.train --shards shards/ --out runs/base --steps 100000

  # Add distillation from a teacher that shares the student's tokenizer
  python -m train.train --shards shards/ --out runs/kd --teacher-hf path/to/teacher

  # Final 10% of steps with fake-INT4 weights (QAT)
  python -m train.train --shards shards/ --out runs/qat --qat-start-frac 0.9
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from data.dataset import TokenShards
from mobile_lm import MobileLM, MobileLMConfig, count_parameters
from mobile_lm.losses import ce_loss, distill_loss
from mobile_lm.qat import enable_qat

TeacherFn = Callable[[torch.Tensor], torch.Tensor]


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        # Norm weights and biases are excluded from weight decay.
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def save_checkpoint(model: MobileLM, optimizer, step: int, out_dir: str, qat: bool) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{step:07d}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": model.cfg.to_dict(),
            # Export must reproduce the INT4 grid the model was trained on.
            "qat": qat,
        },
        path,
    )
    return path


def train(
    cfg: MobileLMConfig,
    shard_dir: str,
    out_dir: str,
    steps: int,
    batch_size: int = 8,
    grad_accum: int = 1,
    lr: float = 3e-4,
    warmup: int = 200,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    kd_alpha: float = 0.5,
    kd_temperature: float = 2.0,
    qat_start_frac: float | None = None,
    teacher: TeacherFn | None = None,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 50,
    ckpt_every: int = 1000,
    resume: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    model = MobileLM(cfg).to(device)
    optimizer = build_optimizer(model, lr, weight_decay)
    start_step = 0
    qat_converted = False
    if resume:
        state = torch.load(resume, map_location=device)
        if state.get("qat"):
            enable_qat(model)  # same state_dict keys, so this only changes the forward pass
            qat_converted = True
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print(f"resumed from {resume} at step {start_step}")

    print(f"parameters: {count_parameters(model):,}")
    data = TokenShards(shard_dir, "train", cfg.max_seq_len, samples=steps * batch_size * grad_accum, seed=seed)
    loader = DataLoader(data, batch_size=batch_size, num_workers=0)
    it = iter(loader)
    # Skip batches already consumed before a resume so data order stays the same.
    for _ in range(start_step * grad_accum):
        next(it)

    qat_step = None if qat_start_frac is None else int(steps * qat_start_frac)
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    model.train()
    history: list[float] = []
    t0 = time.time()

    for step in range(start_step, steps):
        if qat_step is not None and step >= qat_step and not qat_converted:
            n = enable_qat(model)
            qat_converted = True
            print(f"step {step}: QAT enabled on {n} linear layers")

        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, lr, warmup, steps)

        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            batch = next(it).to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(inputs).logits
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(inputs)
                    loss = distill_loss(logits, targets, t_logits,
                                        alpha=kd_alpha, temperature=kd_temperature)
                else:
                    loss = ce_loss(logits, targets)
            (loss / grad_accum).backward()
            loss_acc += loss.item() / grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        history.append(loss_acc)

        if (step + 1) % log_every == 0:
            avg = sum(history[-log_every:]) / log_every
            rate = (step + 1 - start_step) / (time.time() - t0)
            print(f"step {step + 1}/{steps} loss {avg:.4f} lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"({rate:.2f} steps/s)")
        if (step + 1) % ckpt_every == 0:
            print("saved", save_checkpoint(model, optimizer, step + 1, out_dir, qat_converted))

    final = save_checkpoint(model, optimizer, steps, out_dir, qat_converted)
    print("saved", final)
    return {"final_checkpoint": final, "final_loss": history[-1] if history else None}


def load_hf_teacher(path: str, device: str) -> TeacherFn:
    """Wrap a Hugging Face causal LM. It must share the student's token IDs."""
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()

    def run(input_ids: torch.Tensor) -> torch.Tensor:
        return teacher(input_ids=input_ids).logits.float()

    return run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config-json", help="JSON file with MobileLMConfig fields (default: 1.5B preset)")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--kd-alpha", type=float, default=0.5, help="weight on hard CE when --teacher-hf is set (1.0 = pure CE)")
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--teacher-hf", help="path to a teacher with the same tokenizer")
    p.add_argument("--qat-start-frac", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume")
    args = p.parse_args()

    cfg = MobileLMConfig(**json.load(open(args.config_json))) if args.config_json else MobileLMConfig()
    teacher = load_hf_teacher(args.teacher_hf, args.device) if args.teacher_hf else None
    train(
        cfg=cfg,
        shard_dir=args.shards,
        out_dir=args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup=args.warmup,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        qat_start_frac=args.qat_start_frac,
        teacher=teacher,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
