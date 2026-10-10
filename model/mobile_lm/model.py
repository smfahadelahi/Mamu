"""Decoder-only transformer: RMSNorm, RoPE, GQA attention, SwiGLU MLP.

The forward pass supports an incremental KV cache so that the same code path
used for single-token decoding on device can be checked against full-sequence
evaluation during training and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MobileLMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No mean subtraction, unlike LayerNorm.
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def build_rope_cache(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE (Hugging Face LLaMA convention).

    Pairs dimension i with i + head_dim/2, so exported weights load directly into
    LlamaForCausalLM and llama.cpp's "llama" architecture without permutation.
    x: (batch, heads, seq, head_dim); cos/sin: (seq, head_dim/2).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: MobileLMConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        new_past = (k, v)

        # Expand KV heads to query heads for attention (GQA). The cache itself
        # stores only n_kv_heads, which is what saves memory on device.
        rep = self.n_heads // self.n_kv_heads
        k_full = k.repeat_interleave(rep, dim=1)
        v_full = v.repeat_interleave(rep, dim=1)

        # Causal mask is only needed when attending over more than one new
        # token with no cache; single-token decoding sees all past positions.
        is_causal = past is None and t > 1
        out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.wo(out), new_past


class SwiGLU(nn.Module):
    def __init__(self, cfg: MobileLMConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: MobileLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, past):
        h, new_past = self.attn(self.attn_norm(x), cos, sin, past)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_past


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    past: list[tuple[torch.Tensor, torch.Tensor]]


class MobileLM(nn.Module):
    def __init__(self, cfg: MobileLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> ForwardOutput:
        b, t = input_ids.shape
        start = 0 if past is None else past[0][0].shape[2]
        end = start + t
        if end > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {end} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos = self.rope_cos[start:end]
        sin = self.rope_sin[start:end]

        x = self.embed(input_ids)
        new_past = []
        for i, block in enumerate(self.blocks):
            layer_past = None if past is None else past[i]
            x, lp = block(x, cos, sin, layer_past)
            new_past.append(lp)
        x = self.final_norm(x)
        return ForwardOutput(logits=self.lm_head(x), past=new_past)


def count_parameters(model: nn.Module) -> int:
    # Tied weights are yielded once by parameters(), so this is the true count.
    return sum(p.numel() for p in model.parameters())
