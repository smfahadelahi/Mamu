"""Export a MobileLM checkpoint to a Hugging Face LLaMA-format directory.

The architecture is exactly LLaMA (GQA, rotate-half RoPE, RMSNorm, SwiGLU, tied
embeddings), so the output loads with transformers' LlamaForCausalLM and converts
to GGUF with llama.cpp's convert_hf_to_gguf.py (architecture "llama").

Usage:
  python -m export.to_hf --ckpt runs/qat/ckpt_0100000.pt --out export/hf --tokenizer tokenizer/tokenizer.model
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from mobile_lm import MobileLM, MobileLMConfig
from mobile_lm.qat import fake_quant_int4

# Our parameter name -> LLaMA parameter name (per block, index filled in later).
BLOCK_MAP = {
    "attn.wq.weight": "self_attn.q_proj.weight",
    "attn.wk.weight": "self_attn.k_proj.weight",
    "attn.wv.weight": "self_attn.v_proj.weight",
    "attn.wo.weight": "self_attn.o_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "mlp.gate.weight": "mlp.gate_proj.weight",
    "mlp.up.weight": "mlp.up_proj.weight",
    "mlp.down.weight": "mlp.down_proj.weight",
}


def to_llama_state_dict(model: MobileLM, dtype: torch.dtype = torch.float16,
                        qat: bool = False) -> dict[str, torch.Tensor]:
    sd = dict(model.state_dict())
    if qat:
        # Apply the same INT4 grid used during QAT to the block linears (not embeddings).
        for k, v in sd.items():
            if k.startswith("blocks.") and k.endswith(".weight") and "norm" not in k:
                sd[k] = fake_quant_int4(v.detach(), 32).detach()
    out: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": sd["embed.weight"],
        "model.norm.weight": sd["final_norm.weight"],
    }
    for i in range(model.cfg.n_layers):
        for ours, theirs in BLOCK_MAP.items():
            out[f"model.layers.{i}.{theirs}"] = sd[f"blocks.{i}.{ours}"]
    # lm_head is tied to embed_tokens in LLaMA-format configs, so it is omitted.
    if not model.cfg.tie_embeddings:
        out["lm_head.weight"] = sd["lm_head.weight"]
    return {k: v.detach().to(dtype).contiguous().cpu() for k, v in out.items()}


def llama_config(cfg: MobileLMConfig, tokenizer_dir: str | None = None) -> dict:
    conf = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.ffn_hidden,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.rms_eps,
        "rope_theta": cfg.rope_theta,
        "rope_scaling": None,
        "tie_word_embeddings": cfg.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "initializer_range": 0.02,
        "use_cache": True,
        "torch_dtype": "float16",
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    return conf


def export(ckpt_path: str, out_dir: str, tokenizer_path: str | None = None) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = MobileLMConfig(**state["config"])
    model = MobileLM(cfg)
    model.load_state_dict(state["model"])
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    save_file(to_llama_state_dict(model, qat=bool(state.get("qat"))), os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(llama_config(cfg), f, indent=2)
    if tokenizer_path:
        shutil.copy(tokenizer_path, os.path.join(out_dir, "tokenizer.model"))
    print(f"exported {ckpt_path} -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer")
    args = p.parse_args()
    export(args.ckpt, args.out, args.tokenizer)


if __name__ == "__main__":
    main()
