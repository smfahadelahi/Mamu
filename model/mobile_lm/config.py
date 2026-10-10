"""Architecture configuration for the on-device model.

Defaults target ~1.5B parameters with a 32k vocabulary, GQA (16 query / 4 KV
heads), SwiGLU, RMSNorm and RoPE, following the mobile budget in the design
doc. Use `param_count()` and `kv_cache_bytes()` to check any change against
the RAM budget before training.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MobileLMConfig:
    vocab_size: int = 32_000
    d_model: int = 2048
    # 32 layers (not 24): with d_model=2048 and a 32k vocab, 24 layers gives
    # only ~1.13B parameters. 32 layers with tied embeddings gives ~1.49B.
    n_layers: int = 32
    n_heads: int = 16
    n_kv_heads: int = 4          # GQA: 4x smaller KV cache than MHA
    ffn_hidden: int = 5504       # ~8/3 * d_model, SwiGLU (3 matrices)
    max_seq_len: int = 2048      # mobile context budget
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def param_count(self) -> int:
        """Exact count of trainable parameters (RMSNorm weights included)."""
        head_dim = self.head_dim
        attn = (
            self.d_model * self.n_heads * head_dim            # Wq
            + 2 * self.d_model * self.n_kv_heads * head_dim   # Wk, Wv
            + self.n_heads * head_dim * self.d_model          # Wo
        )
        mlp = 3 * self.d_model * self.ffn_hidden              # gate, up, down
        norms = 2 * self.d_model                              # per block
        per_layer = attn + mlp + norms
        embed = self.vocab_size * self.d_model
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return self.n_layers * per_layer + embed + head + self.d_model  # final norm

    def kv_cache_bytes(self, seq_len: int | None = None, bytes_per_elem: int = 2) -> int:
        """KV cache size for one sequence. Default is fp16 (2 bytes per element)."""
        seq_len = self.max_seq_len if seq_len is None else seq_len
        per_token = 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_elem
        return per_token * seq_len

    def weight_bytes_quantized(self, bits_per_weight: float = 4.5) -> int:
        """Approximate size of GGUF Q4_K_M-style weights (~4.5 bits/weight average)."""
        return int(self.param_count() * bits_per_weight / 8)

    def to_dict(self) -> dict:
        return asdict(self)
