"""Named model configs.

- pilot-300m: ~316M parameters for a first run on one consumer GPU (24 GB).
  Used to validate data, training, and the phone app before the full run.
- mobile-1.5b: the ~1.49B on-device target (32 layers, d_model 2048).
"""

from .config import MobileLMConfig

PRESETS: dict[str, MobileLMConfig] = {
    "pilot-300m": MobileLMConfig(
        d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4,
        ffn_hidden=2816, max_seq_len=1024,
    ),
    "mobile-1.5b": MobileLMConfig(),
}


def get_preset(name: str) -> MobileLMConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
