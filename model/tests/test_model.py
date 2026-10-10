import pytest
import torch

from mobile_lm import MobileLM, MobileLMConfig, count_parameters
from mobile_lm.config import MobileLMConfig as Cfg


def tiny_cfg(**overrides) -> MobileLMConfig:
    base = dict(
        vocab_size=257, d_model=64, n_layers=2, n_heads=8, n_kv_heads=2,
        ffn_hidden=176, max_seq_len=64,
    )
    base.update(overrides)
    return MobileLMConfig(**base)


def test_default_config_hits_mobile_budget():
    cfg = MobileLMConfig()
    params = cfg.param_count()
    assert 1.4e9 < params < 1.6e9, params
    # 4-bit weights should fit the 1.0-2.5 GB budget with room for the KV cache.
    assert cfg.weight_bytes_quantized(4.5) < 1.0e9
    # fp16 KV cache at 2k context should be about 134 MB, well under 250 MB.
    assert cfg.kv_cache_bytes(2048) == 2 * 32 * 4 * 128 * 2 * 2048
    assert cfg.kv_cache_bytes(2048) < 250 * 2**20


def test_default_config_matches_real_module_param_count():
    # Meta-device build: exact shapes without allocating ~6 GB of fp32 weights.
    cfg = MobileLMConfig()
    with torch.device("meta"):
        model = MobileLM(cfg)
    assert count_parameters(model) == cfg.param_count()


def test_param_count_matches_module_untied():
    cfg = tiny_cfg(tie_embeddings=False)
    assert count_parameters(MobileLM(cfg)) == cfg.param_count()


def test_param_count_matches_module_tied():
    cfg = tiny_cfg(tie_embeddings=True)
    assert count_parameters(MobileLM(cfg)) == cfg.param_count()


def test_gqa_shrinks_kv_cache():
    mha = tiny_cfg(n_kv_heads=8)
    gqa = tiny_cfg(n_kv_heads=2)
    assert gqa.kv_cache_bytes(64) * 4 == mha.kv_cache_bytes(64)


@pytest.mark.parametrize("bad", [
    dict(n_heads=6, n_kv_heads=4),     # heads not divisible by kv heads
    dict(d_model=60, n_heads=8),       # d_model not divisible by heads
    dict(d_model=24, n_heads=8),       # head_dim 3 is odd
])
def test_invalid_configs_rejected(bad):
    with pytest.raises(ValueError):
        tiny_cfg(**bad)


def test_incremental_decoding_matches_full_forward():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MobileLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(ids).logits

        # Prefill 8 tokens, then decode the rest one at a time with the cache.
        out = model(ids[:, :8])
        steps = [out.logits]
        past = out.past
        for i in range(8, 12):
            out = model(ids[:, i:i + 1], past=past)
            steps.append(out.logits)
            past = out.past
        incremental = torch.cat(steps, dim=1)

    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-4)


def test_kv_cache_stores_only_kv_heads():
    cfg = tiny_cfg()
    model = MobileLM(cfg).eval()
    with torch.no_grad():
        out = model(torch.randint(0, cfg.vocab_size, (1, 5)))
    k, v = out.past[0]
    assert k.shape == (1, cfg.n_kv_heads, 5, cfg.head_dim)
    assert v.shape == k.shape


def test_context_limit_enforced():
    cfg = tiny_cfg(max_seq_len=16)
    model = MobileLM(cfg).eval()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_tied_embeddings_share_storage():
    model = MobileLM(tiny_cfg(tie_embeddings=True))
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
