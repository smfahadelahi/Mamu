import json
import os

import numpy as np
import pytest
import torch

from data.dataset import TokenShards
from data.prepare import tokenize_to_shards, train_tokenizer
from export.to_hf import to_llama_state_dict, llama_config
from mobile_lm import MobileLM, MobileLMConfig
from mobile_lm.losses import ce_loss, distill_loss, kd_loss
from mobile_lm.qat import enable_qat, fake_quant_int4
from train.train import lr_at, train


def tiny_cfg(**o):
    base = dict(vocab_size=257, d_model=64, n_layers=2, n_heads=8, n_kv_heads=2,
                ffn_hidden=192, max_seq_len=64)
    base.update(o)
    return MobileLMConfig(**base)


# ---- architecture equivalence with Hugging Face LLaMA -----------------------

def test_matches_hf_llama_logits():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = tiny_cfg()
    ours = MobileLM(cfg).eval()
    hf = LlamaForCausalLM(LlamaConfig(**llama_config(cfg))).eval()
    missing, unexpected = hf.load_state_dict(to_llama_state_dict(ours, torch.float32), strict=False)
    assert not unexpected
    assert all("lm_head" in k for k in missing) or not missing  # only tied head may be absent

    ids = torch.randint(0, cfg.vocab_size, (2, 17))
    with torch.no_grad():
        a = ours(ids).logits
        b = hf(input_ids=ids).logits
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_export_state_dict_names_and_shapes():
    cfg = tiny_cfg()
    sd = to_llama_state_dict(MobileLM(cfg), torch.float16)
    assert sd["model.layers.0.self_attn.k_proj.weight"].shape == (cfg.n_kv_heads * cfg.head_dim, cfg.d_model)
    assert "lm_head.weight" not in sd  # tied
    assert all(v.dtype == torch.float16 for v in sd.values())


# ---- losses -----------------------------------------------------------------

def test_kd_zero_for_identical_logits():
    x = torch.randn(2, 5, 31)
    assert kd_loss(x, x).abs().item() < 1e-6


def test_topk_equals_full_when_k_is_vocab():
    s, t = torch.randn(2, 5, 31), torch.randn(2, 5, 31)
    full = kd_loss(s, t, temperature=1.0)
    topk = kd_loss(s, t, temperature=1.0, top_k=31)
    torch.testing.assert_close(full, topk, rtol=1e-5, atol=1e-5)


def test_distill_loss_mixes_ce_and_kd():
    s, t = torch.randn(2, 5, 31), torch.randn(2, 5, 31)
    y = torch.randint(0, 31, (2, 5))
    mix = distill_loss(s, y, t, alpha=0.3, temperature=2.0)
    expected = 0.3 * ce_loss(s, y) + 0.7 * kd_loss(s, t, 2.0)
    torch.testing.assert_close(mix, expected)


# ---- QAT --------------------------------------------------------------------

def test_fake_quant_values_on_int4_grid():
    torch.manual_seed(0)
    w = torch.randn(4, 64)
    q = fake_quant_int4(w, group_size=32).detach()
    g = q.reshape(4, 2, 32)
    scale = w.reshape(4, 2, 32).abs().amax(-1, keepdim=True) / 7.0
    levels = g / scale
    assert torch.allclose(levels, levels.round(), atol=1e-4)
    assert levels.min() >= -8 - 1e-4 and levels.max() <= 7 + 1e-4


def test_fake_quant_straight_through_gradient():
    w = torch.randn(4, 32, requires_grad=True)
    fake_quant_int4(w).sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_enable_qat_converts_block_linears_only():
    model = MobileLM(tiny_cfg())
    n = enable_qat(model)
    assert n == 2 * 7  # 7 linears per block
    assert type(model.lm_head) is torch.nn.Linear
    assert type(model.blocks[0].mlp.gate).__name__ == "FakeQuantLinear"
    out = model(torch.randint(0, 257, (1, 8))).logits
    assert torch.isfinite(out).all()


# ---- data -------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    docs = []
    for i in range(60):
        docs.append(f"Document number {i}. " + ("The quick brown fox jumps over the lazy dog. " * 6))
    docs.append("Document number 0. " + ("The quick brown fox jumps over the lazy dog. " * 6))  # duplicate
    docs.append("too short")
    path = tmp_path / "corpus.txt"
    path.write_text("\n\n".join(docs), encoding="utf-8")
    return str(path)


def test_tokenizer_and_shards_roundtrip(tmp_path, corpus):
    tok_dir = str(tmp_path / "tok")
    tok_path = train_tokenizer([corpus], tok_dir, vocab_size=300)
    shards = str(tmp_path / "shards")
    stats = tokenize_to_shards([corpus], tok_path, shards, shard_tokens=2000, val_fraction=0.2)
    assert stats["dropped_dupes"] == 1
    assert stats["dropped_quality"] == 1
    assert stats["docs"] == 60
    assert any(f.startswith("train_") for f in os.listdir(shards))

    ds = TokenShards(shards, "train", seq_len=32, samples=4)
    item = ds[0]
    assert item.shape == (33,)
    assert int(item.max()) < 300
    import sentencepiece as spm
    assert spm.SentencePieceProcessor(model_file=tok_path).id_to_piece(2) == "</s>"


def test_dataset_windows_are_deterministic(tmp_path):
    arr = np.arange(5000, dtype=np.uint16) % 200
    arr.tofile(tmp_path / "train_00000.bin")
    a = TokenShards(str(tmp_path), "train", seq_len=16, samples=3, seed=1)
    b = TokenShards(str(tmp_path), "train", seq_len=16, samples=3, seed=1)
    for i in range(3):
        assert torch.equal(a[i], b[i])


# ---- training loop smoke test ----------------------------------------------

def test_train_runs_with_qat_and_resumes(tmp_path):
    arr = (np.arange(20000, dtype=np.uint32) * 7 % 257).astype(np.uint16)
    shards = tmp_path / "shards"
    shards.mkdir()
    arr.tofile(shards / "train_00000.bin")
    cfg = tiny_cfg(max_seq_len=32)
    out = str(tmp_path / "run")

    r1 = train(cfg, str(shards), out, steps=6, batch_size=2, grad_accum=2, lr=1e-3, warmup=2,
               qat_start_frac=0.5, log_every=2, ckpt_every=3)
    assert os.path.exists(r1["final_checkpoint"])
    state = torch.load(r1["final_checkpoint"], map_location="cpu")
    assert state["step"] == 6

    r2 = train(cfg, str(shards), out, steps=8, batch_size=2, grad_accum=2, lr=1e-3, warmup=2,
               log_every=2, ckpt_every=100, resume=r1["final_checkpoint"])
    assert torch.load(r2["final_checkpoint"], map_location="cpu")["step"] == 8


def test_lr_schedule_shape():
    assert lr_at(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(99, 1.0, 10, 100) == pytest.approx(0.1, abs=1e-3)


def test_qat_export_matches_qat_forward():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = tiny_cfg()
    ours = MobileLM(cfg)
    enable_qat(ours)
    ours.eval()  # QAT forward uses INT4-rounded block weights
    hf = LlamaForCausalLM(LlamaConfig(**llama_config(cfg))).eval()
    hf.load_state_dict(to_llama_state_dict(ours, torch.float32, qat=True), strict=False)
    ids = torch.randint(0, cfg.vocab_size, (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(ours(ids).logits, hf(input_ids=ids).logits, rtol=1e-4, atol=1e-4)


def test_validation_loss_logged(tmp_path):
    import json
    arr = (np.arange(20000, dtype=np.uint32) * 7 % 257).astype(np.uint16)
    shards = tmp_path / "shards"
    shards.mkdir()
    arr.tofile(shards / "train_00000.bin")
    arr[:5000].tofile(shards / "val_00000.bin")
    out = str(tmp_path / "run")
    train(tiny_cfg(max_seq_len=32), str(shards), out, steps=4, batch_size=2, grad_accum=1,
          lr=1e-3, warmup=1, log_every=2, eval_every=2, eval_batches=2)
    records = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    assert [r["step"] for r in records] == [2, 4]
    assert "val_loss" in records[0] and "val_loss" in records[1]
    assert np.isfinite(records[-1]["val_loss"])


def test_presets_sizes():
    from mobile_lm.presets import get_preset
    pilot = get_preset("pilot-300m")
    assert 280e6 < pilot.param_count() < 330e6
    assert pilot.ffn_hidden % 32 == 0 and pilot.d_model % 32 == 0
    assert 1.4e9 < get_preset("mobile-1.5b").param_count() < 1.6e9
    with pytest.raises(KeyError):
        get_preset("nope")
