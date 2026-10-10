# mobile_lm: on-device model pipeline

A ~1.5B decoder-only LLaMA-compatible model (GQA 16:4, RoPE, RMSNorm, SwiGLU, 32k
SentencePiece vocab, tied embeddings), with the full path from raw text to a Q4_K_M
GGUF file for llama.cpp.

```
corpus/*.txt ──► data.prepare train-tokenizer ──► tokenizer.model
corpus/*.txt ──► data.prepare tokenize ─────────► shards/*.bin
shards/      ──► train.train (CE, optional KD, QAT) ──► ckpt_*.pt
ckpt_*.pt    ──► export.to_hf ──► hf/ (LLaMA format)
hf/          ──► export/gguf.sh ──► model-q4_k_m.gguf
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

## Steps

```bash
# 1. Tokenizer (32k SentencePiece BPE with byte fallback)
python -m data.prepare train-tokenizer --files 'corpus/*.txt' --out tokenizer/ --vocab-size 32000

# 2. Shards (quality filter, exact-dedup, deterministic train/val split)
python -m data.prepare tokenize --files 'corpus/*.txt' --tokenizer tokenizer/tokenizer.model --out shards/

# 3. Train. Keep the config's vocab_size equal to the tokenizer's size.
python -m train.train --shards shards/ --out runs/base --steps 100000 --device cuda

# 4. Optional distillation (teacher must share the student's token IDs)
python -m train.train --shards shards/ --out runs/kd --teacher-hf path/to/teacher --kd-alpha 0.5

# 5. QAT on the last 10% of steps. Resume a run from a checkpoint with --resume.
python -m train.train --shards shards/ --out runs/qat --qat-start-frac 0.9 --resume runs/base/ckpt_0090000.pt

# 6. Export to LLaMA format, then GGUF Q4_K_M
python -m export.to_hf --ckpt runs/qat/ckpt_0100000.pt --out export/hf --tokenizer tokenizer/tokenizer.model
LLAMA_CPP_DIR=/path/to/llama.cpp export/gguf.sh export/hf export/gguf

# 7. Check llama.cpp against PyTorch on the same token IDs.
#    Build the probe against a built llama.cpp (static libs in its build dir):
#    g++ -std=c++17 -O1 -I$LLAMA_CPP_DIR/include -I$LLAMA_CPP_DIR/ggml/include \
#        tools/llama_logits_probe.cpp -o llama_logits_probe -Wl,--start-group <libllama.a, libggml*.a> -Wl,--end-group -lpthread -ldl -lm
python tools/compare_logits.py ./llama_logits_probe runs/qat/ckpt_0100000.pt export/gguf/model-f16.gguf
```

## What is verified (CPU tests and a tiny end-to-end run)

- The model's logits match Hugging Face `LlamaForCausalLM` on the same weights.
- Incremental KV-cache decoding matches full-sequence forward.
- A QAT export reproduces the INT4 grid used in training.
- A tiny model (vocab 300) went through train → export → llama.cpp `convert_hf_to_gguf.py`
  → `llama-quantize Q4_K_M`. llama.cpp's f16 logits matched PyTorch to a max abs diff
  of 0.003, with 100% argmax agreement.

## Not verified here

- Real-scale training. This sandbox has no GPU, so the 1.5B run and its throughput are untested.
- Distillation from a real teacher. Teacher logits must share the student's token IDs.
  Teachers with a different tokenizer (for example Qwen or Llama 3) need precomputed
  top-k logits aligned to the student's tokens. That is not implemented.
- Quality of the exported model. Q4_K_M is applied by llama.cpp, so QAT's
  symmetric INT4 grid only approximates the final format.
