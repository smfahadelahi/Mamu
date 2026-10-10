#!/usr/bin/env bash
# End-to-end pilot run: tokenizer -> shards -> train -> HF export -> GGUF Q4_K_M.
#
# Required:
#   CORPUS_GLOB      glob of plain-text files, documents separated by blank lines
#   LLAMA_CPP_DIR    llama.cpp checkout (for convert_hf_to_gguf.py and gguf-py)
# Optional:
#   RUN_DIR (runs/pilot), PRESET (pilot-300m), STEPS (20000), BATCH (8), ACCUM (4),
#   VOCAB (32000), QAT_FRAC (0.9), TEACHER_HF (unset = no distillation)
#
# Pilot budget: set STEPS for your token target. 32,768 tokens per step at the defaults, so
# 20,000 steps is about 0.66B tokens. Chinchilla-optimal for 300M params is about 6B tokens.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CORPUS_GLOB:?set CORPUS_GLOB to your text files}"
: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a llama.cpp checkout}"
RUN_DIR="${RUN_DIR:-runs/pilot}"
PRESET="${PRESET:-pilot-300m}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-4}"
VOCAB="${VOCAB:-32000}"
QAT_FRAC="${QAT_FRAC:-0.9}"

mkdir -p "$RUN_DIR"
TOK="$RUN_DIR/tokenizer"
SHARDS="$RUN_DIR/shards"

if [ ! -f "$TOK/tokenizer.model" ]; then
  python -m data.prepare train-tokenizer --files "$CORPUS_GLOB" --out "$TOK" --vocab-size "$VOCAB"
fi
if [ ! -d "$SHARDS" ] || [ -z "$(ls -A "$SHARDS" 2>/dev/null)" ]; then
  python -m data.prepare tokenize --files "$CORPUS_GLOB" --tokenizer "$TOK/tokenizer.model" --out "$SHARDS"
fi

EXTRA=()
if [ -n "${TEACHER_HF:-}" ]; then EXTRA+=(--teacher-hf "$TEACHER_HF"); fi

# Resumes automatically from the newest checkpoint if one exists.
LAST=$(ls -1 "$RUN_DIR"/ckpt_*.pt 2>/dev/null | sort | tail -n 1 || true)
RESUME=()
if [ -n "$LAST" ]; then RESUME=(--resume "$LAST"); fi

python -m train.train --shards "$SHARDS" --out "$RUN_DIR" --preset "$PRESET" \
  --tokenizer "$TOK/tokenizer.model" --steps "$STEPS" --batch-size "$BATCH" \
  --grad-accum "$ACCUM" --qat-start-frac "$QAT_FRAC" "${EXTRA[@]}" "${RESUME[@]}"

FINAL=$(ls -1 "$RUN_DIR"/ckpt_*.pt | sort | tail -n 1)
python -m export.to_hf --ckpt "$FINAL" --out "$RUN_DIR/hf" --tokenizer "$TOK/tokenizer.model"
LLAMA_CPP_DIR="$LLAMA_CPP_DIR" export/gguf.sh "$RUN_DIR/hf" "$RUN_DIR/gguf"
echo "done: $RUN_DIR/gguf/model-q4_k_m.gguf"
