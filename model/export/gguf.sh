#!/usr/bin/env bash
# Convert an exported Hugging Face directory to GGUF and quantise to Q4_K_M.
# Usage: LLAMA_CPP_DIR=/path/to/llama.cpp LLAMA_QUANTIZE=/path/to/llama-quantize \
#        export/gguf.sh export/hf out/
set -euo pipefail
HF_DIR="$1"; OUT="$2"
: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a llama.cpp checkout}"
QUANT="${LLAMA_QUANTIZE:-llama-quantize}"
mkdir -p "$OUT"
PYTHONPATH="$LLAMA_CPP_DIR/gguf-py" python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$HF_DIR" --outfile "$OUT/model-f16.gguf" --outtype f16
"$QUANT" "$OUT/model-f16.gguf" "$OUT/model-q4_k_m.gguf" Q4_K_M
echo "wrote $OUT/model-q4_k_m.gguf"
