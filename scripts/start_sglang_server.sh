#!/usr/bin/env bash
# Start the customized SGLang server for Unlimited-OCR-2.
#
# Requirements: Linux, an NVIDIA GPU (24 GB+ VRAM recommended), Python 3.10
# via uv, and the customized wheel from the upstream repository.
#
# Usage:
#   ./scripts/start_sglang_server.sh
#
# Environment overrides:
#   MODEL_PATH=/path/to/model
#   HOST=0.0.0.0
#   PORT=10000
#   MEM_FRACTION=0.7
#   PAGE_WINDOW=10
#   VENV=.venv-uocr

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT_DIR/.venv-uocr}"
PY="$VENV/bin/python"
MODEL_PATH="${MODEL_PATH:-baidu/Unlimited-OCR-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10000}"
MEM_FRACTION="${MEM_FRACTION:-0.7}"
PAGE_WINDOW="${PAGE_WINDOW:-10}"
WHEEL="$ROOT_DIR/wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This server needs an NVIDIA GPU and driver." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY not found. Run the SGLang notebook first, or create the venv:" >&2
  echo "  uv venv --python 3.10 $VENV" >&2
  exit 1
fi

if [[ ! -f "$WHEEL" ]]; then
  echo "ERROR: customized SGLang wheel not found at $WHEEL" >&2
  echo "Download it from https://github.com/baidu/Unlimited-OCR/tree/main/wheel" >&2
  exit 1
fi

export SGLANG_UOCR_EANABLE_DOC_LIMIT="${SGLANG_UOCR_EANABLE_DOC_LIMIT:-1}"
export SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW="${SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW:-$PAGE_WINDOW}"
export SGLANG_UOCR_DOC_BASE_PAGE_WINDOW="${SGLANG_UOCR_DOC_BASE_PAGE_WINDOW:-$PAGE_WINDOW}"
export SGLANG_UOCR_MAX_PAGE_TOKENS="${SGLANG_UOCR_MAX_PAGE_TOKENS:-20000}"
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB="${SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB:-64}"
export SGLANG_UOCR_SHM_WRAP_MODEL_DATA="${SGLANG_UOCR_SHM_WRAP_MODEL_DATA:-1}"
export SGLANG_UOCR_SHM_WRAP_MIN_MB="${SGLANG_UOCR_SHM_WRAP_MIN_MB:-1}"
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Starting Unlimited-OCR-2 on http://$HOST:$PORT"
echo "Model: $MODEL_PATH"
echo "Logs will appear below. Press Ctrl+C to stop."

exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Unlimited-OCR-2 \
  --chat-template unlimited-ocr \
  --tokenizer-backend fastokens \
  --tokenizer-worker-num 8 \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests 64 \
  --max-prefill-tokens 2048 \
  --enable-custom-logit-processor \
  --enable-streaming-session \
  --allow-auto-truncate \
  --disable-radix-cache \
  --mm-preprocess-cache-size-mb 0 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 64 \
  --cuda-graph-backend-prefill full \
  --cuda-graph-max-bs-prefill 16384 \
  --cuda-graph-bs-prefill 1024 2048 \
  --cuda-graph-config '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}' \
  --host "$HOST" \
  --port "$PORT"
