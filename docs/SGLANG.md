# SGLang server guide

SGLang is the recommended backend for long documents, batch jobs and concurrent
requests. The upstream model authors ship a **customized wheel** that includes
the Unlimited-OCR-2 chat template, the page-window scheduler and the
`DeepseekOCRNoRepeatNGramLogitProcessor` used in their evaluation.

Use the guided notebook: [`notebooks/Unlimited_OCR_SGLang_Server.ipynb`](../notebooks/Unlimited_OCR_SGLang_Server.ipynb).
This page explains the moving parts.

## Requirements

- Linux (Ubuntu 22.04+ recommended)
- NVIDIA GPU, 24 GB+ VRAM; A100/H100 for hundred-page documents
- NVIDIA driver + CUDA runtime compatible with the wheel
- Python 3.10 (managed by `uv` in the notebook)
- The customized wheel from
  <https://github.com/baidu/Unlimited-OCR/tree/main/wheel>

The stock `pip install sglang` does **not** include the Unlimited-OCR-2
integration. Use the wheel.

## Setup

```bash
cd __REPO__

# uv creates an isolated Python 3.10 environment
uv venv --python 3.10 .venv-uocr

uv pip install --python .venv-uocr/bin/python \
  wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl

uv pip install --python .venv-uocr/bin/python \
  pymupdf==1.27.2.2 fastokens==0.3.2 requests
```

If the wheel is not in `wheel/`, download it:

```bash
mkdir -p wheel
curl -L --fail -o wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl \
  "https://github.com/baidu/Unlimited-OCR/raw/main/wheel/sglang-0.5.20.dev13%2Bg75b9b06b02-py3-none-any.whl"
```

## Start the server

Use the script:

```bash
./scripts/start_sglang_server.sh
```

Or launch it manually with the upstream memory flags:

```bash
export SGLANG_UOCR_EANABLE_DOC_LIMIT=1
export SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW=10
export SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=10
export SGLANG_UOCR_MAX_PAGE_TOKENS=20000
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64
export SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1
export SGLANG_UOCR_SHM_WRAP_MIN_MB=1
export SGLANG_VLM_CACHE_SIZE_MB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv-uocr/bin/python -m sglang.launch_server \
  --model-path baidu/Unlimited-OCR-2 \
  --served-model-name Unlimited-OCR-2 \
  --chat-template unlimited-ocr \
  --tokenizer-backend fastokens \
  --tokenizer-worker-num 8 \
  --mem-fraction-static 0.7 \
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
  --host 0.0.0.0 \
  --port 10000
```

Wait for `The server is fired up and ready to roll!` or check:

```bash
curl http://127.0.0.1:10000/v1/models
```

## The flags that matter for memory

| Flag | Why |
|---|---|
| `--mem-fraction-static 0.7` | Share of VRAM reserved for weights + KV. 0.75 leaves too little headroom at 32+ concurrent requests. |
| `--max-prefill-tokens 2048` | Drives a temporary tensor of roughly `[tokens, context/32]`. 4096 doubles that peak. |
| `--cuda-graph-bs-prefill 1024 2048` | The largest bucket must equal `--max-prefill-tokens`. |
| `--cuda-graph-config ... full_prefill_max_req 64` | Request slots for the full-prefill graph. The default (8) makes 16/32 concurrency fall back to eager. |
| `SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64` | Caps the SAM attention bias chunk. The default 512 allows a ~462 MiB allocation and can OOM. |
| `SGLANG_UOCR_EANABLE_DOC_LIMIT=1` | Enables page-window scheduling. Set `0` to parse a whole document in one window. |
| `SGLANG_UOCR_DOC_*_PAGE_WINDOW=10` | Pages per window. Smaller lowers peak memory but hurts cross-page merging. |

## Client request

The notebook contains the full client. The essential request shape:

```python
payload = {
    "model": "Unlimited-OCR-2",
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 8000 * len(pages),
    "temperature": 0,
    "skip_special_tokens": False,
    "images_config": {"crop_mode": True},  # True = gundam, False = base
    "custom_logit_processor": NGRAM_PROCESSOR,
    "custom_params": {
        "ngram_size": 35,
        "window_size": 1024 if len(pages) > 1 else 128,
        "page_token_id": 128827,
        "eos_token_id": 1,
        "max_page_tokens": 20000,
    },
    "stream": True,
}
```

`content` is the page images in order plus a text part:
`"<PAGE><image>" * len(pages) + "Multi page parsing."`.

The response is a stream of server-sent events. Concatenate the `delta.content`
fields, then run the result through `vendor/process_unlimitedOCR2.py` to get
Markdown.

## Concurrency

`--max-running-requests 64` is the server-side slot count. The client should
match it with a bounded worker pool. Start with 4 concurrent documents; raise it
until the GPU is saturated or memory pressure appears.

For batch evaluation, the upstream `infer.py` sends each document as a single
request. A simple client loop with `concurrency=4` is a good starting point.

## Long documents

- Keep `SGLANG_UOCR_EANABLE_DOC_LIMIT=1` for hundred-page books.
- Use a 10-page window on 24 GB; try 20 on 80 GB.
- Set `MAX_PAGES=0` in the client to send the whole document.
- Use a generous timeout (the notebook uses 6000 seconds).
- Test with the first 10 pages before sending a 300-page book.

## Stopping the server

```bash
kill "$(cat sglang_server.pid)"
```

Or run the stop cell in the notebook.

## Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi` not found | Install the NVIDIA driver; this path is Linux + NVIDIA only. |
| Server exits immediately | Read `sglang_server.log`. Usually a driver, VRAM or wheel problem. |
| Port 10000 busy | Change `PORT` in the notebook/script. |
| Wheel download fails | Download from the upstream `wheel/` directory manually. |
| OOM during prefill | Lower `--max-prefill-tokens` to 1024 and `--mem-fraction-static` to 0.6. |
| OOM during long docs | Lower the page window to 5 and `--max-running-requests` to 16. |
| Requests hang | Check `nvidia-smi`; the GPU may be at 100% for minutes on a long book. |
| Wrong output format | Make sure `--chat-template unlimited-ocr` and the custom logit processor are enabled. |
| `fastokens` import error | Install `fastokens==0.3.2` in the same venv as SGLang. |

## When not to use SGLang

- You are on a laptop, CPU, Mac, or Colab free tier.
- You only have a few pages.
- You do not want to manage a Linux GPU environment.

Use the Transformers notebooks instead.
