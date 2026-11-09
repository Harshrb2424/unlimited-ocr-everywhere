#@title 2. Start the SGLang server
import os
import subprocess
import time
from pathlib import Path

import requests

VENV = Path.cwd() / ".venv-uocr"
PY = VENV / "bin" / "python"
MODEL_ID = "baidu/Unlimited-OCR-2"
HOST = "127.0.0.1"
PORT = 10000
SERVER_URL = f"http://{HOST}:{PORT}"

env = os.environ.copy()
env.update(
    {
        # Page-window scheduling for very long documents.
        "SGLANG_UOCR_EANABLE_DOC_LIMIT": "1",
        "SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW": "10",
        "SGLANG_UOCR_DOC_BASE_PAGE_WINDOW": "10",
        "SGLANG_UOCR_MAX_PAGE_TOKENS": "20000",
        # Lower peak memory for the SAM attention bias.
        "SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB": "64",
        # Pass large crop tensors through /dev/shm.
        "SGLANG_UOCR_SHM_WRAP_MODEL_DATA": "1",
        "SGLANG_UOCR_SHM_WRAP_MIN_MB": "1",
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)

cmd = [
    str(PY), "-m", "sglang.launch_server",
    "--model-path", MODEL_ID,
    "--served-model-name", "Unlimited-OCR-2",
    "--chat-template", "unlimited-ocr",
    "--tokenizer-backend", "fastokens",
    "--tokenizer-worker-num", "8",
    "--mem-fraction-static", "0.7",
    "--max-running-requests", "64",
    "--max-prefill-tokens", "2048",
    "--enable-custom-logit-processor",
    "--enable-streaming-session",
    "--allow-auto-truncate",
    "--disable-radix-cache",
    "--mm-preprocess-cache-size-mb", "0",
    "--cuda-graph-backend-decode", "full",
    "--cuda-graph-max-bs-decode", "64",
    "--cuda-graph-backend-prefill", "full",
    "--cuda-graph-max-bs-prefill", "16384",
    "--cuda-graph-bs-prefill", "1024", "2048",
    "--cuda-graph-config",
    '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}',
    "--host", HOST,
    "--port", str(PORT),
]

log_handle = open("sglang_server.log", "wb")
proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
Path("sglang_server.pid").write_text(str(proc.pid), encoding="utf-8")
print("Started SGLang PID", proc.pid)
print("Log: sglang_server.log")
print("Waiting for the server. The first start downloads the weights (5-20 min) ...")

deadline = time.time() + 1800
while time.time() < deadline:
    if proc.poll() is not None:
        print("Server exited with code", proc.returncode)
        print(Path("sglang_server.log").read_text(errors="replace")[-5000:])
        raise RuntimeError("SGLang server failed to start")
    try:
        response = requests.get(f"{SERVER_URL}/v1/models", timeout=2)
        if response.status_code == 200:
            print("Server ready:", SERVER_URL)
            break
    except Exception:
        pass
    time.sleep(5)
else:
    raise RuntimeError("Timed out waiting for SGLang. Inspect sglang_server.log.")
