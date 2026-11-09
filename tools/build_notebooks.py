#!/usr/bin/env python3
"""Build the downloadable notebooks from the templates in tools/templates/.

Run from the repository root:

    python tools/build_notebooks.py

The generated notebooks are self-contained: every code cell is inlined, so a
user can download a single .ipynb and run it without cloning the repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = Path(__file__).resolve().parent / "templates"
NOTEBOOKS = ROOT / "notebooks"
OWNER = "__OWNER__"
REPO = "__REPO__"
REPO_URL = f"https://github.com/{OWNER}/{REPO}"


def read_template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8").rstrip("\n")


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str, title: str | None = None) -> dict:
    metadata = {}
    if title:
        metadata["cellView"] = "form"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict], colab_gpu: str | None = None) -> dict:
    metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    if colab_gpu:
        metadata["colab"] = {"provenance": [], "gpuType": colab_gpu}
        metadata["accelerator"] = "GPU"
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


TIER_INTRO = """# {title}

**{subtitle}**

This notebook turns every PDF or image in `input/` into Markdown in `output/`
using Baidu's **Unlimited-OCR-2** model. It is self-contained: dependencies,
model download, PDF rendering, post-processing and figure cropping are handled
for you.

| Item | Value |
|---|---|
| **Hardware** | {hardware} |
| **Model** | `baidu/Unlimited-OCR-2` (3B parameters, ~6.7 GB in BF16) |
| **First run** | Downloads ~7 GB of weights; expect 10-20 minutes on a fast connection |
| **Privacy** | Local inference. On Colab, files are processed in Google's cloud. |
| **Repo** | [{repo_url}]({repo_url}) |

## How to use

{steps}

> **New here?** Start with the free T4 notebook. It is the lowest-friction way
> to try the model. If a document is long or you need production throughput,
> move to the L4/A100 notebook or the SGLang server notebook.
"""

COLAB_STEPS = """1. In Colab: **Runtime -> Change runtime type -> Hardware accelerator -> GPU**
   and pick **{gpu}**.
2. **Runtime -> Run all**.
3. Wait until the notebook prints **WAITING FOR FILES**.
4. Open the folder icon in the left sidebar, open `input/`, and upload PDFs or images.
5. The notebook processes them automatically and writes Markdown to `output/`.
6. A ZIP of `output/` downloads when the run finishes.

> Colab's `/content` storage is temporary. Set `USE_DRIVE = True` in the
> configuration cell to keep `input/` and `output/` in Google Drive instead.
"""

LOCAL_STEPS = """1. Install Python 3.10-3.13. For NVIDIA GPUs, install a recent driver.
2. Run the install cell, then **Run all**.
3. Put PDFs or images in `input/` (next to this notebook).
4. Run the processing cell again whenever you add more files.
5. Results appear in `output/`; a ZIP is created next to it.

The notebook uses the current working directory. If you opened it from the
repository's `notebooks/` folder, `input/` and `output/` are created there.
"""

TIER_CONFIG_MD = """## Configuration

The defaults below are tuned for **{hardware_short}**. You normally do not need
to change anything. The important knobs are:

| Setting | What it does |
|---|---|
| `MODE` | `auto` (recommended), `base` (lowest memory), `gundam` (most detail) |
| `CHUNK_PAGES` | Pages per model request. `0` = choose from available memory. |
| `DPI` | PDF render resolution. 200 is a good default; 300 is slower. |
| `MAX_PAGES` | `0` = all pages. Set a number to cap very long documents. |
| `MAX_TOKENS_PER_PAGE` | Output budget per page. Raise it if a page is cut off. |
| `FORCE_DTYPE` | `auto` picks BF16/FP16/FP32 for the device. |
| `USE_TRITON` | CUDA-only fused kernels. Disable if you see Triton errors. |

**This tier:** {config_note}

If you change `transformers` or other packages after this notebook has imported
them, use **Runtime -> Restart runtime** (Colab) or **Kernel -> Restart**
(Jupyter), then run all cells again.
"""

TROUBLESHOOTING = """## Troubleshooting

{tier_fixes}

### The model returns empty output or cuts off

- Raise `MAX_TOKENS_PER_PAGE` (for example 8000 -> 12000).
- Check `name.meta.json`: `"truncated": true` means the output hit the budget.
- Process fewer pages per request: `CHUNK_PAGES = 1`.

### Transformers or import errors

This notebook pins `transformers==4.52.1`, the version the model's remote code
targets. Restart the runtime/kernel after installing, then run all cells.

### `CUDA out of memory`

- Set `MODE = "base"`.
- Set `CHUNK_PAGES = 1`.
- Lower `DPI` to 150-200.
- Lower `MAX_TOKENS_PER_PAGE` to 4000.
- Use a larger GPU, or open the SGLang notebook for long documents.

The notebook also retries automatically by splitting a request into smaller
page groups when it hits an out-of-memory error.

### Triton / custom kernel errors

Set `USE_TRITON = False` in the configuration cell, restart the runtime/kernel,
and run all cells again. This uses the slower pure-PyTorch attention path.

### Colab disconnected and files are gone

Set `USE_DRIVE = True` in the configuration cell, or download the ZIP before
the runtime ends. Colab's `/content` storage is temporary.

### Long documents and production use

A free Colab GPU is not meant for hundred-page documents. Use `CHUNK_PAGES` to
limit memory, or move to the SGLang server notebook, which is designed for
long-horizon parsing and concurrent requests.
"""

TIER_FIXES = {
    "t4": """### T4-specific notes

- The T4 is Turing-class, so it has **no native BF16**. The notebook uses
  `float16` and patches the model's image preprocessing automatically.
- 16 GB is the practical minimum. Keep `MODE = "auto"` and `CHUNK_PAGES = 1`
  for reliable runs; 2 pages per request often works but can OOM on dense pages.
- If a single page OOMs, set `MODE = "base"` and `DPI = 150`.
- The T4 is slow for hundred-page documents. For a 50+ page book, use the
  L4/A100 notebook or the SGLang server notebook.
""",
    "pro": """### L4 / A100-specific notes

- L4 (24 GB) and A100 (40/80 GB) support BF16, so the notebook uses it
  automatically.
- `CHUNK_PAGES = 8` is a safe starting point on L4. On an 80 GB A100 you can set
  `CHUNK_PAGES = 0` to send the whole document in one request.
- Gundam mode (tiled views) is used automatically for single pages on large
  GPUs. It is more accurate but uses more memory.
- Colab Pro can still disconnect. Enable `USE_DRIVE` for long runs.
""",
    "gpu": """### Local GPU-specific notes

- **Windows:** WSL2 with an NVIDIA driver is the most reliable setup. Native
  Windows works, but long paths and CUDA DLL conflicts are more common.
- If PyTorch cannot see the GPU, install the wheel that matches your driver:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
  (or `cu121`, `cu118`).
- 8 GB cards may work in `base` mode with one page, but 12 GB is the practical
  minimum. Close games, browsers and other CUDA apps before running.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the notebook does this
  automatically) if you see fragmentation errors.
""",
    "cpu": """### CPU-specific notes

- CPU inference is **experimental and very slow**. A 3B model in float32 needs
  roughly 12 GB of RAM just for the weights; 16 GB is the minimum, 32 GB is
  comfortable.
- The notebook defaults to one page per request, 150 DPI and a 5-page cap. Do
  not expect interactive speed.
- For a free GPU, use the Colab T4 notebook instead. It is dramatically faster.
- Set `OMP_NUM_THREADS` to the number of physical cores if CPU usage is low.
- If the process is killed, your machine ran out of RAM. Lower `MAX_PAGES`,
  `DPI` and `MAX_TOKENS_PER_PAGE`, or use a machine with more memory.
""",
    "mps": """### Apple Silicon-specific notes

- MPS support is **experimental**. The notebook disables CUDA-only Triton
  kernels and sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back
  to the CPU.
- 16 GB of unified memory is the minimum; 32 GB is more comfortable. Close
  memory-hungry apps before running.
- Use `MODE = "base"` and `CHUNK_PAGES = 1` for reliability. Gundam tiling can
  push the vision encoder over the memory limit.
- If you see `MPS backend out of memory`, restart the kernel, lower `DPI`, and
  process a single page.
- For serious throughput, use a CUDA GPU or the SGLang server notebook.
""",
}


def tier_markdown(
    title: str,
    subtitle: str,
    hardware: str,
    hardware_short: str,
    steps: str,
    config_note: str,
    tier_fixes: str,
) -> tuple[str, str, str, str]:
    intro = TIER_INTRO.format(
        title=title,
        subtitle=subtitle,
        hardware=hardware,
        repo_url=REPO_URL,
        steps=steps,
    )
    config = TIER_CONFIG_MD.format(
        hardware_short=hardware_short,
        config_note=config_note,
    )
    fixes = TROUBLESHOOTING.format(tier_fixes=tier_fixes)
    return intro, config, fixes


def build_standard(
    *,
    filename: str,
    title: str,
    subtitle: str,
    hardware: str,
    hardware_short: str,
    colab_gpu: str | None,
    steps: str,
    config_note: str,
    tier_fixes: str,
    install: str,
    root_block: str,
    dpi: int,
    max_pages: int,
    mode: str,
    chunk_pages: int,
    max_tokens_per_page: int,
    auto_download_zip: bool,
    make_zip: bool,
    wait_for_files: bool,
    wait_minutes: int,
    force_dtype: str,
    use_triton: bool,
    require_device: str,
    extra_config: str = "",
) -> Path:
    intro, config_md, fixes = tier_markdown(
        title,
        subtitle,
        hardware,
        hardware_short,
        steps,
        config_note,
        tier_fixes,
    )
    config_code = read_template("config.py")
    replacements = {
        "__ROOT_BLOCK__": root_block,
        "__DPI__": str(dpi),
        "__MAX_PAGES__": str(max_pages),
        "__MODE__": mode,
        "__CHUNK_PAGES__": str(chunk_pages),
        "__MAX_TOKENS_PER_PAGE__": str(max_tokens_per_page),
        "__AUTO_DOWNLOAD_ZIP__": str(auto_download_zip),
        "__MAKE_ZIP__": str(make_zip),
        "__WAIT_FOR_FILES__": str(wait_for_files),
        "__WAIT_MINUTES__": str(wait_minutes),
        "__FORCE_DTYPE__": force_dtype,
        "__USE_TRITON__": str(use_triton),
        "__REQUIRE_DEVICE__": require_device,
        "__EXTRA_CONFIG__": extra_config,
    }
    for key, value in replacements.items():
        config_code = config_code.replace(key, value)

    cells = [
        md_cell(intro),
        code_cell(read_template(install)),
        md_cell(config_md),
        code_cell(config_code),
        md_cell(
            "## Load the model\n\n"
            "The cell below detects the device, picks a safe dtype, loads the "
            "model, and applies small compatibility patches for float16 and "
            "non-CUDA devices. It prints exactly what it chose before inference "
            "starts."
        ),
        code_cell(read_template("load_model.py")),
        code_cell(read_template("postprocess.py")),
        code_cell(read_template("helpers.py")),
        md_cell(
            "## Process the input folder\n\n"
            "Colab waits for uploads; local notebooks process whatever is already "
            "in `input/`. Re-run this cell whenever you add files."
        ),
        code_cell(read_template("run.py")),
        code_cell(read_template("preview.py")),
        code_cell(read_template("zip.py")),
        md_cell(fixes),
    ]
    path = NOTEBOOKS / filename
    path.write_text(
        json.dumps(notebook(cells, colab_gpu), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def build_sglang() -> Path:
    intro = f"""# Unlimited OCR Everywhere - SGLang Server

**For long documents and production throughput: Linux + NVIDIA GPU (24 GB+ VRAM).**

Unlimited-OCR-2 supports two backends. The **Transformers** notebooks are the
easiest way to parse a handful of pages. **SGLang** is the recommended backend
for serving: it adds page-window scheduling, streaming, concurrent requests and
the custom n-gram logit processor used in the paper's evaluation.

This notebook starts the customized SGLang server in the background, waits for
it to become healthy, then sends whole PDFs as single requests and saves
Markdown.

| | |
|---|---|
| **Hardware** | Linux, NVIDIA GPU, 24 GB+ VRAM (A100/H100 recommended) |
| **Model** | `baidu/Unlimited-OCR-2` (3B parameters, ~6.7 GB in BF16) |
| **Python** | 3.10, managed in a `uv` virtualenv by this notebook |
| **Repo** | [{REPO_URL}]({REPO_URL}) |

## How to use

1. Run **Runtime -> Run all** (or run the cells in order).
2. Wait for **Server ready**.
3. Put PDFs or images in `input/` and run the client cell.
4. Results appear in `output/`.
5. Run the stop cell when you are finished.

> This notebook is for a Linux machine with an NVIDIA GPU. On Colab, use the
> T4/L4/A100 Transformers notebooks instead: the customized SGLang wheel and
> CUDA graphs need a full Linux environment.

## What SGLang adds

- **Page-window scheduling** (`SGLANG_UOCR_EANABLE_DOC_LIMIT=1`) so hundred-page
  documents do not need one giant KV cache.
- **Streaming** through the OpenAI-compatible API.
- **Concurrency** for batch jobs.
- The **NoRepeatNGram logit processor** used by the model authors for long
  documents.

The server flags below are taken from the upstream README. The most important
memory knobs are `--mem-fraction-static` (0.7), `--max-prefill-tokens` (2048)
and the SAM bias chunk cap (`SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64`).
"""

    cells = [
        md_cell(intro),
        code_cell(read_template("sglang_install.py")),
        md_cell(
            "## Start the server\n\n"
            "The first start downloads the model weights and can take 5-20 "
            "minutes. The cell waits for `/v1/models` to answer before it "
            "returns. Server logs go to `sglang_server.log`."
        ),
        code_cell(read_template("sglang_server.py")),
        md_cell(
            "## Parse documents\n\n"
            "Set `DPI`, `MAX_PAGES` and `TOKENS_PER_PAGE` at the top of the next "
            "cell, then run it. It processes every PDF/image in `input/` and "
            "writes Markdown, raw output, metadata and cropped figures to "
            "`output/`."
        ),
        code_cell(read_template("sglang_client.py")),
        md_cell(
            "## Stop the server\n\n"
            "Run this when you are done, or before restarting with different "
            "memory flags."
        ),
        code_cell(
            """#@title 4. Stop the SGLang server
import os
import signal
from pathlib import Path

pid_file = Path("sglang_server.pid")
if not pid_file.exists():
    print("No pid file; the server may already be stopped.")
else:
    pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        os.kill(pid, signal.SIGTERM)
        print("Sent SIGTERM to", pid)
    except ProcessLookupError:
        print("Process", pid, "is already gone")
    pid_file.unlink(missing_ok=True)
"""
        ),
        md_cell(
            """## Troubleshooting

### The server exits immediately

Open `sglang_server.log`. The most common causes are:

- **No NVIDIA GPU / driver mismatch**: run `nvidia-smi` first.
- **Not enough VRAM**: lower `--mem-fraction-static` to `0.6` or use a larger GPU.
- **Port 10000 is busy**: change `PORT` in the server cell.
- **Wheel download failed**: download the wheel manually from
  `https://github.com/baidu/Unlimited-OCR/tree/main/wheel` and put it in `wheel/`.

### CUDA out of memory during long documents

- Lower `SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW` / `SGLANG_UOCR_DOC_BASE_PAGE_WINDOW`
  from `10` to `5` or `4`.
- Lower `--max-running-requests` from `64` to `16` or `8`.
- Lower `--max-prefill-tokens` from `2048` to `1024`.
- Keep `SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64`; the default 512 can allocate
  a ~462 MiB block and OOM.

### Requests time out

Long documents can take many minutes. The client uses a 6000-second timeout.
For a hundred-page book, start with a smaller `MAX_PAGES` to verify the pipeline,
then increase it.

### I only have a laptop / Colab

Use the Transformers notebooks instead. SGLang is a serving framework for Linux
GPU machines, not a laptop-first path.
"""
        ),
    ]
    path = NOTEBOOKS / "Unlimited_OCR_SGLang_Server.ipynb"
    path.write_text(
        json.dumps(notebook(cells), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


COLAB_ROOT = """USE_DRIVE = False
DRIVE_FOLDER = "/content/drive/MyDrive/UnlimitedOCR"

if USE_DRIVE:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        ROOT = Path(DRIVE_FOLDER)
    except Exception as exc:
        print("Drive mount failed, using temporary storage:", exc)
        ROOT = Path("/content")
else:
    ROOT = Path("/content")

INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
CODE_DIR = Path("/content/Unlimited-OCR")
VENDOR_DIR = CODE_DIR / "vendor"
"""

LOCAL_ROOT = """ROOT = Path.cwd()
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
REPO_DIR = ROOT.parent if (ROOT.parent / "vendor").exists() else ROOT
VENDOR_DIR = REPO_DIR / "vendor"
"""


def main() -> int:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    built = []
    built.append(
        build_standard(
            filename="Unlimited_OCR_Colab_T4.ipynb",
            title="Unlimited OCR Everywhere - Google Colab (free T4)",
            subtitle="The easiest path: a free Tesla T4, no install, upload and go.",
            hardware="Google Colab, NVIDIA Tesla T4 (16 GB VRAM)",
            hardware_short="a free Colab T4",
            colab_gpu="T4",
            steps=COLAB_STEPS.format(gpu="T4 GPU"),
            config_note=(
                "one page per request, 200 DPI, float16. This is the safest "
                "configuration for 16 GB. You can try `CHUNK_PAGES = 2` on "
                "simple documents."
            ),
            tier_fixes=TIER_FIXES["t4"],
            install="install_colab.py",
            root_block=COLAB_ROOT,
            dpi=200,
            max_pages=50,
            mode="auto",
            chunk_pages=1,
            max_tokens_per_page=8000,
            auto_download_zip=True,
            make_zip=True,
            wait_for_files=True,
            wait_minutes=30,
            force_dtype="auto",
            use_triton=True,
            require_device="cuda",
        )
    )
    built.append(
        build_standard(
            filename="Unlimited_OCR_Colab_L4_A100.ipynb",
            title="Unlimited OCR Everywhere - Google Colab (L4 / A100)",
            subtitle="More VRAM, BF16, larger chunks, long documents.",
            hardware="Google Colab Pro, NVIDIA L4 (24 GB) or A100 (40/80 GB)",
            hardware_short="Colab Pro L4/A100",
            colab_gpu="A100",
            steps=COLAB_STEPS.format(gpu="L4 or A100"),
            config_note=(
                "BF16, 250 DPI and 8 pages per request. On an 80 GB A100 set "
                "`CHUNK_PAGES = 0` to send the whole document in one request."
            ),
            tier_fixes=TIER_FIXES["pro"],
            install="install_colab.py",
            root_block=COLAB_ROOT,
            dpi=250,
            max_pages=0,
            mode="auto",
            chunk_pages=8,
            max_tokens_per_page=8000,
            auto_download_zip=True,
            make_zip=True,
            wait_for_files=True,
            wait_minutes=30,
            force_dtype="auto",
            use_triton=True,
            require_device="cuda",
        )
    )
    built.append(
        build_standard(
            filename="Unlimited_OCR_Local_GPU.ipynb",
            title="Unlimited OCR Everywhere - Local NVIDIA GPU (Windows / Linux)",
            subtitle="Your PC, your files, your GPU. No Colab, no upload limits.",
            hardware="Windows or Linux PC with an NVIDIA GPU (12 GB+ VRAM recommended)",
            hardware_short="a local NVIDIA GPU",
            colab_gpu=None,
            steps=LOCAL_STEPS,
            config_note=(
                "auto device, auto dtype and auto chunk size. A 12 GB card uses "
                "one page per request; a 24 GB card uses eight; a 40 GB+ card can "
                "send the whole document."
            ),
            tier_fixes=TIER_FIXES["gpu"],
            install="install_gpu.py",
            root_block=LOCAL_ROOT,
            dpi=250,
            max_pages=0,
            mode="auto",
            chunk_pages=0,
            max_tokens_per_page=8000,
            auto_download_zip=False,
            make_zip=True,
            wait_for_files=False,
            wait_minutes=0,
            force_dtype="auto",
            use_triton=True,
            require_device="auto",
        )
    )
    built.append(
        build_standard(
            filename="Unlimited_OCR_Local_CPU.ipynb",
            title="Unlimited OCR Everywhere - CPU-only Laptop (experimental)",
            subtitle="No GPU? It runs, but slowly. Use this for a few pages, not a book.",
            hardware="Any x86-64 or ARM64 CPU with 16 GB+ RAM (32 GB recommended)",
            hardware_short="a CPU-only laptop",
            colab_gpu=None,
            steps=LOCAL_STEPS,
            config_note=(
                "float32, one page per request, 150 DPI and a 5-page cap. CPU "
                "inference is experimental; a free Colab T4 is much faster."
            ),
            tier_fixes=TIER_FIXES["cpu"],
            install="install_cpu.py",
            root_block=LOCAL_ROOT,
            dpi=150,
            max_pages=5,
            mode="base",
            chunk_pages=1,
            max_tokens_per_page=4000,
            auto_download_zip=False,
            make_zip=True,
            wait_for_files=False,
            wait_minutes=0,
            force_dtype="float32",
            use_triton=False,
            require_device="cpu",
        )
    )
    built.append(
        build_standard(
            filename="Unlimited_OCR_Apple_Silicon.ipynb",
            title="Unlimited OCR Everywhere - Apple Silicon (M1-M4, experimental)",
            subtitle="Run locally on a Mac with the MPS backend.",
            hardware="Apple Silicon Mac (M1/M2/M3/M4) with 16 GB+ unified memory",
            hardware_short="Apple Silicon",
            colab_gpu=None,
            steps=LOCAL_STEPS,
            config_note=(
                "MPS, float16, one page per request and a 10-page cap. MPS "
                "support is experimental; if a custom op fails, PyTorch falls "
                "back to the CPU automatically."
            ),
            tier_fixes=TIER_FIXES["mps"],
            install="install_mps.py",
            root_block=LOCAL_ROOT,
            dpi=200,
            max_pages=10,
            mode="base",
            chunk_pages=1,
            max_tokens_per_page=6000,
            auto_download_zip=False,
            make_zip=True,
            wait_for_files=False,
            wait_minutes=0,
            force_dtype="float16",
            use_triton=False,
            require_device="mps",
        )
    )
    built.append(build_sglang())

    for path in built:
        print("wrote", path.relative_to(ROOT))
    print(f"\n{len(built)} notebooks generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
