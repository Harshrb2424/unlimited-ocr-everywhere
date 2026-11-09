#@title 2. Configuration
from pathlib import Path
import os
import sys
import json
import re
import time
import shutil
import gc
import importlib.util

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
__ROOT_BLOCK__

# ---------------------------------------------------------------------------
# Model and quality
# ---------------------------------------------------------------------------
MODEL_ID = "baidu/Unlimited-OCR-2"

# 200 dpi is plenty for most documents. 300 dpi is slower and uses more memory.
DPI = __DPI__

# 0 = all pages. Set a number to limit very long documents.
MAX_PAGES = __MAX_PAGES__

# auto   = base for multi-page / low-VRAM, gundam for one page on a big GPU
# base   = one global view per page (lowest memory)
# gundam = tiled views (more detail, more memory)
MODE = "__MODE__"

# 0 = choose from available memory at load time. Set 1 for the lowest memory use.
CHUNK_PAGES = __CHUNK_PAGES__

MAX_TOKENS_PER_PAGE = __MAX_TOKENS_PER_PAGE__
MAX_PAGE_TOKENS = 20000
MAX_MODEL_LEN = 131072

# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------
# Reprocess documents whose .md already exists.
FORCE = False

# Keep the rendered page JPEGs in output/.work/ (they are deleted otherwise).
KEEP_PAGES = False

# Colab only: download a ZIP when the run finishes.
AUTO_DOWNLOAD_ZIP = __AUTO_DOWNLOAD_ZIP__

# Local only: create a ZIP next to the output folder.
MAKE_ZIP = __MAKE_ZIP__

# Colab: wait for uploads. Local: process whatever is already in input/.
WAIT_FOR_FILES = __WAIT_FOR_FILES__
WAIT_MINUTES = __WAIT_MINUTES__

# auto = bfloat16 on Ampere+, float16 on T4/Turing, float32 on CPU.
# Force one with "bfloat16", "float16" or "float32".
FORCE_DTYPE = "__FORCE_DTYPE__"

# The model's fused Triton kernels are CUDA-only. Turn off for CPU/MPS or if
# you see Triton errors.
USE_TRITON = __USE_TRITON__

# cuda / mps / cpu / auto
REQUIRE_DEVICE = "__REQUIRE_DEVICE__"

__EXTRA_CONFIG__

if MODE not in ("auto", "base", "gundam"):
    raise ValueError("MODE must be auto, base or gundam")
MAX_TOKENS_PER_PAGE = max(1, int(MAX_TOKENS_PER_PAGE))

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_README = """UNLIMITED OCR - INPUT FOLDER

Put PDF files or images in this folder (subfolders are allowed).

Supported: .pdf, .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff

Already-processed documents are skipped unless FORCE = True in the
configuration cell. Results are written to ../output/.

Tip: the first run downloads about 7 GB of model weights.
"""

OUTPUT_README = """UNLIMITED OCR - OUTPUT FOLDER

For each input file "name.pdf" (or "name.jpg") you get:

  name.md          final Markdown (figures are in name.assets/)
  name.raw.txt     raw model output (layouts, tables, boxes)
  name.meta.json   pages, timings, token counts, finish_reason
  name.assets/     cropped figure images referenced by name.md

run_summary.json is written after every batch.

You can re-run the notebook; existing .md files are skipped unless
FORCE = True.
"""

(INPUT_DIR / "README.txt").write_text(INPUT_README, encoding="utf-8")
(OUTPUT_DIR / "README.txt").write_text(OUTPUT_README, encoding="utf-8")

print("Input folder :", INPUT_DIR)
print("Output folder:", OUTPUT_DIR)
print("Device target:", REQUIRE_DEVICE)
print("Mode         :", MODE)
