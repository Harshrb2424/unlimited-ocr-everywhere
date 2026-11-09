#!/usr/bin/env python3
"""Print a friendly hardware / dependency report for Unlimited-OCR-2.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import os
import platform
import shutil
import sys


def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 0.0


def main() -> int:
    print("=" * 68)
    print("Unlimited OCR Everywhere - environment check")
    print("=" * 68)
    print("Python :", sys.version.split()[0])
    print("System :", platform.platform())
    print("Machine:", platform.machine())
    print("CPU    :", platform.processor() or "unknown")
    ram = total_ram_gb()
    if ram:
        print(f"RAM    : {ram:.1f} GB")
    print("nvidia-smi:", shutil.which("nvidia-smi") or "not found")

    try:
        import torch
    except ImportError:
        print("\nPyTorch: NOT INSTALLED")
        print("Install it with:")
        print("  pip install torch torchvision")
        print("CPU-only:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("\nRecommendation: read docs/INSTALL.md, then open the notebook for your hardware.")
        return 0

    print("\nPyTorch:", torch.__version__)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        count = torch.cuda.device_count()
        print("CUDA devices:", count)
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            cap = torch.cuda.get_device_capability(i)
            vram = props.total_memory / 1e9
            print(
                f"  [{i}] {props.name} | {vram:.1f} GB | compute {cap[0]}.{cap[1]}"
            )
        best = max(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(count)
        ) / 1e9
        if best >= 40:
            tier = "Colab L4/A100 or Local GPU notebook (whole-document mode)"
        elif best >= 22:
            tier = "Colab L4 notebook or Local GPU notebook (8 pages/request)"
        elif best >= 15:
            tier = "Colab T4 notebook or Local GPU notebook (1-2 pages/request)"
        else:
            tier = "Local GPU notebook with MODE='base' and CHUNK_PAGES=1 (8-12 GB is tight)"
        print("\nRecommended notebook:", tier)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS available: True")
        print("\nRecommended notebook: notebooks/Unlimited_OCR_Apple_Silicon.ipynb")
        print("Note: Apple Silicon support is experimental.")
    else:
        print("GPU: none detected")
        print("\nRecommended notebook: notebooks/Unlimited_OCR_Local_CPU.ipynb")
        print("CPU inference is experimental and very slow.")
        print("If you have internet, the free Colab T4 notebook is much faster.")

    try:
        import transformers

        print("\ntransformers:", transformers.__version__)
        if transformers.__version__ != "4.52.1":
            print("WARNING: this project pins transformers==4.52.1.")
            print("  pip install transformers==4.52.1")
    except ImportError:
        print("\ntransformers: NOT INSTALLED")
        print("  pip install -r requirements/base.txt")

    for package in ("huggingface_hub", "tokenizers", "pymupdf", "PIL", "accelerate"):
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "?")
            print(f"{package}: {version}")
        except ImportError:
            print(f"{package}: NOT INSTALLED")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
