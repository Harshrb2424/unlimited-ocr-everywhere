# Installation

This repo supports four local setups: NVIDIA GPU (Windows/Linux), CPU-only,
Apple Silicon, and Google Colab. The notebooks install most dependencies
themselves; this page covers the parts that are easier to do once in a shell.

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.10 or 3.11 |
| PyTorch | 2.4+ | a recent 2.x build |
| `transformers` | exactly 4.52.1 | exactly 4.52.1 |
| Disk | 12 GB free | 20 GB free |
| GPU | 12 GB VRAM | 24 GB+ VRAM |
| CPU RAM (CPU mode) | 16 GB | 32 GB |

The model's remote code targets `transformers==4.52.1`. Do not upgrade it
casually; if another package forces a newer version, use a separate virtual
environment for this project.

## 1. Clone the repo

```bash
git clone https://github.com/__OWNER__/__REPO__.git
cd __REPO__
```

If you only want a notebook, you can download the `.ipynb` directly and skip
the clone. The notebooks are self-contained.

## 2. Create a virtual environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Install PyTorch for your hardware

PyTorch is not pinned in `requirements/base.txt` because the right wheel depends
on your GPU and driver.

### NVIDIA GPU

Check your driver first:

```bash
nvidia-smi
```

Then install the CUDA wheel that matches it. For a recent driver, CUDA 12.4 is a
safe default:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Other common builds:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True`.

### CPU only

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Apple Silicon

```bash
pip install torch torchvision
```

Verify MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 4. Install the remaining dependencies

```bash
pip install -r requirements/base.txt
```

Or use the tier file after installing PyTorch:

```bash
pip install -r requirements/gpu.txt   # NVIDIA
pip install -r requirements/cpu.txt   # CPU
```

## 5. Run the environment check

```bash
python scripts/check_environment.py
```

It prints Python, OS, RAM, PyTorch, CUDA devices, VRAM, MPS availability and the
recommended notebook. It does not download the model.

## 6. Start Jupyter

```bash
pip install jupyterlab
jupyter lab
```

Open the notebook for your tier from `notebooks/`.

## 7. Optional: CLI instead of a notebook

```bash
python scripts/ocr_pdf.py --input my.pdf
```

The first run downloads the model (~7 GB) to the Hugging Face cache
(`~/.cache/huggingface` by default). Set `HF_HOME` to move it:

```bash
export HF_HOME=/data/huggingface
```

## 8. Optional: pre-download the model

If you want to download the weights before the first notebook run:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("baidu/Unlimited-OCR-2")
PY
```

## Common install problems

### `torch.cuda.is_available()` is False

- The installed wheel is probably CPU-only. Reinstall with the CUDA index URL.
- On Windows, the CUDA toolkit is not required; the driver is. Update the NVIDIA
  driver.
- In WSL2, install the Windows NVIDIA driver, not a Linux driver inside WSL.

### `transformers` version mismatch

```bash
pip install --force-reinstall "transformers==4.52.1"
```

Then restart Jupyter's kernel.

### `pymupdf` fails to build

Use a binary wheel on a supported Python version:

```bash
pip install --only-binary=:all: "pymupdf==1.27.2.2"
```

### Disk space

The model is ~7 GB; PyTorch + CUDA libraries can add several GB. Keep at least
12 GB free.

### Corporate proxy / Hugging Face blocked

Set a mirror or a local path:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Or download the model on another machine and point `MODEL_ID` at the local
directory in the notebook/CLI (`--model /path/to/Unlimited-OCR-2`).
