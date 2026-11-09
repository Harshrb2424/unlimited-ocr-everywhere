# Notebooks

Six self-contained notebooks, one per hardware tier. Download the `.ipynb` you
need and run it; no clone required.

| Notebook | Hardware | Default |
|---|---|---|
| [Colab T4](Unlimited_OCR_Colab_T4.ipynb) | Free Colab, Tesla T4 16 GB | base, float16, 1 page |
| [Colab L4/A100](Unlimited_OCR_Colab_L4_A100.ipynb) | Colab Pro, L4 24 GB / A100 | auto, BF16, 8 pages |
| [Local NVIDIA GPU](Unlimited_OCR_Local_GPU.ipynb) | Windows/Linux, 12 GB+ VRAM | auto, auto chunk |
| [Local CPU](Unlimited_OCR_Local_CPU.ipynb) | 16 GB+ RAM | base, float32, 1 page |
| [Apple Silicon](Unlimited_OCR_Apple_Silicon.ipynb) | M1-M4, 16 GB+ unified | base, float16, 1 page |
| [SGLang Server](Unlimited_OCR_SGLang_Server.ipynb) | Linux + NVIDIA 24 GB+ | page windows, streaming |

## How the notebooks are built

The `.ipynb` files are generated from the cell templates in
[`../tools/templates/`](../tools/templates/) by
[`../tools/build_notebooks.py`](../tools/build_notebooks.py). This keeps the
six tiers consistent and makes it possible to fix a bug once.

```bash
python tools/build_notebooks.py
python tools/validate_notebooks.py
```

Do not hand-edit a generated notebook for a shared change; edit the template
and rebuild. Notebook-only tweaks (metadata, a tier-specific markdown cell) are
fine to make directly.

## Output format

Every notebook writes the same structure:

```
output/
├── document.md
├── document.raw.txt
├── document.meta.json
├── document.assets/
└── run_summary.json
```

See the [README](../README.md#what-it-does) for details.
