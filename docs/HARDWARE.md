# Hardware guide

Unlimited-OCR-2 is a 3B-parameter vision-language model. The BF16 checkpoint is
about **6.7 GB**. Everything above that is activations, the vision encoder, the
KV cache and framework overhead.

## The short version

| Your hardware | Use this | Expectation |
|---|---|---|
| Free Colab T4 (16 GB) | `notebooks/Unlimited_OCR_Colab_T4.ipynb` | Works. Slow for long books. |
| Colab L4 (24 GB) | `notebooks/Unlimited_OCR_Colab_L4_A100.ipynb` | Good balance. |
| Colab A100 (40/80 GB) | `notebooks/Unlimited_OCR_Colab_L4_A100.ipynb` | Whole-document mode on 80 GB. |
| RTX 3060 12 GB | `notebooks/Unlimited_OCR_Local_GPU.ipynb` | One page at a time, base mode. |
| RTX 4090 24 GB | `notebooks/Unlimited_OCR_Local_GPU.ipynb` | Comfortable, 8 pages/request. |
| RTX 5090 / A6000 48 GB | `notebooks/Unlimited_OCR_Local_GPU.ipynb` | Whole-document mode. |
| CPU-only, 16 GB RAM | `notebooks/Unlimited_OCR_Local_CPU.ipynb` | Experimental. Minutes/page. |
| M1/M2/M3/M4, 16 GB | `notebooks/Unlimited_OCR_Apple_Silicon.ipynb` | Experimental. Small docs. |
| Linux + A100/H100 | `notebooks/Unlimited_OCR_SGLang_Server.ipynb` | Production path. |

## Why 12 GB is the practical GPU minimum

A rough budget for one page in base mode:

| Item | Approximate memory |
|---|---|
| BF16 weights | 6.7 GB |
| Vision encoder activations | 1–3 GB |
| KV cache / decoding | 1–3 GB |
| CUDA context + fragmentation | 1–2 GB |
| **Typical peak** | **10–15 GB** |

On an 8 GB card you can sometimes fit a single page with `MODE="base"`,
`DPI=150` and a short output budget, but the margin is thin. A 12 GB card is a
much better experience. 24 GB removes most of the anxiety.

## VRAM and chunk size

The notebooks choose `CHUNK_PAGES` from VRAM:

| VRAM | Default pages per request |
|---|---|
| 10–14 GB | 1 |
| 15–21 GB | 2 |
| 22–39 GB | 8 |
| 40 GB+ | whole document (0) |

You can always override `CHUNK_PAGES` in the configuration cell. Lower is safer;
higher is faster when it fits.

## Modes

- **`base`** — one global view per page. Lowest memory. Best for T4, 8–12 GB
  cards, CPU and MPS.
- **`gundam`** — tiled views. More visual detail, more memory. Best for a single
  page on a 24 GB+ GPU.
- **`auto`** — `base` for multi-page and low-VRAM devices, `gundam` for a single
  page on a large GPU.

For long documents, `base` with chunking is usually the right trade-off.

## DPI

| DPI | Use |
|---|---|
| 150 | Low-memory emergency, small text may suffer |
| 200 | Good default for most documents |
| 250 | Dense text, small tables, local GPU / L4+ |
| 300 | Maximum detail, highest memory and time |

## CPU reality check

A 3B model in float32 needs roughly 12 GB just for the weights. On a typical
laptop CPU:

- One page can take minutes to tens of minutes.
- A 50-page PDF can take hours.
- Memory pressure can kill the process.

Use the CPU notebook for a handful of pages, or use the free Colab T4. CPU
support exists so the repo is honest about "every device", not because it is a
good primary path.

## Apple Silicon reality check

MPS is promising but experimental for this model:

- 16 GB unified memory is the minimum; 32 GB is comfortable.
- The notebook disables CUDA-only Triton kernels and enables the MPS CPU
  fallback.
- Some custom ops may fall back to the CPU, which is slow but usually correct.
- For a long document, use a CUDA GPU or Colab.

## Cloud options

| Service | GPU | Notes |
|---|---|---|
| Google Colab Free | T4 16 GB | Best free starting point |
| Google Colab Pro | L4 24 GB / A100 40 GB | Good balance |
| Google Colab Pro+ | A100 40/80 GB | Long documents |
| RunPod / Vast / Lambda | A100/H100 | Cheapest per hour for batches |
| Kaggle | T4/P100 | Free, but notebook setup differs |

The Colab notebooks are the only ones with Colab-specific cells (Drive mount,
ZIP download). The local GPU notebook works on any Linux NVIDIA cloud box.

## Measuring your own peak

After a run, `name.meta.json` contains timings and token counts. For GPU memory,
add this to the end of the processing cell:

```python
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.max_memory_allocated() / 1e9, "GB peak")
```

If peak is close to your VRAM, lower `CHUNK_PAGES` or `DPI` before processing a
larger document.
