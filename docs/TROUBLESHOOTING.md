# Troubleshooting

Start with the notebook for your tier. Most errors are one of: wrong runtime,
wrong PyTorch wheel, out of memory, or a version mismatch.

## First response checklist

1. Run `python scripts/check_environment.py`.
2. Confirm `transformers==4.52.1`.
3. Confirm the notebook matches your hardware.
4. Read `name.meta.json` for the failing document.
5. For Colab, confirm **Runtime → Change runtime type → GPU**.

## Installation

### `No module named torch`

```bash
pip install torch torchvision
```

For CPU only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### `torch.cuda.is_available()` is False

- You installed a CPU wheel. Reinstall with a CUDA index URL.
- The NVIDIA driver is missing or too old. Update it and reboot.
- On Windows, use the native driver, not a Linux driver in WSL.
- In a container, pass `--gpus all` and install the NVIDIA Container Toolkit.

### `No module named transformers` or version error

```bash
pip install --force-reinstall "transformers==4.52.1"
```

Restart the kernel/runtime. The model's remote code checks the version.

### `pymupdf` build error

```bash
pip install --only-binary=:all: "pymupdf==1.27.2.2"
```

### Hugging Face download stalls

- Check disk space (12 GB+ free).
- Set `HF_HOME` to a larger disk.
- Try `export HF_ENDPOINT=https://hf-mirror.com`.
- Resume: the Hugging Face cache keeps partial downloads.

### SSL / certificate errors

Update certificates or use your organization's proxy:

```bash
export REQUESTS_CA_BUNDLE=/path/to/ca-bundle.crt
export HF_HUB_ETAG_TIMEOUT=60
```

## Model loading

### `trust_remote_code` prompt

The model ships custom modeling files. Answer **yes** / pass
`trust_remote_code=True` (the notebooks do). If you are uncomfortable, review the
files in the Hugging Face repo first.

### `low_cpu_mem_usage` error

The notebooks retry without it automatically. If you are using the CLI, install
`accelerate`:

```bash
pip install accelerate==1.6.0
```

### `OutOfMemoryError` while loading

- Close other GPU apps.
- The checkpoint is ~6.7 GB; loading needs temporary headroom.
- On CPU, you need ~12 GB of free RAM for float32 weights.
- Use a machine with more memory or a GPU runtime.

### `KeyError` / `AttributeError` in remote code

Usually a `transformers` mismatch. Reinstall 4.52.1 and restart. If it persists,
delete the Hugging Face module cache:

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/baidu
```

The next load re-downloads the remote code.

## Inference

### CUDA out of memory

Try, in order:

1. `MODE = "base"`
2. `CHUNK_PAGES = 1`
3. `DPI = 150`
4. `MAX_TOKENS_PER_PAGE = 4000`
5. A larger GPU, or the SGLang path

The notebook automatically splits a page group in half on OOM and retries.

### MPS backend out of memory

- Restart the kernel.
- `MODE = "base"`, `CHUNK_PAGES = 1`, `DPI = 150`.
- Close other apps; unified memory is shared.
- If it still fails, use CPU mode for that document or a CUDA machine.

### Output is empty

- Check `name.raw.txt`. If it is empty, the model returned nothing.
- Try a higher-contrast page, a lower DPI, or `MODE = "gundam"` for a single page.
- Confirm the page is not blank or password-protected.

### Output is cut off

Open `name.meta.json`:

```json
{ "truncated": true }
```

Raise `MAX_TOKENS_PER_PAGE` (e.g. 8000 → 12000), lower `CHUNK_PAGES`, or split
the document manually.

### Repetition / loops

The model uses a no-repeat n-gram processor. If you see loops:

- Keep `no_repeat_ngram_size=35` (the notebooks do).
- For multi-page documents, use `ngram_window=1024`.
- Try a lower `MAX_TOKENS_PER_PAGE`.

### Tables are HTML, not Markdown pipe tables

This is expected. The upstream post-processor emits HTML tables so merged cells
and cross-page tables survive. GitHub, Obsidian, VS Code and most Markdown
renderers display them correctly.

### Figures are missing

- Figures are cropped into `name.assets/`.
- If the raw output has no `image` blocks, there were no figures to crop.
- Check that `KEEP_PAGES` is not needed for debugging; the crop uses the rendered
  page images, which are deleted after processing unless `KEEP_PAGES = True`.

### Text is garbled or has layout tokens

The post-processor handles the layout stream. If you see `<BLOCK>`, `<PAGE>` or
`<FORK>` in the Markdown, the post-processor was not applied. Run the
post-processor cell and re-run the processing cell, or use the CLI.

## Triton and custom kernels

### Triton error / `tl.make_block_ptr is deprecated`

That warning is harmless. An actual crash is different:

1. Set `USE_TRITON = False`.
2. Restart the runtime/kernel.
3. Run all cells.

The pure-PyTorch fallback is slower but stable.

### `torch.compile` errors

The Transformers path does not compile the model. The SGLang path may use CUDA
graphs; see the SGLang section.

## CPU-specific

### Process killed / OOM

- 16 GB RAM is the minimum; 32 GB recommended.
- Lower `MAX_PAGES`, `DPI`, `MAX_TOKENS_PER_PAGE`.
- Close browsers and other apps.
- Add swap (slow but avoids a hard kill).

### Very slow

Expected. CPU inference of a 3B vision-language model is minutes per page. Use
`OMP_NUM_THREADS` equal to physical cores:

```bash
export OMP_NUM_THREADS=8
```

### `.cuda()` error on CPU

The notebook patches the upstream forward pass and installs a
`torch.Tensor.cuda` shim. If you see this error, you are probably running a code
path that bypasses the patch. Use the notebook's cells in order, or use the CLI
(`scripts/ocr_pdf.py`), which applies the same patches.

## Apple Silicon-specific

### `MPS backend out of memory`

Restart the kernel; lower `DPI` and `MAX_PAGES`; use `MODE="base"`.

### Unsupported op / fallback warnings

Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (the notebook does). Warnings about CPU
fallback are expected.

### `torch.backends.mps.is_available()` is False

- macOS 12.3+ and an Apple Silicon Mac are required.
- Install the standard macOS PyTorch wheel, not a CUDA wheel.
- Rosetta Python may not expose MPS; use an arm64 Python.

## Colab-specific

### No GPU

Runtime → Change runtime type → GPU.

### Runtime disconnects

Enable `USE_DRIVE=True`, or download the ZIP as soon as it finishes.

### `Session crashed after using all available RAM`

Use a GPU runtime, lower `CHUNK_PAGES`, or use High-RAM if available.

### Install cell wants a restart

Restart the runtime, then Run all. The pinned `transformers` version must be the
one imported.

## SGLang-specific

See [SGLANG.md](SGLANG.md#troubleshooting). The short version:

- Check `sglang_server.log`.
- `nvidia-smi` must work.
- Port 10000 must be free.
- Use the customized wheel.
- Lower `--mem-fraction-static` and page windows if you OOM.

## Getting help

Open an issue with:

- Notebook name or `scripts/ocr_pdf.py`
- OS and Python version
- GPU/CPU model and memory
- The exact command or cell
- Full traceback
- Whether it failed during install, model load, or inference

The bug report template asks for these.
