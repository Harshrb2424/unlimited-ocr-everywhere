# Google Colab guide

The two Colab notebooks are the fastest way to try Unlimited-OCR-2. They handle
the GPU check, dependency install, model download, PDF rendering, Markdown
conversion and ZIP download.

## Which Colab notebook?

| Notebook | Runtime | VRAM | Use when |
|---|---|---|---|
| `Unlimited_OCR_Colab_T4.ipynb` | T4 GPU (free) | 16 GB | You want free and simple |
| `Unlimited_OCR_Colab_L4_A100.ipynb` | L4 / A100 (Pro) | 24–80 GB | Long docs, BF16, more speed |

## Step by step

1. Open the notebook with the **Open in Colab** badge in the README, or upload
   the `.ipynb` to Colab manually.
2. **Runtime → Change runtime type → Hardware accelerator → GPU**.
   - Free tier: choose **T4**.
   - Pro: choose **L4** or **A100**.
3. **Runtime → Run all**.
4. Wait for the install and model download. The first run is the slow one:
   - Dependencies: 1–3 minutes
   - Model download: ~7 GB, 5–20 minutes depending on connection
   - Model load: 1–3 minutes
5. When you see **WAITING FOR FILES**, open the folder icon in the left sidebar,
   open `input/`, and upload PDFs or images.
6. The notebook processes them and writes Markdown to `output/`.
7. A ZIP of `output/` downloads automatically when the run finishes.

You can add more files and re-run the processing cell without reloading the
model.

## Keeping files: Google Drive

Colab's `/content` is wiped when the runtime disconnects. To keep files:

1. In the configuration cell, set `USE_DRIVE = True`.
2. Run the cell. Colab asks for permission to mount your Drive.
3. `input/` and `output/` now live in
   `My Drive/UnlimitedOCR/`.

The model weights still download to the Colab machine's cache, not Drive. Each
new runtime downloads them again unless you cache them in Drive and point
`HF_HOME` at it (advanced; usually not worth it).

## Runtime limits

| Plan | GPU | Typical session limit |
|---|---|---|
| Free | T4 | A few hours, subject to availability |
| Pro | L4/A100 | Longer, still subject to limits |
| Pro+ | A100 | Longest |

Colab can disconnect for inactivity or capacity reasons. For a 100-page book,
download the ZIP as soon as it finishes, or run on a dedicated GPU box.

## T4 specifics

- The T4 is Turing-class: **no native BF16**. The notebook uses `float16` and
  patches the model's image preprocessing.
- 16 GB is enough for one page per request. `CHUNK_PAGES = 1` is the reliable
  setting.
- `CHUNK_PAGES = 2` often works for simple pages but can OOM on dense scans.
- Long books are slow. Use `MAX_PAGES` to test a chapter first.

## L4 / A100 specifics

- BF16 is native, so quality and speed are better.
- `CHUNK_PAGES = 8` is a safe L4 default.
- On an 80 GB A100, set `CHUNK_PAGES = 0` to send a whole document in one
  request. The model can use cross-page context.
- Gundam mode is used automatically for single pages on large GPUs.

## Uploading files

- Use the left sidebar folder icon → `input/` → upload.
- You can upload a ZIP and unzip it in a cell:
  ```python
  !unzip -q input/my_scans.zip -d input/
  ```
- Subfolders are supported; output names include the subfolder path.
- Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`,
  `.tiff`.

## Downloading results

- The notebook downloads a ZIP automatically (`AUTO_DOWNLOAD_ZIP = True`).
- You can also right-click `output/` in the file pane → **Download**.
- To download a single file:
  ```python
  from google.colab import files
  files.download("/content/output/document.md")
  ```

## Common Colab errors

### "No GPU detected"

You are on a CPU runtime. Change the runtime type to GPU and restart.

### "You are connected to a GPU runtime, but no GPU is available"

Colab capacity issue. Wait, restart the runtime, or try again later.

### Install cell asks to restart

Colab sometimes preloads a different `transformers`. After the install cell,
use **Runtime → Restart runtime**, then **Run all** again.

### "Session crashed after using all available RAM"

You are on a CPU/High-RAM issue, or the model is using too much host memory.
Use a GPU runtime, lower `CHUNK_PAGES`, or use a High-RAM runtime if available.

### `CUDA out of memory`

Set `MODE = "base"`, `CHUNK_PAGES = 1`, `DPI = 150`, and lower
`MAX_TOKENS_PER_PAGE`. The notebook also retries by splitting requests.

### ZIP download does not start

Pop-up blocking or the browser's download settings. Use the file pane to
download `output/` manually, or run the ZIP cell again.

## Using the CLI in Colab

The notebooks are easier, but you can run the CLI:

```python
!git clone https://github.com/Harshrb2424/unlimited-ocr-everywhere.git
%cd unlimited-ocr-everywhere
!python scripts/ocr_pdf.py --input /content/input --output /content/output --zip
```

This is useful for batch jobs or if you prefer a single command.
