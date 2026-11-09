# Changelog

All notable changes to this packaging project are documented here. The model,
paper and upstream code are versioned by Baidu; see
<https://github.com/baidu/Unlimited-OCR>.

## [1.0.0] — 2026-11-09

### Added

- Six self-contained notebooks:
  - Google Colab free T4
  - Google Colab L4 / A100
  - Local NVIDIA GPU (Windows / Linux)
  - Local CPU (experimental)
  - Apple Silicon MPS (experimental)
  - SGLang server for long documents
- `scripts/ocr_pdf.py`, a batch PDF/image to Markdown CLI with the same output
  format as the notebooks.
- `scripts/check_environment.py`, a hardware and dependency report that
  recommends the right notebook.
- `scripts/start_sglang_server.sh`, a production server launcher with the
  upstream memory flags.
- Vendored upstream post-processor (`vendor/process_unlimitedOCR2.py`, MIT).
- Automatic OOM retry: requests are split in half until they fit.
- Float16 image-preprocessing patch for T4/Turing and CPU/MPS.
- `.cuda()` compatibility patch for CPU and Apple Silicon.
- Output structure with Markdown, raw layout, metadata, cropped figures and a
  batch summary.
- Docs: install, hardware, Colab, SGLang, troubleshooting, FAQ.
- GitHub issue templates (bug, hardware report, tier request), PR template,
  Dependabot, and a notebook validation workflow.
- A synthetic example PDF (`examples/sample_invoice.pdf`).

### Known limitations

- CPU and Apple Silicon are experimental and much slower than CUDA.
- The practical GPU minimum is 12 GB of VRAM.
- The SGLang path requires Linux, an NVIDIA GPU and the customized wheel.
- The upstream model is pinned to `transformers==4.52.1`.
