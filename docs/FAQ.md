# FAQ

### What is Unlimited-OCR-2?

A 3B-parameter vision-language model from Baidu's ERNIE Team for document
parsing. It reads page images and emits a structured layout stream (titles,
paragraphs, tables, formulas, figures, bounding boxes) that can be converted to
Markdown. It is designed for long documents, up to hundreds of pages.

### Is this the official Baidu repo?

No. This is an unofficial packaging project. The model, paper and upstream code
are by Baidu. See [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR).
The upstream project is MIT-licensed; this repo is too.

### Why are there six notebooks?

Because "it runs on an A800-80GB" is not useful if you have a T4, an RTX 3060, a
laptop, or a Mac. Each notebook tunes dtype, chunk size, DPI and mode for one
hardware tier, and documents the limits honestly.

### Do I need to clone the repo?

For a notebook, no. Each `.ipynb` is self-contained; download it and run it. The
repo is useful for the CLI, the vendored post-processor and development.

### Do I need a GPU?

A GPU is strongly recommended. The practical minimum is 12 GB of VRAM. CPU
inference works but is experimental and very slow. Apple Silicon MPS works but is
also experimental.

### Will it run on my 8 GB GPU?

Sometimes, for a single page in `base` mode with a low DPI. It is not a
comfortable experience. 12 GB is the practical minimum; 24 GB is comfortable.

### How much disk space?

About 12 GB free: ~7 GB of model weights, plus PyTorch, CUDA libraries and
output. The Hugging Face cache is reusable across runs.

### How long does the first run take?

On a fast connection: 10–20 minutes for the ~7 GB download, plus 1–3 minutes to
load. Later runs skip the download.

### Does it work offline?

After the first download, yes. The model and packages are cached. On Colab, the
cache is temporary; each new runtime downloads again.

### Does it handle scanned PDFs and photos?

Yes. Pages are rendered to images, so scans, phone photos and text-layer PDFs
all work. The model reads the rendered page, not the embedded text.

### Does it preserve tables?

Tables are emitted as HTML inside the Markdown. The upstream post-processor
merges cross-page tables and de-duplicates repeated headers. Most Markdown
renderers display HTML tables correctly.

### Does it extract figures?

Yes. Figure blocks are cropped from the rendered page into `name.assets/` and
linked from the Markdown.

### Does it output formulas?

The model emits formula blocks in the layout stream; the post-processor keeps
them in the Markdown. Check the upstream post-processor for the exact handling of
your document type.

### Can it handle a 300-page book?

Yes, with enough memory and the right backend. On a T4, no. On an L4/A100 with
`CHUNK_PAGES` tuned, or on SGLang with page-window scheduling, yes. Test the
first 10 pages before sending the whole book.

### What is the difference between `base` and `gundam`?

`base` encodes one global view per page: lowest memory, good for multi-page
documents and small GPUs. `gundam` encodes tiled views: more visual detail, more
memory, better for a single dense page on a large GPU.

### What does `CHUNK_PAGES = 0` mean?

"Choose a safe value from available memory." On a 40 GB+ GPU it means the whole
document in one request. On a 12 GB card it means one page. Set a number to force
a specific chunk size.

### Why is `transformers` pinned to 4.52.1?

That is the version the model's remote code targets. Newer versions may work but
are not tested by the model authors. Use a separate virtual environment if
another project needs a newer version.

### Why does the T4 use float16 instead of bfloat16?

Turing GPUs (T4) have no native BF16. The notebook uses float16 and patches the
model's image preprocessing so the vision tensors match. Ampere+ (L4, A100,
RTX 30/40/50) uses BF16.

### Are CPU and Apple Silicon really supported?

They are **experimental**. The notebooks patch the model's hard-coded `.cuda()`
calls and disable CUDA-only Triton kernels. They work for small documents, but
they are not the primary path and not performance-tuned. The repo says so up
front rather than hiding it.

### Can I use it commercially?

The model and upstream code are MIT-licensed. Check the upstream repository and
model card for the definitive terms. This packaging repo is MIT.

### How do I cite it?

See the BibTeX entries in the README or [CITATION.cff](../CITATION.cff). Cite the
original Unlimited OCR Works papers, not this packaging repo, for the model
itself.

### How can I contribute?

Add a hardware report, fix a bug, improve docs, or add a new tier. See
[CONTRIBUTING.md](../CONTRIBUTING.md). Hardware reports are especially useful:
real-world success/failure data keeps the support table honest.

### How do I personalize the Colab badges and links?

The repo ships with `__OWNER__` / `__REPO__` placeholders so it can be forked
without hard-coding someone else's account. If a badge or absolute link still
shows a placeholder, run:

```bash
python tools/personalize.py --owner YOUR_GITHUB_USERNAME --repo unlimited-ocr-everywhere
```

Then commit. The Colab badges and absolute links will point at your fork.
