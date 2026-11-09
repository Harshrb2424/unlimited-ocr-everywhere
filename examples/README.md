# Examples

## `sample_invoice.pdf`

A tiny synthetic one-page document you can run through any notebook before
uploading your own files. It contains:

- A title and subtitle
- Invoice metadata
- A four-row table with numbers
- A notes section
- A boxed figure-like area
- A footer

It is generated from plain PDF text, so it is small (a few KB), safe to commit,
and exercises the same render-to-image path as a scanned document.

### Try it

1. Open a notebook.
2. Put `sample_invoice.pdf` in `input/`.
3. Run the processing cell.
4. Compare `output/sample_invoice.md` with the PDF.

### CLI

```bash
python scripts/ocr_pdf.py --input examples/sample_invoice.pdf --output /tmp/uocr_demo
```

### Expected output shape

```markdown
# UNLIMITED OCR EVERYWHERE

Sample invoice - use this PDF to test the notebook

| Invoice number | UOCR-2026-001 |
...
```

The exact Markdown depends on the model, the mode, and the DPI. The point is to
verify that the pipeline runs end to end: render → infer → post-process → crop →
save.

If the sample works but your document does not, the problem is likely the
document (password protection, unusual fonts, very large pages) rather than the
installation.
