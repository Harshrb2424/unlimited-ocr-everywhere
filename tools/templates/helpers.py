#@title 6. Helper functions
import gc
import json
import re
import shutil
import time
from pathlib import Path

from PIL import Image, ImageSequence

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

INPUT_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
RE_IMAGE_REF = re.compile(
    r"!\[\]\(page(\d+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.jpg\)", re.I
)


def cleanup_memory():
    gc.collect()
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    elif DEVICE == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def safe_doc_id(path):
    try:
        rel = path.resolve().relative_to(INPUT_DIR.resolve())
    except ValueError:
        rel = Path(path.name)
    raw = rel.with_suffix("").as_posix().replace("/", "__")
    doc_id = re.sub(r"[^\w.\-]+", "_", raw).strip("._")
    return doc_id or "document"


def list_inputs():
    return sorted(
        p
        for p in INPUT_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in INPUT_EXTS
        and not p.name.startswith("~$")
    )


def render_pdf(pdf, dest):
    doc = pymupdf.open(str(pdf))
    try:
        if doc.needs_pass:
            raise RuntimeError("PDF is password protected")
        total = doc.page_count
        if total == 0:
            raise RuntimeError("PDF has no pages")
        count = min(total, MAX_PAGES) if MAX_PAGES > 0 else total
        dest.mkdir(parents=True, exist_ok=True)
        pages = []
        for i in range(count):
            pix = doc[i].get_pixmap(dpi=DPI, alpha=False)
            path = dest / f"page_{i + 1:04d}.jpg"
            path.write_bytes(pix.tobytes("jpeg", jpg_quality=92))
            pages.append(path)
        return pages, total
    finally:
        doc.close()


def render_image(image_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    pages = []
    with Image.open(image_path) as src:
        for i, frame in enumerate(ImageSequence.Iterator(src)):
            if MAX_PAGES > 0 and i >= MAX_PAGES:
                break
            img = frame.convert("RGB")
            out = dest / f"page_{i + 1:04d}.jpg"
            img.save(out, "JPEG", quality=92)
            pages.append(out)
    if not pages:
        raise RuntimeError("Image has no frames")
    return pages, len(pages)


def render_document(path, dest):
    if path.suffix.lower() == ".pdf":
        return render_pdf(path, dest)
    return render_image(path, dest)


def normalize_chunk(raw):
    text = (raw or "").strip()
    while text.endswith("<PAGE>"):
        text = text[: -len("<PAGE>")].rstrip()
    if text and "<PAGE>" not in text:
        text = "<PAGE>" + text
    return text


def crop_figure_refs(markdown, pages, assets_dir):
    if not pages:
        return markdown
    assets_dir.mkdir(parents=True, exist_ok=True)
    made = 0

    def repl(match):
        nonlocal made
        page_no = int(match.group(1))
        if page_no < 1 or page_no > len(pages):
            return ""
        try:
            with Image.open(pages[page_no - 1]) as src:
                img = src.convert("RGB")
        except Exception:
            return ""
        width, height = img.size
        x1, y1, x2, y2 = (
            max(0, min(1000, int(match.group(i)))) for i in range(2, 6)
        )
        px1, py1 = int(x1 / 1000 * width), int(y1 / 1000 * height)
        px2, py2 = int(x2 / 1000 * width), int(y2 / 1000 * height)
        if px2 <= px1 or py2 <= py1:
            return ""
        name = f"page{page_no}_{x1}_{y1}_{x2}_{y2}.jpg"
        try:
            img.crop((px1, py1, px2, py2)).save(
                assets_dir / name, "JPEG", quality=90
            )
        except Exception:
            return ""
        made += 1
        return f"![]({assets_dir.name}/{name})"

    result = RE_IMAGE_REF.sub(repl, markdown)
    if made == 0:
        shutil.rmtree(assets_dir, ignore_errors=True)
    return result


def infer_chunk(paths):
    crop_mode = None if MODE == "auto" else (MODE == "gundam")
    # On small GPUs and non-CUDA devices, one global view is the safe default.
    if MODE == "auto" and len(paths) == 1:
        if DEVICE != "cuda" or (vram_gb and vram_gb < 20):
            crop_mode = False
    max_new = min(MAX_TOKENS_PER_PAGE * len(paths), MAX_MODEL_LEN - 2048)
    return model.infer(
        tokenizer=tokenizer,
        prompt="Multi page parsing.",
        image_file=[str(p) for p in paths],
        crop_mode=crop_mode,
        eval_mode=True,
        tps_interval=0 if DEVICE == "cpu" else 500,
        no_repeat_ngram_size=35,
        ngram_window=1024 if len(paths) > 1 else 128,
        page_count_guard=True,
        max_page_tokens=MAX_PAGE_TOKENS,
        max_new_tokens=max(1, max_new),
        max_length=MAX_MODEL_LEN,
        return_dict=True,
    )


def infer_with_retry(paths):
    try:
        return [infer_chunk(paths)]
    except RuntimeError as exc:
        message = str(exc).lower()
        oom = any(
            token in message
            for token in ("out of memory", "mps backend", "cuda error")
        )
        if not oom:
            raise
        cleanup_memory()
        if len(paths) > 1:
            mid = len(paths) // 2
            print(
                f"  out of memory: splitting {len(paths)} pages -> "
                f"{mid} + {len(paths) - mid}"
            )
            return infer_with_retry(paths[:mid]) + infer_with_retry(paths[mid:])
        raise RuntimeError(
            "Out of memory on a single page. Try MODE='base', lower DPI, "
            "lower MAX_TOKENS_PER_PAGE, or a machine with more memory."
        ) from exc


def process_document(path):
    started = time.time()
    doc_id = safe_doc_id(path)
    md_path = OUTPUT_DIR / f"{doc_id}.md"
    raw_path = OUTPUT_DIR / f"{doc_id}.raw.txt"
    meta_path = OUTPUT_DIR / f"{doc_id}.meta.json"
    assets_dir = OUTPUT_DIR / f"{doc_id}.assets"

    if md_path.exists() and not FORCE:
        print(f"  skip (already exists): {md_path.name}")
        return {"doc": doc_id, "status": "skipped"}

    work = OUTPUT_DIR / ".work" / doc_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    pages, total = render_document(path, work)
    print(f"  rendered {len(pages)} page(s) from {path.name}")

    chunk_size = CHUNK if CHUNK > 0 else len(pages)
    chunks = [pages[i : i + chunk_size] for i in range(0, len(pages), chunk_size)]

    raw_parts = []
    chunk_meta = []
    truncated = False

    for index, chunk in enumerate(chunks, 1):
        print(f"  chunk {index}/{len(chunks)}: {chunk[0].stem} .. {chunk[-1].stem}")
        print("    running model (this can take several minutes) ...")
        chunk_started = time.time()
        results = infer_with_retry(chunk)
        for res in results:
            raw_parts.append(normalize_chunk(res.get("text", "")))
            chunk_meta.append(
                {
                    "pages": res.get("expected_pages"),
                    "num_pages_out": res.get("num_pages_out"),
                    "completion_tokens": len(res.get("token_ids") or []),
                    "finish_reason": res.get("finish_reason"),
                    "seconds": round(
                        res.get("time") or (time.time() - chunk_started), 1
                    ),
                }
            )
            if res.get("finish_reason") == "length":
                truncated = True
        raw_path.write_text(
            "\n".join(p for p in raw_parts if p), encoding="utf-8"
        )
        print(f"    {sum(len(p) for p in raw_parts):,} chars so far")
        cleanup_memory()

    raw = "\n".join(p for p in raw_parts if p).strip()
    if not raw:
        raise RuntimeError("The model returned an empty result.")
    raw_path.write_text(raw, encoding="utf-8")

    markdown = convert(raw)
    if not markdown.strip():
        raise RuntimeError("Post-processing returned empty Markdown.")
    shutil.rmtree(assets_dir, ignore_errors=True)
    markdown = crop_figure_refs(markdown, pages, assets_dir)
    md_path.write_text(markdown, encoding="utf-8")

    elapsed = round(time.time() - started, 1)
    meta = {
        "doc": doc_id,
        "source": str(path),
        "kind": "pdf" if path.suffix.lower() == ".pdf" else "image",
        "pages_rendered": len(pages),
        "pages_total": total,
        "dpi": DPI,
        "mode": MODE,
        "chunk_pages": CHUNK,
        "chunks": chunk_meta,
        "truncated": truncated,
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "seconds": elapsed,
        "markdown_chars": len(markdown),
        "raw_chars": len(raw),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not KEEP_PAGES:
        shutil.rmtree(work, ignore_errors=True)
        try:
            (OUTPUT_DIR / ".work").rmdir()
        except OSError:
            pass

    print(f"  saved {md_path.name} ({len(markdown):,} chars, {elapsed}s)")
    if truncated:
        print("  WARNING: output hit max_new_tokens; the last page may be cut off.")
    return {
        "doc": doc_id,
        "status": "ok",
        "pages": len(pages),
        "seconds": elapsed,
        "truncated": truncated,
    }


def run_all():
    files = list_inputs()
    if not files:
        print("No PDFs or images found in", INPUT_DIR)
        return []

    print(f"Found {len(files)} input file(s)")
    results = []
    batch_started = time.time()
    for index, path in enumerate(files, 1):
        try:
            rel = path.relative_to(INPUT_DIR)
        except ValueError:
            rel = path.name
        print(f"[{index}/{len(files)}] {rel}")
        try:
            results.append(process_document(path))
        except Exception as exc:
            print("  ERROR:", exc)
            cleanup_memory()
            results.append(
                {"doc": safe_doc_id(path), "status": "error", "error": str(exc)}
            )

    summary = {
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "seconds": round(time.time() - batch_started, 1),
        "total": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "results": results,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nDone in {summary['seconds']}s: "
        f"{summary['ok']} ok, {summary['skipped']} skipped, "
        f"{summary['failed']} failed"
    )
    print("Results:", OUTPUT_DIR)
    return results
