#@title 3. Parse PDFs and images through the server
import base64
import gc
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import fitz
import requests
from PIL import Image, ImageSequence

SERVER_URL = "http://127.0.0.1:10000"
MODEL_NAME = "Unlimited-OCR-2"
INPUT_DIR = Path.cwd() / "input"
OUTPUT_DIR = Path.cwd() / "output"
VENDOR_DIR = Path.cwd() / "vendor"
if not VENDOR_DIR.exists():
    VENDOR_DIR = Path.cwd().parent / "vendor"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
MAX_PAGES = 0
TOKENS_PER_PAGE = 8000
CROP_MODE = True
FORCE = False
KEEP_PAGES = False
INPUT_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# The n-gram processor is only installed inside the SGLang venv. Ask that
# interpreter for the serialized processor string instead of importing sglang
# into the notebook kernel (which may have a different torch build).
PY = Path.cwd() / ".venv-uocr" / "bin" / "python"
NGRAM_PROCESSOR = subprocess.check_output(
    [
        str(PY),
        "-c",
        "from sglang.srt.sampling.custom_logit_processor import "
        "DeepseekOCRNoRepeatNGramLogitProcessor as P; print(P.to_str())",
    ],
    text=True,
).strip()

# Load the upstream post-processor (vendored copy first, download as fallback).
POST_PATH = VENDOR_DIR / "process_unlimitedOCR2.py"
VENDOR_DIR.mkdir(parents=True, exist_ok=True)
if not POST_PATH.exists() or POST_PATH.stat().st_size < 1000:
    url = (
        "https://raw.githubusercontent.com/baidu/Unlimited-OCR/main/"
        "utils/process_unlimitedOCR2.py"
    )
    urllib.request.urlretrieve(url, POST_PATH)
spec = importlib.util.spec_from_file_location("process_unlimitedOCR2", POST_PATH)
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)
convert = _module.convert

session = requests.Session()
session.trust_env = False
RE_IMAGE_REF = re.compile(
    r"!\[\]\(page(\d+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.jpg\)", re.I
)


def list_inputs():
    return sorted(
        p
        for p in INPUT_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in INPUT_EXTS
        and not p.name.startswith("~$")
    )


def safe_doc_id(path):
    try:
        rel = path.resolve().relative_to(INPUT_DIR.resolve())
    except ValueError:
        rel = Path(path.name)
    raw = rel.with_suffix("").as_posix().replace("/", "__")
    return re.sub(r"[^\w.\-]+", "_", raw).strip("._") or "document"


def render_document(path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    pages = []
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        try:
            if doc.needs_pass:
                raise RuntimeError("PDF is password protected")
            total = doc.page_count
            if total == 0:
                raise RuntimeError("PDF has no pages")
            count = min(total, MAX_PAGES) if MAX_PAGES > 0 else total
            matrix = fitz.Matrix(DPI / 72, DPI / 72)
            for i in range(count):
                pix = doc[i].get_pixmap(matrix=matrix, alpha=False)
                out = dest / f"page_{i + 1:04d}.jpg"
                pix.save(str(out))
                pages.append(out)
            return pages, total
        finally:
            doc.close()
    with Image.open(path) as src:
        for i, frame in enumerate(ImageSequence.Iterator(src)):
            if MAX_PAGES > 0 and i >= MAX_PAGES:
                break
            out = dest / f"page_{i + 1:04d}.jpg"
            frame.convert("RGB").save(out, "JPEG", quality=92)
            pages.append(out)
    if not pages:
        raise RuntimeError("Image has no frames")
    return pages, len(pages)


def encode_image(image_path):
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_content(image_paths):
    prompt = "<PAGE><image>" * len(image_paths) + "Multi page parsing."
    return [encode_image(path) for path in image_paths] + [
        {"type": "text", "text": prompt}
    ]


def generate(image_paths, crop_mode=True, tokens_per_page=8000):
    window_size = 1024 if len(image_paths) > 1 else 128
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": build_content(image_paths)}],
        "max_tokens": tokens_per_page * len(image_paths),
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"crop_mode": crop_mode},
        "custom_logit_processor": NGRAM_PROCESSOR,
        "custom_params": {
            "ngram_size": 35,
            "window_size": window_size,
            "page_token_id": 128827,
            "eos_token_id": 1,
            "max_page_tokens": 20000,
        },
        "stream": True,
    }
    response = session.post(
        f"{SERVER_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=6000,
        stream=True,
    )
    response.raise_for_status()

    chunks = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = (choices[0].get("delta") or {}).get("content") or ""
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks)


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

    raw = generate(pages, crop_mode=CROP_MODE, tokens_per_page=TOKENS_PER_PAGE)
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
        "mode": "gundam" if CROP_MODE else "base",
        "model": MODEL_NAME,
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
    return {
        "doc": doc_id,
        "status": "ok",
        "pages": len(pages),
        "seconds": elapsed,
    }


files = list_inputs()
print("Input folder:", INPUT_DIR)
print(f"Found {len(files)} file(s)")
results = []
for index, path in enumerate(files, 1):
    print(f"[{index}/{len(files)}] {path.name}")
    try:
        results.append(process_document(path))
    except Exception as exc:
        print("  ERROR:", exc)
        results.append({"doc": safe_doc_id(path), "status": "error", "error": str(exc)})
    gc.collect()

summary = {
    "model": MODEL_NAME,
    "server": SERVER_URL,
    "seconds": round(sum(r.get("seconds", 0) for r in results), 1),
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
    f"\nDone: {summary['ok']} ok, {summary['skipped']} skipped, "
    f"{summary['failed']} failed"
)
print("Results:", OUTPUT_DIR)
