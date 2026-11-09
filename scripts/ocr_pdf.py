#!/usr/bin/env python3
"""Batch PDF / image to Markdown with Baidu Unlimited-OCR-2.

Examples
--------
    python scripts/ocr_pdf.py --input my.pdf
    python scripts/ocr_pdf.py --input scans/ --output out --mode auto
    python scripts/ocr_pdf.py --input book.pdf --device cuda --chunk-pages 4 --zip
    python scripts/ocr_pdf.py --input page.png --device cpu --mode base

The script is intentionally dependency-light: it uses the same model, post-
processor and output layout as the notebooks, so results are interchangeable.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

try:
    from PIL import Image, ImageSequence
except ImportError:  # pragma: no cover - friendly error in main()
    Image = None
    ImageSequence = None

SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
RE_IMAGE_REF = re.compile(
    r"!\[\]\(page(\d+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.jpg\)", re.I
)
POSTPROCESSOR_URL = (
    "https://raw.githubusercontent.com/baidu/Unlimited-OCR/main/"
    "utils/process_unlimitedOCR2.py"
)
ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs and images to Markdown with Unlimited-OCR-2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="PDF/image file(s) or directories to scan recursively",
    )
    parser.add_argument("--output", "-o", default="output", help="output directory")
    parser.add_argument("--model", default="baidu/Unlimited-OCR-2", help="HF model id")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="inference device",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="model dtype",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "base", "gundam"),
        help="image encoding mode",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    parser.add_argument(
        "--max-pages", type=int, default=0, help="0 = all pages"
    )
    parser.add_argument(
        "--chunk-pages",
        type=int,
        default=0,
        help="pages per request; 0 = choose from available memory",
    )
    parser.add_argument(
        "--max-tokens-per-page", type=int, default=8000, help="output budget per page"
    )
    parser.add_argument(
        "--max-page-tokens", type=int, default=20000, help="per-page output guard"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=131072, help="model context length"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing .md")
    parser.add_argument(
        "--keep-pages",
        action="store_true",
        help="keep rendered page JPEGs in output/.work/",
    )
    parser.add_argument(
        "--zip", action="store_true", help="create a ZIP of the output directory"
    )
    parser.add_argument(
        "--no-triton",
        action="store_true",
        help="disable the CUDA Triton kernels (slower, pure PyTorch)",
    )
    parser.add_argument(
        "--version", action="version", version="unlimited-ocr-everywhere 1.0.0"
    )
    return parser.parse_args()


def total_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 0.0


def pick_device(torch, requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "No CUDA GPU detected. Install the CUDA PyTorch wheel, or use "
                "--device cpu / --device mps."
            )
        return "cuda"
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("WARNING: MPS is unavailable; falling back to CPU (very slow).")
            return "cpu"
        return "mps"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(torch, device: str, requested: str):
    if requested != "auto":
        dtype = getattr(torch, requested)
        if device == "cpu" and dtype == torch.float16:
            print("float16 is not supported on CPU; using float32 instead.")
            return torch.float32
        return dtype
    if device == "cuda":
        try:
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            native_bf16 = torch.cuda.is_bf16_supported()
        return torch.bfloat16 if native_bf16 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def pick_chunk(device: str, vram_gb: float, requested: int) -> int:
    if requested > 0:
        return requested
    if device != "cuda":
        return 1
    if vram_gb >= 40:
        return 0
    if vram_gb >= 22:
        return 8
    if vram_gb >= 15:
        return 2
    return 1


def load_postprocessor():
    VENDOR.mkdir(parents=True, exist_ok=True)
    path = VENDOR / "process_unlimitedOCR2.py"
    if not path.exists() or path.stat().st_size < 1000:
        print("Downloading process_unlimitedOCR2.py ...")
        try:
            urllib.request.urlretrieve(POSTPROCESSOR_URL, path)
        except Exception as exc:
            raise SystemExit(
                f"Could not download the post-processor: {exc}\n"
                f"Download it manually from {POSTPROCESSOR_URL} to {path}"
            ) from exc
    spec = importlib.util.spec_from_file_location("process_unlimitedOCR2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.convert


def collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            files.extend(
                p
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTS
                and not p.name.startswith("~$")
            )
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTS:
                print(f"WARNING: skipping unsupported file: {path}")
                continue
            files.append(path)
        else:
            raise SystemExit(f"Input does not exist: {path}")
    return sorted(set(files))


def render_pdf(pdf: Path, dest: Path, dpi: int, max_pages: int):
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    doc = pymupdf.open(str(pdf))
    try:
        if doc.needs_pass:
            raise RuntimeError("PDF is password protected")
        total = doc.page_count
        if total == 0:
            raise RuntimeError("PDF has no pages")
        count = min(total, max_pages) if max_pages > 0 else total
        dest.mkdir(parents=True, exist_ok=True)
        pages = []
        for i in range(count):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            out = dest / f"page_{i + 1:04d}.jpg"
            out.write_bytes(pix.tobytes("jpeg", jpg_quality=92))
            pages.append(out)
        return pages, total
    finally:
        doc.close()


def render_image(path: Path, dest: Path, max_pages: int):
    dest.mkdir(parents=True, exist_ok=True)
    pages = []
    with Image.open(path) as src:
        for i, frame in enumerate(ImageSequence.Iterator(src)):
            if max_pages > 0 and i >= max_pages:
                break
            out = dest / f"page_{i + 1:04d}.jpg"
            frame.convert("RGB").save(out, "JPEG", quality=92)
            pages.append(out)
    if not pages:
        raise RuntimeError("Image has no frames")
    return pages, len(pages)


def safe_doc_id(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        rel = Path(path.name)
    raw = rel.with_suffix("").as_posix().replace("/", "__")
    return re.sub(r"[^\w.\-]+", "_", raw).strip("._") or "document"


def normalize_chunk(raw: str) -> str:
    text = (raw or "").strip()
    while text.endswith("<PAGE>"):
        text = text[: -len("<PAGE>")].rstrip()
    if text and "<PAGE>" not in text:
        text = "<PAGE>" + text
    return text


def crop_figure_refs(markdown: str, pages: list[Path], assets_dir: Path) -> str:
    if not pages:
        return markdown
    assets_dir.mkdir(parents=True, exist_ok=True)
    made = 0

    def repl(match: re.Match) -> str:
        nonlocal made
        page_no = int(match.group(1))
        if page_no < 1 or page_no > len(pages):
            return ""
        try:
            with Image.open(pages[page_no - 1]) as src:
                image = src.convert("RGB")
        except Exception:
            return ""
        width, height = image.size
        x1, y1, x2, y2 = (
            max(0, min(1000, int(match.group(i)))) for i in range(2, 6)
        )
        px1, py1 = int(x1 / 1000 * width), int(y1 / 1000 * height)
        px2, py2 = int(x2 / 1000 * width), int(y2 / 1000 * height)
        if px2 <= px1 or py2 <= py1:
            return ""
        name = f"page{page_no}_{x1}_{y1}_{x2}_{y2}.jpg"
        try:
            image.crop((px1, py1, px2, py2)).save(
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


def main() -> int:
    args = parse_args()

    try:
        import torch
    except ImportError:
        print(
            "PyTorch is not installed. Install it first:\n"
            "  pip install torch torchvision\n"
            "For a CPU-only machine:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        return 2

    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError:
        print(
            "transformers is not installed. Run:\n"
            "  pip install -r requirements/base.txt",
            file=sys.stderr,
        )
        return 2

    if Image is None:
        print("Pillow is not installed. Run: pip install pillow", file=sys.stderr)
        return 2

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = pick_device(torch, args.device)
    dtype = pick_dtype(torch, device, args.dtype)
    use_triton = (not args.no_triton) and device == "cuda"

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: CUDA | {name} | {vram_gb:.1f} GB")
        if vram_gb < 10:
            print(
                "ERROR: less than 10 GB VRAM. 12 GB is the practical minimum.",
                file=sys.stderr,
            )
            return 2
    elif device == "mps":
        vram_gb = 0.0
        print(f"Device: Apple Silicon MPS | unified memory {total_ram_gb():.1f} GB")
        print("WARNING: MPS support is experimental.")
    else:
        vram_gb = 0.0
        print(f"Device: CPU | {platform.processor() or 'unknown'} | RAM {total_ram_gb():.1f} GB")
        print("WARNING: CPU inference is experimental and very slow.")

    print("dtype:", str(dtype).replace("torch.", ""))
    print("Triton kernels:", "on" if use_triton else "off")

    print("Loading tokenizer and model (~6.7 GB) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_kwargs = dict(
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    if not use_triton:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        sparse_cfg = getattr(config, "sparse_attention_config", None)
        if isinstance(sparse_cfg, dict):
            sparse_cfg["dash_use_triton"] = False
        load_kwargs["config"] = config

    try:
        model = AutoModel.from_pretrained(
            args.model, low_cpu_mem_usage=True, **load_kwargs
        )
    except Exception as exc:
        print("low_cpu_mem_usage failed, retrying without it:", exc)
        gc.collect()
        model = AutoModel.from_pretrained(args.model, **load_kwargs)
    model = model.eval().to(device)
    torch.set_grad_enabled(False)
    print("Model loaded.")

    def patch_image_dtype(target_dtype):
        patched = 0
        wanted = {type(model).__module__}
        try:
            wanted.add(type(model.model).__module__)
        except Exception:
            pass
        for module_name, module in list(sys.modules.items()):
            if not (module_name.endswith("modeling_unlimitedocr") or module_name in wanted):
                continue
            original = getattr(module, "_uocr_build_inputs", None)
            if original is None or getattr(original, "_uocr_dtype_patched", False):
                continue

            def wrapper(*wrap_args, __orig=original, __dtype=target_dtype, **wrap_kwargs):
                input_ids, images, seq_mask, spatial = __orig(*wrap_args, **wrap_kwargs)
                if images is not None:
                    images = (images[0].to(__dtype), images[1].to(__dtype))
                return input_ids, images, seq_mask, spatial

            wrapper._uocr_dtype_patched = True
            module._uocr_build_inputs = wrapper
            patched += 1
        return patched

    def patch_non_cuda_forward():
        import inspect
        import textwrap

        module = sys.modules.get(type(model).__module__)
        if module is None:
            return []
        patched = []
        for cls_name, cls in list(vars(module).items()):
            if not inspect.isclass(cls) or cls.__module__ != module.__name__:
                continue
            fn = cls.__dict__.get("forward")
            if fn is None:
                continue
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            if ".cuda()" not in src:
                continue
            new_src = textwrap.dedent(src).replace(
                ".cuda()", ".to(inputs_embeds.device)"
            )
            namespace = dict(module.__dict__)
            try:
                exec(compile(new_src, f"<patched {cls_name}.forward>", "exec"), namespace)
                cls.forward = namespace["forward"]
                patched.append(cls_name)
            except Exception as exc:
                print(f"  could not patch {cls_name}.forward: {exc}")
        return patched

    if dtype != torch.bfloat16:
        count = patch_image_dtype(dtype)
        print(f"Patched image preprocessing: {count} module(s).")
    if device != "cuda":
        patched = patch_non_cuda_forward()
        print("Patched forward:", ", ".join(patched) or "no .cuda() calls found")
        try:
            def _cuda_shim(self, *shim_args, **shim_kwargs):
                return self.to(device)

            torch.Tensor.cuda = _cuda_shim
            print("Applied torch.Tensor.cuda compatibility shim.")
        except Exception as exc:
            print("Could not apply Tensor.cuda shim:", exc)

    chunk = pick_chunk(device, vram_gb, args.chunk_pages)
    print("Pages per request:", "whole document" if chunk == 0 else chunk)

    convert = load_postprocessor()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_inputs(args.input)
    if not files:
        print("No supported input files found.", file=sys.stderr)
        return 1
    print(f"Found {len(files)} input file(s)")
    print("Output:", output_dir)

    def cleanup():
        gc.collect()
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        elif device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def infer_chunk(paths: list[Path]):
        crop_mode = None if args.mode == "auto" else (args.mode == "gundam")
        if args.mode == "auto" and len(paths) == 1:
            if device != "cuda" or (vram_gb and vram_gb < 20):
                crop_mode = False
        max_new = min(
            args.max_tokens_per_page * len(paths), args.max_model_len - 2048
        )
        return model.infer(
            tokenizer=tokenizer,
            prompt="Multi page parsing.",
            image_file=[str(path) for path in paths],
            crop_mode=crop_mode,
            eval_mode=True,
            tps_interval=0 if device == "cpu" else 500,
            no_repeat_ngram_size=35,
            ngram_window=1024 if len(paths) > 1 else 128,
            page_count_guard=True,
            max_page_tokens=args.max_page_tokens,
            max_new_tokens=max(1, max_new),
            max_length=args.max_model_len,
            return_dict=True,
        )

    def infer_with_retry(paths: list[Path]):
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
            cleanup()
            if len(paths) > 1:
                mid = len(paths) // 2
                print(f"  out of memory: splitting {len(paths)} -> {mid} + {len(paths) - mid}")
                return infer_with_retry(paths[:mid]) + infer_with_retry(paths[mid:])
            raise RuntimeError(
                "Out of memory on a single page. Try --mode base, a lower --dpi, "
                "or a machine with more memory."
            ) from exc

    def process_document(path: Path) -> dict:
        started = time.time()
        doc_id = safe_doc_id(path)
        md_path = output_dir / f"{doc_id}.md"
        raw_path = output_dir / f"{doc_id}.raw.txt"
        meta_path = output_dir / f"{doc_id}.meta.json"
        assets_dir = output_dir / f"{doc_id}.assets"

        if md_path.exists() and not args.force:
            print(f"  skip (already exists): {md_path.name}")
            return {"doc": doc_id, "status": "skipped"}

        work = output_dir / ".work" / doc_id
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".pdf":
            pages, total = render_pdf(path, work, args.dpi, args.max_pages)
        else:
            pages, total = render_image(path, work, args.max_pages)
        print(f"  rendered {len(pages)} page(s) from {path.name}", flush=True)

        chunk_size = chunk if chunk > 0 else len(pages)
        chunks = [pages[i : i + chunk_size] for i in range(0, len(pages), chunk_size)]

        raw_parts: list[str] = []
        chunk_meta = []
        truncated = False
        for index, group in enumerate(chunks, 1):
            print(
                f"  chunk {index}/{len(chunks)}: {group[0].stem} .. {group[-1].stem}",
                flush=True,
            )
            chunk_started = time.time()
            results = infer_with_retry(group)
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
                "\n".join(part for part in raw_parts if part), encoding="utf-8"
            )
            print(f"    {sum(len(part) for part in raw_parts):,} chars so far")
            cleanup()

        raw = "\n".join(part for part in raw_parts if part).strip()
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
            "dpi": args.dpi,
            "mode": args.mode,
            "chunk_pages": chunk,
            "chunks": chunk_meta,
            "truncated": truncated,
            "model": args.model,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "seconds": elapsed,
            "markdown_chars": len(markdown),
            "raw_chars": len(raw),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if not args.keep_pages:
            shutil.rmtree(work, ignore_errors=True)
            try:
                (output_dir / ".work").rmdir()
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

    results = []
    batch_started = time.time()
    for index, path in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        try:
            results.append(process_document(path))
        except Exception as exc:
            print("  ERROR:", exc, file=sys.stderr)
            cleanup()
            results.append({"doc": path.stem, "status": "error", "error": str(exc)})

    summary = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "seconds": round(time.time() - batch_started, 1),
        "total": len(results),
        "ok": sum(1 for item in results if item.get("status") == "ok"),
        "failed": sum(1 for item in results if item.get("status") == "error"),
        "skipped": sum(1 for item in results if item.get("status") == "skipped"),
        "results": results,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nDone in {summary['seconds']}s: {summary['ok']} ok, "
        f"{summary['skipped']} skipped, {summary['failed']} failed"
    )
    print("Results:", output_dir)

    if args.zip and summary["ok"] > 0:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        zip_base = output_dir.parent / f"unlimited_ocr_output_{stamp}"
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(output_dir))
        print("ZIP:", zip_path)

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
