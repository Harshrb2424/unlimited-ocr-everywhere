#@title 3. Detect the device and load the model
import gc
import inspect
import os
import platform
import sys
import textwrap

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("SPARSE_DIAG_LOG", "0")
os.environ.setdefault("CODE_STAMP_LOG", "0")
os.environ.setdefault("RECALL_LOG", "0")

import torch


def total_ram_gb():
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Pick the device
# ---------------------------------------------------------------------------
if REQUIRE_DEVICE == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Use Runtime -> Change runtime type -> GPU "
            "(Colab) or install the CUDA-enabled PyTorch wheel (local), then "
            "re-run this notebook."
        )
    DEVICE = "cuda"
elif REQUIRE_DEVICE == "mps":
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not mps_ok:
        print("WARNING: MPS is not available. Falling back to CPU (very slow).")
        DEVICE = "cpu"
    else:
        DEVICE = "mps"
elif REQUIRE_DEVICE == "cpu":
    DEVICE = "cpu"
else:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name} | {vram_gb:.1f} GB | compute {cap[0]}.{cap[1]}")
    if vram_gb < 10:
        raise RuntimeError(
            f"Only {vram_gb:.1f} GB VRAM detected. Unlimited-OCR-2 needs about "
            "12 GB minimum (the BF16 weights alone are ~6.7 GB). Use a T4/L4/A100 "
            "runtime or a larger local GPU."
        )
    if vram_gb < 12:
        print("WARNING: less than 12 GB VRAM; use MODE='base' and CHUNK_PAGES=1.")
elif DEVICE == "mps":
    gpu_name = "Apple Silicon (MPS)"
    vram_gb = 0.0
    print(f"Device: {gpu_name} | unified memory: {total_ram_gb():.1f} GB")
    print("Apple Silicon support is experimental. Close other apps before running.")
else:
    gpu_name = platform.processor() or "CPU"
    vram_gb = 0.0
    ram_gb = total_ram_gb()
    print(f"Device: CPU ({gpu_name}) | system RAM: {ram_gb:.1f} GB")
    if ram_gb and ram_gb < 16:
        print("WARNING: CPU inference needs roughly 12 GB just for float32 weights.")
        print("         Use a 16 GB+ machine, or rent a Colab T4 for free.")
    print("CPU inference is experimental and much slower than a GPU.")

# ---------------------------------------------------------------------------
# Pick the dtype
# ---------------------------------------------------------------------------
if FORCE_DTYPE != "auto":
    DTYPE = getattr(torch, FORCE_DTYPE)
    if DEVICE == "cpu" and DTYPE == torch.float16:
        print("float16 is not supported on CPU; using float32 instead.")
        DTYPE = torch.float32
elif DEVICE == "cuda":
    try:
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        native_bf16 = torch.cuda.is_bf16_supported()
    DTYPE = torch.bfloat16 if native_bf16 else torch.float16
elif DEVICE == "mps":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

print("Using dtype:", str(DTYPE).replace("torch.", ""))

# Triton kernels are CUDA-only.
USE_TRITON = bool(USE_TRITON) and DEVICE == "cuda"

# ---------------------------------------------------------------------------
# Load the tokenizer and model
# ---------------------------------------------------------------------------
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers as _tf

print("transformers:", _tf.__version__)
if _tf.__version__ != "4.52.1":
    raise RuntimeError(
        f"transformers {_tf.__version__} is loaded, but this notebook needs 4.52.1. "
        "Restart the runtime/kernel, then run all cells again."
    )

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading model (~6.7 GB) ...")
load_kwargs = dict(
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=DTYPE,
)
if not USE_TRITON:
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    sparse_cfg = getattr(config, "sparse_attention_config", None)
    if isinstance(sparse_cfg, dict):
        sparse_cfg["dash_use_triton"] = False
        print("Triton kernels: off (dense PyTorch fallback)")
    else:
        print("WARNING: could not disable Triton in the model config")
    load_kwargs["config"] = config
else:
    print("Triton kernels: on")

try:
    model = AutoModel.from_pretrained(MODEL_ID, low_cpu_mem_usage=True, **load_kwargs)
except Exception as exc:
    print("low_cpu_mem_usage failed, retrying without it:", exc)
    gc.collect()
    model = AutoModel.from_pretrained(MODEL_ID, **load_kwargs)

model = model.eval().to(DEVICE)
torch.set_grad_enabled(False)
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
if DEVICE == "cuda":
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Compatibility patches for float16/float32 and CPU/MPS
# ---------------------------------------------------------------------------
def patch_image_dtype(dtype):
    """The upstream preprocessing builds bfloat16 image tensors. Cast them."""
    patched = 0
    wanted = {type(model).__module__}
    try:
        wanted.add(type(model.model).__module__)
    except Exception:
        pass
    for name, module in list(sys.modules.items()):
        if not (name.endswith("modeling_unlimitedocr") or name in wanted):
            continue
        original = getattr(module, "_uocr_build_inputs", None)
        if original is None or getattr(original, "_uocr_dtype_patched", False):
            continue

        def wrapper(*args, __orig=original, __dtype=dtype, **kwargs):
            input_ids, images, seq_mask, spatial = __orig(*args, **kwargs)
            if images is not None:
                images = (images[0].to(__dtype), images[1].to(__dtype))
            return input_ids, images, seq_mask, spatial

        wrapper._uocr_dtype_patched = True
        module._uocr_build_inputs = wrapper
        patched += 1
    return patched


def patch_non_cuda_forward(model):
    """Upstream forward has hard-coded .cuda() calls. Rewrite them for CPU/MPS."""
    module = sys.modules.get(type(model).__module__)
    if module is None:
        return []
    patched = []
    for name, cls in list(vars(module).items()):
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
            exec(compile(new_src, f"<patched {name}.forward>", "exec"), namespace)
            cls.forward = namespace["forward"]
            patched.append(name)
        except Exception as exc:
            print(f"  could not patch {name}.forward: {exc}")
    return patched


if DTYPE != torch.bfloat16:
    n = patch_image_dtype(DTYPE)
    print(f"Patched image preprocessing for {str(DTYPE).replace('torch.', '')}: {n} module(s).")

if DEVICE != "cuda":
    patched = patch_non_cuda_forward(model)
    print("Patched forward for", DEVICE + ":", ", ".join(patched) or "no .cuda() calls found")
    try:
        def _cuda_shim(self, *args, **kwargs):
            return self.to(DEVICE)

        torch.Tensor.cuda = _cuda_shim
        print("Applied torch.Tensor.cuda compatibility shim.")
    except Exception as exc:
        print("Could not apply Tensor.cuda shim (source patch should cover it):", exc)


# ---------------------------------------------------------------------------
# Pick a safe number of pages per request
# ---------------------------------------------------------------------------
if CHUNK_PAGES > 0:
    CHUNK = CHUNK_PAGES
elif DEVICE != "cuda":
    CHUNK = 1
elif vram_gb >= 40:
    CHUNK = 0          # whole document in one request
elif vram_gb >= 22:
    CHUNK = 8
elif vram_gb >= 15:
    CHUNK = 2
else:
    CHUNK = 1

print("Pages per request:", "whole document" if CHUNK == 0 else CHUNK)
print("Ready.")
