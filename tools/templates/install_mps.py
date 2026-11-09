#@title 1. Install dependencies (first run only)
# Pinned to the versions the model's remote code targets.
import importlib.util
import subprocess
import sys


def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


# The standard macOS PyTorch wheel includes the MPS backend.
if importlib.util.find_spec("torch") is None:
    pip_install("torch", "torchvision")

pip_install(
    "transformers==4.52.1",
    "huggingface-hub==0.36.0",
    "tokenizers==0.21.4",
    "accelerate==1.6.0",
    "addict==2.4.0",
    "einops==0.8.2",
    "easydict==1.13",
    "pymupdf==1.27.2.2",
    "pillow",
    "sentencepiece",
)

print("Python:", sys.version.split()[0])
print("Dependencies installed.")
print("If pip replaced transformers after it was already imported, restart the kernel once.")
