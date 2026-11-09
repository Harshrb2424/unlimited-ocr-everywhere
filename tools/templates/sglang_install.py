#@title 1. Install uv, the customized SGLang wheel, and client helpers
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], **kwargs)


if not shutil.which("uv"):
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])

UV = shutil.which("uv")
if not UV:
    candidate = Path(sys.executable).parent / ("uv.exe" if os.name == "nt" else "uv")
    if candidate.exists():
        UV = str(candidate)
if not UV:
    raise RuntimeError(
        "uv was installed but its executable was not found. Install uv manually: "
        "https://docs.astral.sh/uv/getting-started/installation/"
    )

VENV = Path.cwd() / ".venv-uocr"
PY = VENV / "bin" / "python"
if not PY.exists():
    run([UV, "venv", "--python", "3.10", str(VENV)])

WHEEL = Path("wheel") / "sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl"
WHEEL.parent.mkdir(parents=True, exist_ok=True)
if not WHEEL.exists():
    WHEEL_URL = (
        "https://github.com/baidu/Unlimited-OCR/raw/main/wheel/"
        "sglang-0.5.20.dev13%2Bg75b9b06b02-py3-none-any.whl"
    )
    run(["curl", "-L", "--fail", "-o", str(WHEEL), WHEEL_URL])

run([UV, "pip", "install", "--python", str(PY), str(WHEEL)])
run(
    [
        UV,
        "pip",
        "install",
        "--python",
        str(PY),
        "pymupdf==1.27.2.2",
        "fastokens==0.3.2",
        "requests",
    ]
)

# The notebook kernel only needs the client-side helpers.
run([sys.executable, "-m", "pip", "install", "-q", "requests", "pymupdf==1.27.2.2", "pillow"])

print("SGLang venv:", VENV)
print(subprocess.check_output([str(PY), "--version"], text=True).strip())
