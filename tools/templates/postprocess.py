#@title 5. Load the post-processor
import importlib.util
import os
import urllib.request

POST_DIR = VENDOR_DIR
POST_DIR.mkdir(parents=True, exist_ok=True)
POST_PATH = POST_DIR / "process_unlimitedOCR2.py"
POST_URL = "https://raw.githubusercontent.com/baidu/Unlimited-OCR/main/utils/process_unlimitedOCR2.py"

if not POST_PATH.exists() or POST_PATH.stat().st_size < 1000:
    print("Downloading process_unlimitedOCR2.py ...")
    try:
        urllib.request.urlretrieve(POST_URL, POST_PATH)
    except Exception as exc:
        print("Download failed, trying wget:", exc)
        os.system(f"wget -q '{POST_URL}' -O '{POST_PATH}'")

if not POST_PATH.exists() or POST_PATH.stat().st_size < 1000:
    raise RuntimeError(
        "Could not download process_unlimitedOCR2.py. Download it manually from "
        f"{POST_URL} and place it at {POST_PATH}."
    )

spec = importlib.util.spec_from_file_location("process_unlimitedOCR2", POST_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
convert = module.convert

_test = convert("<PAGE><BLOCK>text[0,0,100,100]<FORK>ok")
print("Post-processor ready:", _test)
