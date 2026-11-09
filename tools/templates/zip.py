#@title 9. Zip the results
import datetime
from pathlib import Path

if not any(OUTPUT_DIR.glob("*.md")):
    print("Nothing to zip yet.")
elif MAKE_ZIP or AUTO_DOWNLOAD_ZIP:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_base = Path(ROOT) / f"unlimited_ocr_output_{stamp}"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(OUTPUT_DIR))
    print("ZIP:", zip_path)
    if AUTO_DOWNLOAD_ZIP:
        try:
            from google.colab import files

            files.download(zip_path)
        except Exception as exc:
            print("Automatic download failed; download the ZIP manually:", exc)
else:
    print("ZIP disabled (MAKE_ZIP = False).")
