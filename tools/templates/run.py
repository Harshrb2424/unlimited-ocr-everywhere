#@title 7. Process the input folder
import time

if WAIT_FOR_FILES:
    print("=" * 72)
    print("WAITING FOR FILES")
    print("Upload PDFs or images to:", INPUT_DIR)
    print("In Colab: left sidebar -> folder icon -> open that folder -> upload")
    print("=" * 72)
    deadline = time.time() + WAIT_MINUTES * 60
    last_note = 0.0
    while not list_inputs() and time.time() < deadline:
        if time.time() - last_note > 30:
            print("  still waiting ...", time.strftime("%H:%M:%S"))
            last_note = time.time()
        time.sleep(5)

files = list_inputs()
if not files:
    print("No files found. Add PDFs/images to", INPUT_DIR, "and re-run this cell.")
else:
    print(f"Found {len(files)} file(s). Starting ...")
    run_all()
    print("\nAdd more files and re-run THIS cell to process them.")
