#@title 8. Preview the Markdown
from IPython.display import Markdown, display

md_files = sorted(OUTPUT_DIR.glob("*.md"))
if not md_files:
    print("No Markdown files yet.")
for md_file in md_files:
    print("\n" + "=" * 72)
    print(md_file)
    print("=" * 72)
    text = md_file.read_text(encoding="utf-8", errors="replace")
    display(Markdown(text[:6000]))
    if len(text) > 6000:
        print(f"... preview truncated; open {md_file.name} for the full text.")
