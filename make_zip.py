"""
Create the final deliverable zip: Supermarket_Sales_Analysis_Colab.zip
Contains:
  - Supermarket_Sales_Analysis.ipynb  (self-contained, upload directly to Colab)
  - The modular project (main.py + src/ + data/ + outputs/) for local use
"""
import zipfile
from pathlib import Path

BASE = Path(".")
ZIP_NAME = "Supermarket_Sales_Analysis_Colab.zip"

# Files/folders to include
INCLUDE = [
    "Supermarket_Sales_Analysis.ipynb",
    "main.py",
    "requirements.txt",
    "README.md",
    "src",
    "data",
    "outputs",
]

EXCLUDE_PATTERNS = [
    ".ipynb_checkpoints",
    "__pycache__",
    "generate_colab_notebook.py",
    "fix_generator.py",
    "validate_nb.py",
    "make_zip.py",
    ".pyc",
    # Exclude large experiment .pt files (only keep best_model.pt)
]

def should_exclude(path: Path) -> bool:
    s = str(path)
    for pat in EXCLUDE_PATTERNS:
        if pat in s:
            return True
    return False

file_count = 0
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for item in INCLUDE:
        p = BASE / item
        if p.is_file():
            if not should_exclude(p):
                zf.write(p, p)
                print(f"  + {p}")
                file_count += 1
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not should_exclude(f):
                    zf.write(f, f)
                    print(f"  + {f}")
                    file_count += 1

size_mb = Path(ZIP_NAME).stat().st_size / (1024 * 1024)
print(f"\nZIP created: {ZIP_NAME}")
print(f"  Files     : {file_count}")
print(f"  Size      : {size_mb:.2f} MB")
print(f"\nTo use in Colab:")
print(f"  1. Upload 'Supermarket_Sales_Analysis.ipynb' to colab.research.google.com")
print(f"  2. Click 'Runtime > Run all'")
print(f"  3. All 14 plots and metrics will appear inline.")
print(f"  4. A 'supermarket_outputs.zip' will be auto-downloaded at the end.")
