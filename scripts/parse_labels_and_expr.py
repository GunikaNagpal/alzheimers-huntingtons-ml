from pathlib import Path
import re
import pandas as pd
import GEOparse
import sys

DATA_DIR = Path("data")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# Prefer the plain TXT first (your good file), then gz
candidates = [
    DATA_DIR / "GSE33000_series_matrix.txt",
    DATA_DIR / "GSE33000_series_matrix.txt.gz",
]

src = next((p for p in candidates if p.exists() and p.is_file()), None)

def load_from(path: Path):
    print(f"Loading with GEOparse from: {path}")
    gse = GEOparse.get_GEO(filepath=str(path))
    print("Samples detected:", len(gse.gsms))
    return gse

def download_fresh():
    print("Local file invalid/missing. Downloading fresh series matrix via GEOparse…")
    gse = GEOparse.get_GEO(geo="GSE33000", destdir=str(DATA_DIR))
    print("Downloaded. Samples detected:", len(gse.gsms))
    return gse

# Load, verify, fallback if needed
if src is not None:
    gse = load_from(src)
    if len(gse.gsms) == 0:
        print("⚠️ Parsed 0 samples — falling back to fresh download.")
        gse = download_fresh()
else:
    gse = download_fresh()

if len(gse.gsms) == 0:
    sys.exit("❌ Still 0 samples. Need a valid series_matrix in data/.")

# --- Robust label extraction across all metadata fields ---
pat_ad   = re.compile(r"\b(alzheimer|alzheimer's|ad)\b", re.I)
pat_hd   = re.compile(r"\b(huntington|huntington's|hd)\b", re.I)
pat_ctrl = re.compile(r"\b(control|healthy|normal|nondemented|non[- ]demented)\b", re.I)

rows = []
for gsm_id, gsm in gse.gsms.items():
    parts = []
    for k, v in gsm.metadata.items():
        if isinstance(v, (list, tuple)):
            parts.extend(map(str, v))
        else:
            parts.append(str(v))
    blob = " | ".join(parts).lower()

    diag = None
    if pat_ad.search(blob): diag = "AD"
    elif pat_hd.search(blob): diag = "HD"
    elif pat_ctrl.search(blob): diag = "Control"

    rows.append({"SampleID": gsm_id, "Diagnosis": diag})

labels = pd.DataFrame(rows).dropna(subset=["Diagnosis"]).reset_index(drop=True)
print("Label counts:\n", labels["Diagnosis"].value_counts())
labels.to_csv(OUT / "labels.csv", index=False)

# Expression matrix (genes x samples)
expr = gse.pivot_samples("VALUE")
expr.to_csv(OUT / "expression.csv")
print("Expr shape:", expr.shape)
print("✅ Saved outputs/labels.csv and outputs/expression.csv")


