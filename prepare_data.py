"""Τρέχει ΜΙΑ φορά τοπικά: κατεβάζει το glossAPI/amna-press και κρατά μόνο τις
στήλες που χρειάζεται η εφαρμογή (χωρίς το text_markdown, που είναι ~1.4 GB).
Αποτέλεσμα: data/amna_slim.parquet (μερικές δεκάδες MB) -> πάει στο GitHub."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

COLS = ["id", "category", "title", "created", "tags", "url"]
OUT = Path("data/amna_slim.parquet")


def main():
    local = snapshot_download(
        repo_id="glossAPI/amna-press",
        repo_type="dataset",
        allow_patterns=["*.parquet"],
    )
    files = sorted(Path(local).rglob("*.parquet"))
    print("Βρέθηκαν αρχεία:", [f.name for f in files])

    tables = []
    for f in files:
        schema_cols = pq.ParquetFile(f).schema_arrow.names
        cols = [c for c in COLS if c in schema_cols]
        print(f.name, "-> στήλες:", cols)
        tables.append(pq.read_table(f, columns=cols))
    table = pa.concat_tables(tables, promote_options="default")

    df = table.to_pandas()
    df = df[df["created"].notna()]
    df = df[df["created"] >= "2016-01-01"]          # μόνο 2016-2026
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"OK: {len(df):,} άρθρα -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
