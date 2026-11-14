# Ατζέντα Ειδήσεων ανά Κατηγορία (ΑΠΕ-ΜΠΕ, 2016–2026)

Streamlit εφαρμογή: γράφεις όρο → γράφημα άρθρων ανά μήνα/κατηγορία → κορυφώσεις →
σύνοψη «γιατί κορυφώνεται» με LLM. Dataset: `glossAPI/amna-press`.

## Εκτέλεση τοπικά
```bash
pip install -r requirements.txt
python prepare_data.py          # 1 φορά· κατεβάζει ~1.5 GB, παράγει data/amna_slim.parquet
streamlit run app.py
```

## GitHub + Streamlit Community Cloud
1. `git init && git add . && git commit -m "init" ` και push σε νέο repo (το `data/amna_slim.parquet` πρέπει να ανέβει· <100 MB).
2. share.streamlit.io → New app → διάλεξε repo, branch, `app.py`.
3. Settings → Secrets: `ANTHROPIC_API_KEY = "sk-ant-..."`
