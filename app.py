import re
import unicodedata
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ατζέντα Ειδήσεων ανά Κατηγορία", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "amna_slim.parquet"
STOP = set("""και να το τα την της του των τον στη στο στην στον στα στις στους σε με για από ως
θα που ότι ο η οι ένα μια ένας μιας ενός είναι έχει δεν αλλά ή ανά μετά προς κατά υπό
παρά μέσω όπως μετά σχετικά νέα νέο νέος νέες""".split())


# ------------------------------------------------------------------ βοηθητικά
def norm(s: str) -> str:
    """πεζά, χωρίς τόνους, ς->σ  (ώστε 'εκλογές' ~ 'ΕΚΛΟΓΕΣ' ~ 'εκλογη')"""
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("ς", "σ")


def stem_word(w: str) -> str:
    """Απλό στέλεχος: κόβει έως 2 καταλήξεις (α,ε,η,ι,ο,υ,ω,σ,ν) αν η λέξη μένει >=5 γράμματα.
    πυρκαγιά/πυρκαγιές -> πυρκαγ · εκλογές -> εκλογ · οικονομία -> οικονομ"""
    for _ in range(2):
        if len(w) > 5 and w[-1] in "αεηιουωσν":
            w = w[:-1]
    return w


@st.cache_data(show_spinner="Φόρτωση δεδομένων…")
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["created"])
    df["month"] = df["created"].dt.to_period("M").dt.to_timestamp()
    tags = df["tags"].fillna("") if "tags" in df else ""
    df["key"] = (df["title"].fillna("") + " " + tags).map(norm)
    return df


def monthly_table(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cats) -> pd.DataFrame:
    m = d.groupby(["month", "category"]).size().unstack(fill_value=0)
    idx = pd.date_range(start, end, freq="MS")
    return m.reindex(index=idx, columns=cats, fill_value=0)


def find_peaks(s: pd.Series, k: float = 2.0, min_count: int = 3):
    """Μήνας-κορυφή = τοπικό μέγιστο με τιμή >= μέσος όρος + k·τυπική απόκλιση
    (και τουλάχιστον min_count άρθρα)."""
    v = s.values.astype(float)
    if len(v) < 3 or v.sum() == 0:
        return []
    thr = max(v.mean() + k * v.std(), min_count)
    out = []
    for i in range(len(v)):
        left = i == 0 or v[i] >= v[i - 1]
        right = i == len(v) - 1 or v[i] > v[i + 1]
        if v[i] >= thr and left and right:
            out.append(s.index[i])
    return out


def sample_titles(d: pd.DataFrame, month, per_cat: int = 6) -> pd.DataFrame:
    x = d[d["month"] == month]
    parts = [g.sample(min(len(g), per_cat), random_state=0) for _, g in x.groupby("category")]
    return pd.concat(parts) if parts else x


def keyword_summary(titles: pd.DataFrame, term_norm: str) -> str:
    """Εφεδρική σύνοψη χωρίς LLM: συχνότερες λέξεις ανά κατηγορία."""
    lines = []
    for cat, g in titles.groupby("category"):
        words = re.findall(r"\w{4,}", " ".join(g["title"]).lower())
        words = [w for w in words if w not in STOP and term_norm not in norm(w)]
        top = pd.Series(words).value_counts().head(5).index.tolist()
        lines.append(f"- **{cat}**: " + (", ".join(top) if top else "—"))
    return "\n".join(lines)


def llm_summary(prompt: str, api_key: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=700, messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def build_prompt(term, month, counts: pd.Series, titles: pd.DataFrame) -> str:
    cnt = ", ".join(f"{c}: {int(n)}" for c, n in counts.items())
    blocks = []
    for cat, g in titles.groupby("category"):
        blocks.append(f"[{cat}]\n" + "\n".join(f"- {t}" for t in g["title"]))
    return (
        f"Αναλύεις την ειδησεογραφία του ΑΠΕ-ΜΠΕ για τον όρο «{term}».\n"
        f"Τον μήνα {month:%m/%Y} παρατηρείται κορύφωση. Άρθρα ανά κατηγορία: {cnt}.\n\n"
        "Δείγμα τίτλων ανά κατηγορία:\n\n" + "\n\n".join(blocks) + "\n\n"
        "Γράψε σύνοψη 4-6 προτάσεων στα ελληνικά: (1) ποιο γεγονός/γεγονότα εξηγούν "
        "την κορύφωση, (2) πώς διαφέρει η προσέγγιση κάθε κατηγορίας, (3) ποια κατηγορία "
        "οδηγεί την ατζέντα. Βασίσου ΜΟΝΟ στους τίτλους· αν δεν προκύπτει σαφές γεγονός, "
        "πες το ρητά."
    )


def get_secret_key() -> str:
    """Διαβάζει το key από τα secrets· αν δεν υπάρχει secrets.toml, επιστρέφει κενό."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------ UI
st.title("📰 Η Ατζέντα των Ειδήσεων ανά Κατηγορία")
st.caption("Dataset: glossAPI/amna-press (ΑΠΕ-ΜΠΕ) · αναζήτηση στους τίτλους και στα tags")

if not DATA_PATH.exists():
    st.error("Λείπει το data/amna_slim.parquet — τρέξε πρώτα `python prepare_data.py`.")
    st.stop()

df = load_data()
all_cats = sorted(df["category"].dropna().unique())

with st.sidebar:
    st.header("Ρυθμίσεις")
    y_min, y_max = int(df["created"].dt.year.min()), int(df["created"].dt.year.max())
    years = st.slider("Έτη", y_min, y_max, (max(y_min, 2016), y_max))
    k = st.slider("Ευαισθησία κορυφώσεων (k·σ πάνω από τον μέσο)", 0.5, 4.0, 2.0, 0.5)
    min_count = st.number_input("Ελάχιστα άρθρα για κορύφωση", 1, 100, 3)
    max_peaks = st.number_input("Μέγιστες κορυφώσεις για σύνοψη", 1, 15, 5)
    cats = st.multiselect("Κατηγορίες", all_cats, default=all_cats)
    st.divider()
    st.subheader("LLM (Anthropic)")
    api_key = st.text_input(
        "API key", type="password", value=get_secret_key()
    )
    model = st.text_input("Μοντέλο", value="claude-sonnet-5-5")

if not cats:
    st.warning("Διάλεξε τουλάχιστον μία κατηγορία από το πλαϊνό μενού.")
    st.stop()

term = st.text_input("Όρος αναζήτησης", value="εκλογές",
                     help="Γράψε τον όρο κανονικά (π.χ. πυρκαγιά, εκλογές, πανδημία). "
                          "Αν υπάρχουν πολλές λέξεις, πρέπει να υπάρχουν όλες στον τίτλο.")
use_stem = st.checkbox("Βρες και άλλες πτώσεις/πληθυντικούς (πυρκαγιά → πυρκαγιές, πυρκαγιών)", value=True)

if not term.strip():
    st.stop()

words = [norm(w) for w in term.split() if w.strip()]
stems = [stem_word(w) if use_stem else w for w in words]
tn = stems[0]
start = pd.Timestamp(years[0], 1, 1)
end = pd.Timestamp(years[1], 12, 1)
mask = (df["month"] >= start) & (df["month"] <= end) & df["category"].isin(cats)
for st_ in stems:
    mask &= df["key"].str.contains(re.escape(st_), regex=True)
d = df[mask]

if d.empty:
    st.warning("Δεν βρέθηκαν άρθρα για αυτόν τον όρο.")
    st.stop()

mt = monthly_table(d, start, end, cats)
total = mt.sum(axis=1)
c1, c2, c3 = st.columns(3)
c1.metric("Άρθρα που ταιριάζουν", f"{len(d):,}")
c2.metric("Κατηγορία με τα περισσότερα", mt.sum().idxmax())
c3.metric("Μήνας με τη μεγαλύτερη κίνηση", f"{total.idxmax():%m/%Y}")

# ---- γράφημα 1: στοιβαγμένες στήλες ανά μήνα/κατηγορία
long = mt.reset_index(names="month").melt("month", var_name="Κατηγορία", value_name="Άρθρα")
bars = (
    alt.Chart(long)
    .mark_bar()
    .encode(
        x=alt.X("month:T", title="Μήνας"),
        y=alt.Y("Άρθρα:Q", title="Πλήθος άρθρων"),
        color=alt.Color("Κατηγορία:N"),
        tooltip=[alt.Tooltip("month:T", format="%m/%Y", title="Μήνας"), "Κατηγορία", "Άρθρα"],
    )
    .properties(height=380)
    .interactive(bind_y=False)
)

# ---- κορυφώσεις ανά κατηγορία
cat_peaks = {c: find_peaks(mt[c], k, int(min_count)) for c in cats}
peak_rows = [{"month": m, "Κατηγορία": c, "Άρθρα": int(mt.loc[m, c])}
             for c, ms in cat_peaks.items() for m in ms]
peaks_df = pd.DataFrame(peak_rows)

lines = alt.Chart(long).mark_line().encode(
    x=alt.X("month:T", title="Μήνας"), y=alt.Y("Άρθρα:Q"), color="Κατηγορία:N"
)
layers = lines
if not peaks_df.empty:
    pts = alt.Chart(peaks_df).mark_point(filled=True, size=110).encode(
        x="month:T", y="Άρθρα:Q", color="Κατηγορία:N",
        tooltip=[alt.Tooltip("month:T", format="%m/%Y", title="Κορύφωση"), "Κατηγορία", "Άρθρα"],
    )
    layers = lines + pts

tab1, tab2 = st.tabs(["Συνολική εικόνα (στοιβαγμένο)", "Κορυφώσεις ανά κατηγορία"])
tab1.altair_chart(bars, use_container_width=True)
tab2.altair_chart(layers.properties(height=380).interactive(bind_y=False), use_container_width=True)
cat_rows = [
    {
        "Κατηγορία": c,
        "Σύνολο άρθρων": int(mt[c].sum()),
        "Κορυφώσεις (μήνας: άρθρα)": ", ".join(
            f"{m:%m/%Y}: {int(mt.loc[m, c])}"
            for m in sorted(cat_peaks[c], key=lambda x: mt.loc[x, c], reverse=True)
        ) or "—",
    }
    for c in cats
]
tab2.markdown("**Κορυφώσεις ανά κατηγορία** (τοπικά μέγιστα πάνω από μέσο όρο + k·σ της κάθε κατηγορίας)")
tab2.dataframe(pd.DataFrame(cat_rows), hide_index=True, use_container_width=True)

# ---- πίνακας κορυφώσεων (συνολικά) + συνεισφορά κατηγορίας
st.subheader("Κορυφώσεις του όρου")
peaks = find_peaks(total, k, int(min_count))
if not peaks:
    st.info("Δεν εντοπίστηκαν κορυφώσεις με αυτές τις ρυθμίσεις — μείωσε το k.")
    st.stop()
peaks = sorted(peaks, key=lambda m: total[m], reverse=True)[: int(max_peaks)]
rows = []
for m in peaks:
    r = mt.loc[m]
    rows.append({"Μήνας": f"{m:%m/%Y}", "Σύνολο": int(r.sum()),
                 "Κατηγορία-οδηγός": r.idxmax(), "Μερίδιο %": round(100 * r.max() / r.sum()),
                 **{c: int(r[c]) for c in cats}})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---- σύνοψη «γιατί κορυφώνεται»
st.subheader("Γιατί κορυφώνεται;")
if not api_key:
    st.caption("Χωρίς API key εμφανίζεται απλή σύνοψη λέξεων-κλειδιών· βάλε key στο πλαϊνό μενού για LLM.")

if st.button("Δημιουργία σύνοψης για τις κορυφώσεις"):
    for m in peaks:
        titles = sample_titles(d, m)
        with st.expander(f"{m:%m/%Y} — {int(total[m])} άρθρα", expanded=True):
            if api_key:
                try:
                    st.markdown(llm_summary(build_prompt(term, m, mt.loc[m], titles), api_key, model))
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Σφάλμα LLM ({e}). Εμφανίζεται εφεδρική σύνοψη.")
                    st.markdown(keyword_summary(titles, tn))
            else:
                st.markdown(keyword_summary(titles, tn))
            st.markdown("**Τίτλοι άρθρων (δείγμα):**")
            show = titles[["category", "title", "url"]].rename(
                columns={"category": "Κατηγορία", "title": "Τίτλος", "url": "Σύνδεσμος"})
            st.dataframe(show, hide_index=True, use_container_width=True,
                         column_config={"Σύνδεσμος": st.column_config.LinkColumn()})