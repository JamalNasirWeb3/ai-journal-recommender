#!/usr/bin/env python3
"""Sprint 4: NLP matching engine.

Run as a script to pre-compute and cache all journal representations:
  python nlp.py

Cached artefacts (all in project root):
  journal_embeddings.npz  -- sentence-transformer embeddings, float32 (N x 384)
  tfidf_vectorizer.pkl    -- fitted sklearn TfidfVectorizer
  tfidf_matrix.npz        -- sparse TF-IDF matrix (N x vocab)

When imported, the public API is:
  precompute(df)                     -- offline: build and save all artefacts
  match(title, area, abstract, df)   -- online: return top-K rows with nlp_score + confidence
  load_model()                       -- load SentenceTransformer (cached after first call)

3-level matching strategy:
  L1 (30%) -- TF-IDF cosine similarity over journal keywords + subjects
  L2 (10%) -- LCC-code overlap with user's topic area (soft boost)
  L3 (60%) -- sentence-transformer cosine similarity (all-MiniLM-L6-v2)

Confidence badge (based on L3 similarity):
  High   >= 0.35
  Medium >= 0.20
  Low     < 0.20
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

INPUT_PARQUET = Path("journals_scored.parquet")
EMBEDDINGS_FILE = Path("journal_embeddings.npz")
TFIDF_VEC_FILE = Path("tfidf_vectorizer.pkl")
TFIDF_MAT_FILE = Path("tfidf_matrix.npz")

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH = 256

_model_cache: "SentenceTransformer | None" = None

# ---------------------------------------------------------------------------
# LCC topic-area mapping
# ---------------------------------------------------------------------------

# Maps lowercase topic-area keywords -> LCC top-level prefixes
TOPIC_LCC: dict[str, list[str]] = {
    # Medicine & Health
    "medicine":            ["R", "RC", "RA", "RM", "RS"],
    "medical":             ["R", "RC", "RA"],
    "public health":       ["RA"],
    "epidemiology":        ["RA"],
    "pharmacology":        ["RM", "RS"],
    "pharmacy":            ["RS"],
    "nursing":             ["RT"],
    "dentistry":           ["RK"],
    "surgery":             ["RD"],
    "oncology":            ["RC"],
    "cardiology":          ["RC"],
    "psychiatry":          ["RC"],
    "neuroscience":        ["RC", "QP"],
    "neurology":           ["RC"],
    "pediatrics":          ["RJ"],
    "radiology":           ["RC", "RM"],
    # Life sciences
    "biology":             ["QH", "QK", "QL", "QM", "QP", "QR"],
    "biochemistry":        ["QD", "QP"],
    "genetics":            ["QH"],
    "genomics":            ["QH"],
    "microbiology":        ["QR"],
    "ecology":             ["QH", "GE"],
    "botany":              ["QK"],
    "zoology":             ["QL"],
    "physiology":          ["QP"],
    "immunology":          ["QR"],
    # Physical sciences
    "chemistry":           ["QD", "TP"],
    "physics":             ["QC"],
    "astronomy":           ["QB"],
    "geology":             ["QE"],
    "geoscience":          ["QE", "GB"],
    "oceanography":        ["GC"],
    "meteorology":         ["QC"],
    "climatology":         ["QC", "GE"],
    "mathematics":         ["QA"],
    "statistics":          ["QA"],
    # Computer science & Engineering
    "computer science":    ["QA", "TK"],
    "computing":           ["QA", "TK"],
    "artificial intelligence": ["QA", "TK"],
    "machine learning":    ["QA", "TK"],
    "data science":        ["QA", "HA"],
    "information technology": ["TK", "QA"],
    "engineering":         ["T", "TA", "TK", "TP", "TS"],
    "electrical engineering": ["TK"],
    "civil engineering":   ["TA"],
    "mechanical engineering": ["TJ"],
    "chemical engineering":["TP"],
    "materials science":   ["TA", "TN"],
    "environmental engineering": ["TD", "GE"],
    "biotechnology":       ["TP", "QH"],
    # Social sciences
    "economics":           ["HB", "HC", "HD", "HF", "HG", "HJ"],
    "finance":             ["HG"],
    "accounting":          ["HF"],
    "business":            ["HF", "HD"],
    "management":          ["HD"],
    "marketing":           ["HF"],
    "sociology":           ["HM", "HN", "HQ"],
    "anthropology":        ["GN"],
    "psychology":          ["BF"],
    "political science":   ["J", "JC", "JK"],
    "international relations": ["JX", "JZ"],
    "public administration": ["JK", "JS"],
    "law":                 ["K", "KF", "KD", "KE"],
    "geography":           ["G", "GB", "GE", "GF"],
    "urban":               ["HT", "NA"],
    "demography":          ["HB"],
    "communication":       ["P", "PN"],
    "media":               ["P", "PN"],
    "information science": ["Z"],
    "library science":     ["Z"],
    # Humanities
    "linguistics":         ["P", "PE", "PQ"],
    "language":            ["P", "PE"],
    "literature":          ["P", "PR", "PS", "PQ"],
    "history":             ["D", "DA", "DC", "DK", "DS"],
    "archaeology":         ["CC", "GN"],
    "philosophy":          ["B", "BC", "BD"],
    "ethics":              ["BJ"],
    "religion":            ["BL", "BR", "BS"],
    "theology":            ["BT", "BV", "BX"],
    "art":                 ["N", "NA", "ND", "NK"],
    "architecture":        ["NA"],
    "music":               ["M", "ML", "MT"],
    # Agriculture & Environment
    "agriculture":         ["S", "SB", "SD", "SF"],
    "food science":        ["TX", "TP"],
    "nutrition":           ["TX", "RM"],
    "forestry":            ["SD"],
    "veterinary":          ["SF"],
    "fisheries":           ["SH"],
    "environmental science": ["GE", "TD", "QH"],
    "sustainability":      ["GE", "HC"],
    "energy":              ["TK", "HD"],
    # Education
    "education":           ["L", "LA", "LB", "LC"],
    "pedagogy":            ["LB"],
    "sport":               ["GV"],
    "physical education":  ["GV"],
    # General / catch-all
    "science":             ["Q"],
    "technology":          ["T"],
    "social science":      ["H"],
    "humanities":          ["B", "D", "N", "P"],
}


def _topic_to_lcc_prefixes(area: str) -> set[str]:
    """Map a free-text topic area to a set of LCC top-level prefixes."""
    area_lower = area.lower().strip()
    prefixes: set[str] = set()
    for topic, codes in TOPIC_LCC.items():
        if topic in area_lower or area_lower in topic:
            prefixes.update(codes)
    # Fallback: treat the area itself as a prefix if it looks like one
    if not prefixes:
        m = re.match(r"^([A-Z]{1,3})\d*$", area.strip())
        if m:
            prefixes.add(m.group(1))
    return prefixes


def _extract_lcc_prefixes(lcc_val) -> frozenset[str]:
    """Extract top-level LCC letter prefixes from a journal's lcc_codes value."""
    if not lcc_val or pd.isna(lcc_val):
        return frozenset()
    prefixes = set()
    for part in re.split(r"[|;,]", str(lcc_val)):
        m = re.match(r"([A-Z]+)", part.strip())
        if m:
            prefixes.add(m.group(1))
    return frozenset(prefixes)


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def _scope_text(row: pd.Series) -> str:
    """Build a journal's scope text for embedding (used offline)."""
    parts = []
    if pd.notna(row.get("title")):
        parts.append(str(row["title"]))
    if pd.notna(row.get("subjects")):
        # Strip LCC-style prefixes, keep descriptive part after colon
        subj = re.sub(r"\b[A-Z]{1,4}\d*(?:-\d+)?\b", "", str(row["subjects"]))
        parts.append(subj.strip(" |;,"))
    if pd.notna(row.get("keywords")):
        parts.append(str(row["keywords"]))
    return ". ".join(filter(None, parts))


def _tfidf_text(row: pd.Series) -> str:
    """Build a journal's TF-IDF text (keywords + subjects, no title bias)."""
    parts = []
    if pd.notna(row.get("keywords")):
        parts.append(str(row["keywords"]))
    if pd.notna(row.get("subjects")):
        parts.append(str(row["subjects"]))
    return " ".join(parts)


def _query_text(title: str, area: str | None, abstract: str | None) -> str:
    """Build the query string from user input."""
    parts = [title.strip()]
    if area:
        parts.append(area.strip())
    if abstract:
        parts.append(abstract.strip()[:600])  # cap abstract length
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> "SentenceTransformer":
    """Load (and cache) the sentence-transformer model."""
    global _model_cache
    if _model_cache is None:
        if not HAS_ST:
            raise ImportError(
                "sentence-transformers is required.  "
                "Run: pip install sentence-transformers"
            )
        print(f"  Loading model {EMBED_MODEL} ...")
        _model_cache = SentenceTransformer(EMBED_MODEL)
    return _model_cache


# ---------------------------------------------------------------------------
# Offline pre-computation
# ---------------------------------------------------------------------------

def precompute(df: pd.DataFrame) -> None:
    """Build and save all NLP artefacts from a scored journals DataFrame."""
    if not HAS_ST:
        print("ERROR: sentence-transformers not installed.  pip install sentence-transformers")
        return

    n = len(df)

    # --- TF-IDF ---
    print(f"  Building TF-IDF over {n:,} journals ...")
    tfidf_texts = [_tfidf_text(row) for _, row in df.iterrows()]
    vec = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-z][a-z0-9\-]{1,}\b",
    )
    mat = vec.fit_transform(tfidf_texts)

    with open(TFIDF_VEC_FILE, "wb") as f:
        pickle.dump(vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_npz(TFIDF_MAT_FILE, mat)
    print(f"    -> TF-IDF matrix {mat.shape}, vocab {len(vec.vocabulary_):,}")

    # --- Sentence-transformer embeddings ---
    print(f"  Computing embeddings ({EMBED_MODEL}, batch={EMBED_BATCH}) ...")
    model = load_model()
    scope_texts = [_scope_text(row) for _, row in df.iterrows()]
    embeddings = model.encode(
        scope_texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # pre-normalize for fast cosine via dot product
    )
    embeddings = embeddings.astype(np.float32)

    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings,
        indices=df.index.values.astype(np.int32),
    )
    size_mb = EMBEDDINGS_FILE.stat().st_size / 1_048_576
    print(f"    -> Embeddings {embeddings.shape}, saved {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Runtime matching
# ---------------------------------------------------------------------------

def _load_artefacts() -> tuple[np.ndarray, np.ndarray, TfidfVectorizer, object]:
    """Load pre-computed artefacts. Raises FileNotFoundError if missing."""
    for path in (EMBEDDINGS_FILE, TFIDF_VEC_FILE, TFIDF_MAT_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python nlp.py' first to pre-compute artefacts."
            )
    data = np.load(EMBEDDINGS_FILE)
    embeddings = data["embeddings"]   # float32, pre-normalised
    indices = data["indices"]
    with open(TFIDF_VEC_FILE, "rb") as f:
        vec = pickle.load(f)
    mat = load_npz(TFIDF_MAT_FILE)
    return embeddings, indices, vec, mat


def match(
    title: str,
    area: str | None = None,
    abstract: str | None = None,
    df: pd.DataFrame | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Match user input against journals using 3-level NLP strategy.

    Returns a copy of the top_k rows from df, sorted descending, with added columns:
      nlp_score   float  combined NLP relevance (0-1)
      l3_sim      float  raw L3 cosine similarity
      confidence  str    High / Medium / Low
    """
    if df is None:
        df = pd.read_parquet(INPUT_PARQUET)

    embeddings, indices, vec, tfidf_mat = _load_artefacts()
    query = _query_text(title, area, abstract)

    # Map original df index -> position in the precomputed artefact arrays.
    # Artefacts were built from the full parquet (N=22890); df may be a filtered
    # subset, so we must slice l1/l3 down to only the rows present in df.
    idx_to_pos = {int(orig): pos for pos, orig in enumerate(indices)}
    df_positions = np.array([idx_to_pos[i] for i in df.index], dtype=np.int64)

    # -- L1: TF-IDF cosine similarity (full matrix, then slice) --
    q_tfidf = vec.transform([query])
    l1_all = sklearn_cos(q_tfidf, tfidf_mat).ravel().astype(np.float32)
    l1_scores = np.clip(l1_all[df_positions], 0.0, 1.0)

    # -- L2: LCC topic-area boost (already aligned to df rows) --
    l2_scores = np.zeros(len(df), dtype=np.float32)
    if area:
        query_prefixes = _topic_to_lcc_prefixes(area)
        if query_prefixes:
            journal_prefixes = [_extract_lcc_prefixes(v) for v in df["lcc_codes"]]
            for i, jp in enumerate(journal_prefixes):
                if jp:
                    overlap = len(jp & query_prefixes)
                    l2_scores[i] = min(overlap / len(query_prefixes), 1.0)

    # -- L3: sentence-transformer embedding similarity (full matrix, then slice) --
    model = load_model()
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    l3_all = (embeddings @ q_emb.T).ravel()
    l3_scores = np.clip(l3_all[df_positions], 0.0, 1.0)

    # -- Combine (all three arrays are now len(df)) --
    w_l1 = 0.30
    w_l2 = 0.10 if area else 0.0
    w_l3 = 1.0 - w_l1 - w_l2
    nlp_scores = (l1_scores * w_l1 + l2_scores * w_l2 + l3_scores * w_l3).clip(0.0, 1.0)

    # Build result; exclude predatory journals (final_score == 0.0)
    result = df.copy()
    result["nlp_score"] = nlp_scores
    result["l3_sim"] = l3_scores
    result = result[result["final_score"] > 0.0].copy()

    result["confidence"] = pd.cut(
        result["l3_sim"],
        bins=[-np.inf, 0.20, 0.35, np.inf],
        labels=["Low", "Medium", "High"],
    )

    result = result.nlargest(top_k, "nlp_score")
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HAS_ST:
        print("ERROR: sentence-transformers not installed.")
        print("       Run: pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading {INPUT_PARQUET} ...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  {len(df):,} journals")

    print("\nPre-computing NLP artefacts ...")
    precompute(df)

    print("\nArtefacts written:")
    for path in (EMBEDDINGS_FILE, TFIDF_VEC_FILE, TFIDF_MAT_FILE):
        size = path.stat().st_size / 1_048_576
        print(f"  {path}  ({size:.1f} MB)")

    # Quick smoke test
    print("\nSmoke test — 'deep learning applications in medical imaging':")
    results = match(
        title="Deep learning applications in medical imaging",
        area="machine learning",
        df=df,
        top_k=5,
    )
    display_cols = ["rank", "title", "sjr_quartile", "apc_usd", "nlp_score", "confidence"]
    display_cols = [c for c in display_cols if c in results.columns]
    # Encode for Windows terminal
    out = results[display_cols].to_string(index=False)
    print(out.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))


if __name__ == "__main__":
    main()
