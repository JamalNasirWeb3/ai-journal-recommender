#!/usr/bin/env python3
"""Sprint 2: Enrich journals_clean.parquet with external data sources.

Inputs:
  journals_clean.parquet          — Sprint 1 output (required)
  scimago*.csv / sjr*.csv         — SCImago SJR export, semicolon-delimited
                                    (optional; download from scimagojr.com/journalrank.php)

New columns added to journals_enriched.parquet:
  sjr_score         Float64   SCImago SJR impact score
  sjr_quartile      object    Q1 / Q2 / Q3 / Q4, or None
  h_index           Int64     SCImago H-index
  openalex_id       object    OpenAlex source URL
  articles_per_year Float64   3-year average articles/year (OpenAlex)
  is_in_scopus      boolean   journal has a Scopus ID in OpenAlex
  is_predatory      boolean   publisher or ISSN appears in Beall's List

Cache:
  .openalex_cache.json  — persists OpenAlex lookups between runs;
                          delete this file to force a full re-fetch
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

INPUT_PARQUET = Path("journals_clean.parquet")
OUTPUT_PARQUET = Path("journals_enriched.parquet")
OPENALEX_CACHE = Path(".openalex_cache.json")

OPENALEX_EMAIL = "jfk24572@gmail.com"
OPENALEX_BATCH = 50        # ISSNs per API request
OPENALEX_DELAY = 0.25      # seconds between requests (~4 req/s; polite pool allows 10/s)

BEALLS_PUBLISHERS_URL = (
    "https://raw.githubusercontent.com/stop-predatory-journals/"
    "stop-predatory-journals.github.io/master/_data/publishers.csv"
)
BEALLS_JOURNALS_URL = (
    "https://raw.githubusercontent.com/stop-predatory-journals/"
    "stop-predatory-journals.github.io/master/_data/journals.csv"
)


# ---------------------------------------------------------------------------
# ISSN helpers
# ---------------------------------------------------------------------------

def normalize_issn(val) -> str | None:
    """Return ISSN normalized to XXXX-XXXX, or None if unparseable."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    digits = re.sub(r"[^0-9Xx]", "", str(val)).upper()
    return f"{digits[:4]}-{digits[4:]}" if len(digits) == 8 else None


def issn_candidates(row: pd.Series) -> list[str]:
    """Return normalized ISSNs for a row, print ISSN first, deduped."""
    seen, out = set(), []
    for col in ("issn_print", "issn_online"):
        n = normalize_issn(row.get(col))
        if n and n not in seen:
            out.append(n)
            seen.add(n)
    return out


# ---------------------------------------------------------------------------
# SCImago SJR
# ---------------------------------------------------------------------------

def find_scimago_file() -> Path | None:
    """Locate a local SCImago CSV by common naming patterns."""
    for pat in ("scimago*.csv", "sjr*.csv", "SJR*.csv", "scimagojr*.csv"):
        matches = sorted(Path(".").glob(pat))
        if matches:
            return matches[-1]
    return None


def load_scimago(path: Path) -> dict[str, dict]:
    """Load SCImago SJR CSV and return {normalized_issn: {sjr_score, sjr_quartile, h_index}}.

    SCImago exports use semicolon delimiters and may use commas as decimal
    separators for SJR scores (European locale).
    """
    print(f"  Loading {path} ...")
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        sjr_str = str(row.get("sjr") or "").replace(",", ".").strip()
        try:
            sjr_score: float | None = float(sjr_str) if sjr_str else None
        except ValueError:
            sjr_score = None

        quartile = str(row.get("sjr best quartile") or "").strip()
        quartile = quartile if quartile in {"Q1", "Q2", "Q3", "Q4"} else None

        try:
            h_raw = int(row.get("h index") or 0)
            h_index: int | None = h_raw if h_raw > 0 else None
        except (ValueError, TypeError):
            h_index = None

        record = {"sjr_score": sjr_score, "sjr_quartile": quartile, "h_index": h_index}
        for raw_issn in str(row.get("issn") or "").split(","):
            n = normalize_issn(raw_issn.strip())
            if n:
                result.setdefault(n, record)

    print(f"    -> {len(result):,} ISSN entries loaded")
    return result


def enrich_scimago(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    sjr_scores, quartiles, h_indices = [], [], []
    for _, row in df.iterrows():
        rec = next((lookup[i] for i in issn_candidates(row) if i in lookup), None)
        sjr_scores.append(rec["sjr_score"] if rec else None)
        quartiles.append(rec["sjr_quartile"] if rec else None)
        h_indices.append(rec["h_index"] if rec else None)

    df["sjr_score"] = pd.array(sjr_scores, dtype="Float64")
    df["sjr_quartile"] = quartiles
    df["h_index"] = pd.array(h_indices, dtype="Int64")

    print(f"    -> {df['sjr_score'].notna().sum():,} / {len(df):,} journals matched")
    return df


# ---------------------------------------------------------------------------
# OpenAlex API
# ---------------------------------------------------------------------------

def _load_openalex_cache() -> dict:
    if OPENALEX_CACHE.exists():
        return json.loads(OPENALEX_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_openalex_cache(cache: dict) -> None:
    OPENALEX_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _fetch_openalex_batch(issns: list[str]) -> dict[str, dict]:
    """Query OpenAlex Sources API for a batch of ISSNs using OR filter.

    Returns {normalized_issn: record} for every source found.
    """
    url = (
        f"https://api.openalex.org/sources"
        f"?filter=issn:{'|'.join(issns)}"
        f"&per_page=200&mailto={OPENALEX_EMAIL}"
    )
    try:
        resp = _requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"\n    WARNING: OpenAlex request failed: {exc}")
        return {}

    out: dict[str, dict] = {}
    for src in data.get("results") or []:
        counts = src.get("counts_by_year") or []
        recent = sorted(counts, key=lambda x: x["year"], reverse=True)[:3]
        apy: float | None = (
            sum(y["works_count"] for y in recent) / len(recent) if recent else None
        )
        record = {
            "openalex_id": src.get("id"),
            "articles_per_year": apy,
            "is_core": bool(src.get("is_core")),
        }
        for raw_issn in src.get("issn") or []:
            n = normalize_issn(raw_issn)
            if n:
                out.setdefault(n, record)
    return out


def enrich_openalex(df: pd.DataFrame) -> pd.DataFrame:
    if not HAS_REQUESTS:
        print("  Skipping - install 'requests': pip install requests")
        df["openalex_id"] = pd.NA
        df["articles_per_year"] = pd.array([None] * len(df), dtype="Float64")
        df["is_core"] = pd.array([pd.NA] * len(df), dtype="boolean")
        return df

    # Collect all unique ISSNs across the dataset
    seen: set[str] = set()
    all_issns: list[str] = []
    for _, row in df.iterrows():
        for issn in issn_candidates(row):
            if issn not in seen:
                all_issns.append(issn)
                seen.add(issn)

    cache = _load_openalex_cache()
    to_fetch = [i for i in all_issns if i not in cache]
    batches = [to_fetch[x: x + OPENALEX_BATCH] for x in range(0, len(to_fetch), OPENALEX_BATCH)]

    print(
        f"  {len(all_issns):,} unique ISSNs | "
        f"{len(cache):,} cached | "
        f"{len(to_fetch):,} to fetch ({len(batches)} batches)"
    )

    for idx, batch in enumerate(batches, 1):
        results = _fetch_openalex_batch(batch)
        for issn in batch:
            cache[issn] = results.get(issn)  # None = queried but not found
        if idx % 25 == 0 or idx == len(batches):
            print(f"    {idx}/{len(batches)} batches …", end="\r", flush=True)
        time.sleep(OPENALEX_DELAY)

    if batches:
        _save_openalex_cache(cache)
        print(f"\n    Cache saved -> {OPENALEX_CACHE}  ({len(cache):,} entries)")

    oa_ids, apys, core_flags = [], [], []
    for _, row in df.iterrows():
        rec = next((cache[i] for i in issn_candidates(row) if cache.get(i)), None)
        oa_ids.append(rec["openalex_id"] if rec else None)
        apys.append(rec["articles_per_year"] if rec else None)
        core_flags.append(bool(rec["is_core"]) if rec else None)

    df["openalex_id"] = oa_ids
    df["articles_per_year"] = pd.array(apys, dtype="Float64")
    df["is_core"] = pd.array(
        [pd.NA if v is None else v for v in core_flags],
        dtype="boolean",
    )
    print(f"    -> {df['openalex_id'].notna().sum():,} / {len(df):,} journals matched")
    return df


# ---------------------------------------------------------------------------
# Beall's List
# ---------------------------------------------------------------------------

def _parse_bealls_csv(text: str) -> set[str]:
    """Extract lowercase 'name' values from a Beall's List CSV (url,name,abbr)."""
    names = set()
    for line in text.splitlines()[1:]:  # skip header
        parts = line.split(",", 2)
        if len(parts) >= 2 and parts[1].strip():
            names.add(parts[1].strip().lower())
    return names


def fetch_bealls() -> tuple[re.Pattern | None, re.Pattern | None]:
    """Fetch Beall's List CSVs from GitHub.

    Returns (publisher_regex, journal_title_regex) — either may be None if fetch failed.
    """
    publisher_names: set[str] = set()
    journal_names: set[str] = set()

    for url, label, target in [
        (BEALLS_PUBLISHERS_URL, "publishers", publisher_names),
        (BEALLS_JOURNALS_URL, "journals", journal_names),
    ]:
        try:
            resp = _requests.get(url, timeout=30)
            resp.raise_for_status()
            target.update(_parse_bealls_csv(resp.text))
            print(f"    Fetched Beall's {label} ({len(target):,} entries)")
        except Exception as exc:
            print(f"    WARNING: Could not fetch Beall's {label}: {exc}")

    def _build_re(names: set[str], min_len: int = 6) -> re.Pattern | None:
        significant = sorted((n for n in names if len(n) >= min_len), key=len, reverse=True)
        return (
            re.compile("|".join(re.escape(n) for n in significant), re.IGNORECASE)
            if significant else None
        )

    return _build_re(publisher_names), _build_re(journal_names)


def enrich_bealls(df: pd.DataFrame) -> pd.DataFrame:
    if not HAS_REQUESTS:
        print("  Skipping - install 'requests': pip install requests")
        df["is_predatory"] = pd.array([False] * len(df), dtype="boolean")
        return df

    publisher_re, journal_re = fetch_bealls()

    def _is_predatory(row) -> bool:
        if publisher_re:
            pub_raw = row.get("publisher")
            if pub_raw and not pd.isna(pub_raw):
                if publisher_re.search(str(pub_raw)):
                    return True
        if journal_re:
            title_raw = row.get("title")
            if title_raw and not pd.isna(title_raw):
                if journal_re.search(str(title_raw)):
                    return True
        return False

    df["is_predatory"] = pd.array(
        [_is_predatory(row) for _, row in df.iterrows()],
        dtype="boolean",
    )
    print(f"    -> {int(df['is_predatory'].sum()):,} journals flagged as predatory")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-openalex", action="store_true", help="Skip OpenAlex API calls")
    parser.add_argument("--skip-bealls", action="store_true", help="Skip Beall's List fetch")
    args = parser.parse_args()

    print(f"Loading {INPUT_PARQUET} ...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  {len(df):,} journals, {len(df.columns)} columns")

    # --- [1/3] SCImago ---
    print("\n[1/3] SCImago SJR")
    sjr_path = find_scimago_file()
    if sjr_path:
        df = enrich_scimago(df, load_scimago(sjr_path))
    else:
        print("  WARNING: No SCImago CSV found - sjr_score / sjr_quartile / h_index will be null.")
        print("           Download: https://www.scimagojr.com/journalrank.php  Export CSV")
        df["sjr_score"] = pd.array([None] * len(df), dtype="Float64")
        df["sjr_quartile"] = [None] * len(df)
        df["h_index"] = pd.array([None] * len(df), dtype="Int64")

    # --- [2/3] OpenAlex ---
    print("\n[2/3] OpenAlex")
    if args.skip_openalex:
        print("  Skipped (--skip-openalex)")
        df["openalex_id"] = pd.NA
        df["articles_per_year"] = pd.array([None] * len(df), dtype="Float64")
        df["is_core"] = pd.array([pd.NA] * len(df), dtype="boolean")
    else:
        df = enrich_openalex(df)

    # --- [3/3] Beall's List ---
    print("\n[3/3] Beall's List")
    if args.skip_bealls:
        print("  Skipped (--skip-bealls)")
        df["is_predatory"] = pd.array([False] * len(df), dtype="boolean")
    else:
        df = enrich_bealls(df)

    # --- Summary ---
    print("\n--- Enrichment summary ---")
    print(f"  SCImago matched:   {df['sjr_score'].notna().sum():>6,} / {len(df):,}")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        print(f"    {q}:             {(df['sjr_quartile'] == q).sum():>6,}")
    print(f"  OpenAlex matched:  {df['openalex_id'].notna().sum():>6,} / {len(df):,}")
    print(f"  Is core (Scopus/WoS): {df['is_core'].sum():>6,}")
    print(f"  Predatory flag:    {df['is_predatory'].sum():>6,}")

    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(
        f"\nWrote {len(df):,} rows -> {OUTPUT_PARQUET}  "
        f"({OUTPUT_PARQUET.stat().st_size / 1_048_576:.1f} MB)"
    )


if __name__ == "__main__":
    main()
