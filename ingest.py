#!/usr/bin/env python3
"""Sprint 1: Ingest and clean the DOAJ journal CSV into a parquet file.

Outputs journals_clean.parquet with:
- Renamed snake_case columns
- Cleaned strings, booleans, dates, and numerics
- apc_usd: APC amounts converted to USD (0.0 for no-APC journals)
- Normalized score columns (0-1): weeks_score, apc_score, volume_score, review_score
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = Path("doaj_journalcsv_20260507_2321_utf8.csv")
OUTPUT_PARQUET = Path("journals_clean.parquet")

# Approximate USD exchange rates (mid-2026)
EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,     "EUR": 1.09,    "GBP": 1.27,    "CHF": 1.12,
    "AUD": 0.65,    "CAD": 0.74,    "SGD": 0.74,    "NZD": 0.61,
    "JPY": 0.0067,  "CNY": 0.138,   "KRW": 0.00072, "TWD": 0.031,
    "IDR": 0.000063,"INR": 0.012,   "PKR": 0.0036,  "BDT": 0.0086,
    "NPR": 0.0075,  "LKR": 0.0033,  "MYR": 0.226,   "THB": 0.028,
    "PHP": 0.0174,  "VND": 0.000039,"KZT": 0.0022,
    "BRL": 0.19,    "ARS": 0.001,   "CLP": 0.001,   "MXN": 0.052,
    "PEN": 0.267,   "COP": 0.00024,
    "ZAR": 0.055,   "NGN": 0.00063, "GHS": 0.065,   "UGX": 0.00027,
    "KES": 0.0077,  "XOF": 0.00163, "XAF": 0.00163, "MAD": 0.099,
    "EGP": 0.020,   "LYD": 0.207,
    "RUB": 0.011,   "UAH": 0.024,   "PLN": 0.25,    "CZK": 0.044,
    "RON": 0.22,    "RSD": 0.0093,  "NOK": 0.093,   "SEK": 0.095,
    "BAM": 0.557,   "HUF": 0.0028,
    "IRR": 0.0000238,"IQD": 0.00076,"YER": 0.004,   "SYP": 0.000077,
    "SAR": 0.267,
    "TRY": 0.029,   "KPW": 0.0011,  "GMD": 0.013,
}

# Higher = more rigorous review
REVIEW_SCORE_MAP: dict[str, float] = {
    "double anonymous peer review": 1.0,
    "double blind peer review":     1.0,
    "double blind":                 1.0,
    "anonymous peer review":        0.75,
    "single anonymous peer review": 0.75,
    "blind peer review":            0.75,
    "open peer review":             0.60,
    "peer review":                  0.50,
    "editorial review":             0.25,
    "none":                         0.00,
}

COLUMN_RENAMES: dict[str, str] = {
    "Journal title":                                                          "title",
    "Journal URL":                                                            "url",
    "URL in DOAJ":                                                            "doaj_url",
    "When did the journal start to publish all content using an open license?": "oa_start_year",
    "Alternative title":                                                      "alt_title",
    "Journal ISSN (print version)":                                           "issn_print",
    "Journal EISSN (online version)":                                         "issn_online",
    "Keywords":                                                               "keywords",
    "Languages in which the journal accepts manuscripts":                     "languages",
    "Publisher":                                                              "publisher",
    "Country of publisher":                                                   "country",
    "Other organisation":                                                     "other_org",
    "Country of other organisation":                                          "other_org_country",
    "Journal license":                                                        "license",
    "License attributes":                                                     "license_attributes",
    "URL for license terms":                                                  "license_url",
    "Machine-readable CC licensing information embedded or displayed in articles": "machine_readable_license",
    "Author holds copyright without restrictions":                            "author_copyright",
    "Copyright information URL":                                              "copyright_url",
    "Review process":                                                         "review_process",
    "Review process information URL":                                         "review_url",
    "Journal plagiarism screening policy":                                    "plagiarism_check",
    "URL for journal's aims & scope":                                         "aims_url",
    "URL for the Editorial Board page":                                       "board_url",
    "URL for journal's instructions for authors":                             "authors_url",
    "Average number of weeks between article submission and publication":     "weeks_to_pub",
    "APC":                                                                    "has_apc",
    "APC information URL":                                                    "apc_url",
    "APC amount":                                                             "apc_amount_raw",
    "Journal waiver policy (for developing country authors etc)":             "has_waiver",
    "Waiver policy information URL":                                          "waiver_url",
    "Has other fees":                                                         "has_other_fees",
    "Other fees information URL":                                             "other_fees_url",
    "Preservation Services":                                                  "preservation_services",
    "Preservation Service: national library":                                 "preservation_national_lib",
    "Preservation information URL":                                           "preservation_url",
    "Deposit policy directory":                                               "deposit_policy",
    "URL for deposit policy":                                                 "deposit_url",
    "Persistent article identifiers":                                         "persistent_ids",
    "Does the journal comply to DOAJ's definition of open access?":           "oa_compliant",
    "Continues":                                                              "continues",
    "Continued By":                                                           "continued_by",
    "LCC Codes":                                                              "lcc_codes",
    "Subscribe to Open":                                                      "subscribe_to_open",
    "Mirror Journal":                                                         "mirror_journal",
    "Open Journals Collective":                                               "open_journals_collective",
    "Subjects":                                                               "subjects",
    "Added on Date":                                                          "added_date",
    "Last updated Date":                                                      "updated_date",
    "Last Full Review Date":                                                  "last_review_date",
    "Number of Article Records":                                              "article_count",
    "Most Recent Article Added":                                              "last_article_date",
}


def parse_apc_to_usd(raw: str) -> float | None:
    """Convert an APC string like '600 USD; 30000 INR' to a USD float.

    When multiple currencies are listed, USD is preferred; otherwise the first
    parseable amount is converted. Returns None when the string is empty or
    no recognisable currency is found.
    """
    if not raw or not raw.strip():
        return None
    fallback: float | None = None
    for part in raw.split(";"):
        m = re.match(r"^([\d,\.]+)\s+([A-Z]{3})$", part.strip())
        if not m:
            continue
        try:
            amount = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        currency = m.group(2)
        rate = EXCHANGE_RATES.get(currency)
        if rate is None:
            continue
        if currency == "USD":
            return amount
        if fallback is None:
            fallback = amount * rate
    return fallback


def score_review_process(val: str | None) -> float:
    """Map a review process string to a 0-1 quality score."""
    if not val:
        return 0.0
    lower = val.lower()
    for key, score in REVIEW_SCORE_MAP.items():
        if key in lower:
            return score
    # A review type exists but doesn't match known patterns
    return 0.30


def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]; NaN propagates, constant → NaN."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - lo) / (hi - lo)


def main() -> None:
    print(f"Loading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8", low_memory=False)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    df = df.rename(columns=COLUMN_RENAMES)

    # --- String columns: strip whitespace, empty string → None ---
    str_cols = [
        "title", "alt_title", "issn_print", "issn_online", "keywords",
        "languages", "publisher", "country", "other_org", "other_org_country",
        "license", "license_attributes", "review_process", "lcc_codes",
        "subjects", "continues", "continued_by", "persistent_ids",
        "preservation_services", "preservation_national_lib", "deposit_policy",
        "apc_amount_raw",
    ]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

    # --- Boolean columns: 'Yes'/'No' → True/False ---
    bool_cols = [
        "machine_readable_license", "author_copyright", "has_apc",
        "has_waiver", "has_other_fees", "plagiarism_check",
        "oa_compliant", "subscribe_to_open", "mirror_journal",
        "open_journals_collective",
    ]
    for col in bool_cols:
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
            .map({"yes": True, "no": False})
            .astype("boolean")
        )

    # --- Numeric columns ---
    df["weeks_to_pub"] = pd.to_numeric(df["weeks_to_pub"], errors="coerce")
    df["article_count"] = pd.to_numeric(df["article_count"], errors="coerce")
    df["oa_start_year"] = (
        pd.to_numeric(df["oa_start_year"], errors="coerce").astype("Int64")
    )

    # --- Date columns ---
    for col in ["added_date", "updated_date", "last_article_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["last_review_date"] = pd.to_datetime(df["last_review_date"], errors="coerce")

    # --- APC: parse raw string → USD ---
    df["apc_usd"] = df["apc_amount_raw"].apply(
        lambda x: parse_apc_to_usd(x) if pd.notna(x) else None
    )
    # Journals that explicitly charge no APC → 0 USD
    df.loc[df["has_apc"] == False, "apc_usd"] = 0.0

    # --- Derived flags ---
    df["has_doi"] = df["persistent_ids"].str.contains("DOI", na=False, case=False)
    df["has_preservation"] = df["preservation_services"].notna()
    df["has_deposit_policy"] = df["deposit_policy"].notna()

    # --- Normalized scoring columns (0 = worst, 1 = best) ---
    # Fewer weeks is better → invert
    df["weeks_score"] = 1.0 - minmax_norm(df["weeks_to_pub"])
    # Lower APC is better → invert; no-APC (0 USD) naturally scores 1.0
    df["apc_score"] = 1.0 - minmax_norm(df["apc_usd"].clip(lower=0))
    # More articles = more established journal
    df["volume_score"] = minmax_norm(df["article_count"])
    # Peer review rigor
    df["review_score"] = df["review_process"].apply(score_review_process)

    # --- Summary ---
    print(f"\nColumn summary:")
    print(f"  Journals with APC:            {df['has_apc'].sum():>6,}")
    print(f"  APC amounts parsed to USD:    {df['apc_usd'].notna().sum():>6,}")
    apc_known = df.loc[df['has_apc'] == True, 'apc_usd']
    print(f"  APC unparseable (has_apc=Yes):{(apc_known.isna()).sum():>6,}")
    print(f"  Journals with DOI:            {df['has_doi'].sum():>6,}")
    print(f"  Journals with preservation:   {df['has_preservation'].sum():>6,}")
    print(f"  Missing weeks_to_pub:         {df['weeks_to_pub'].isna().sum():>6,}")
    apc_stats = df.loc[df['apc_usd'] > 0, 'apc_usd']
    print(f"\nAPC (USD) among paying journals:")
    print(f"  min={apc_stats.min():.0f}  median={apc_stats.median():.0f}"
          f"  mean={apc_stats.mean():.0f}  max={apc_stats.max():.0f}")

    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nWrote {len(df):,} rows -> {OUTPUT_PARQUET}  "
          f"({OUTPUT_PARQUET.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
