#!/usr/bin/env python3
"""Sprint 3: Scoring engine and KMeans clustering.

Reads:  journals_enriched.parquet
Writes: journals_scored.parquet

New columns:
  b3_score      Float64  Editorial process block score (0-1)
  b4_score      Float64  Quality signals block score (0-1)
  b5_score      Float64  Cost control block score (0-1)
  final_score   Float64  Weighted composite (0-1); 0.0 for predatory journals
  cluster       Int8     KMeans cluster index (0-4); -1 for predatory journals
  cluster_label object   Human-readable cluster name

Scoring blocks (weights normalized to 0-1 internally):
  B3 editorial process (25%): review_score 70% + plagiarism_check 20% + has_doi 10%
  B4 quality signals   (20%): sjr_norm 35% + h_norm 15% + is_core 30% + volume_score 20%
                               -- renormalized when SCImago data is absent
  B5 cost control      (15%): apc_score 80% + has_waiver 20%

  final_score = (B3*0.25 + B4*0.20 + B5*0.15) / 0.60

Hard exclusion: is_predatory=True -> final_score=0.0, cluster=-1
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn is required.  pip install scikit-learn")
    sys.exit(1)

INPUT_PARQUET = Path("journals_enriched.parquet")
OUTPUT_PARQUET = Path("journals_scored.parquet")

N_CLUSTERS = 5
RANDOM_STATE = 42

B3_WEIGHT = 0.25
B4_WEIGHT = 0.20
B5_WEIGHT = 0.15
WEIGHT_TOTAL = B3_WEIGHT + B4_WEIGHT + B5_WEIGHT  # 0.60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]; NaN propagates, constant series -> NaN."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - lo) / (hi - lo)


def to_float(series: pd.Series, fill: float = 0.0) -> pd.Series:
    """Cast nullable bool/int/float to float64, replacing NA with fill."""
    return pd.Series(series.to_numpy(dtype=float, na_value=fill), index=series.index)


# ---------------------------------------------------------------------------
# Block scorers
# ---------------------------------------------------------------------------

def compute_b3(df: pd.DataFrame) -> pd.Series:
    """B3 - Editorial process (0-1).

    review_score (70%) + plagiarism_check (20%) + has_doi (10%)
    """
    review = df["review_score"].fillna(0.0)
    plagiarism = to_float(df["plagiarism_check"])
    has_doi = df["has_doi"].astype(float)
    return (review * 0.70 + plagiarism * 0.20 + has_doi * 0.10).clip(0.0, 1.0)


def compute_b4(df: pd.DataFrame) -> pd.Series:
    """B4 - Quality signals (0-1).

    With SCImago:    sjr_norm (35%) + h_norm (15%) + is_core (30%) + volume_score (20%)
    Without SCImago: is_core (60%) + volume_score (40%)

    Weights are renormalized based on available data so journals without
    SCImago are not systematically penalized.
    """
    sjr_norm = minmax_norm(df["sjr_score"])
    h_norm = minmax_norm(to_float(df["h_index"], fill=np.nan))
    is_core = to_float(df["is_core"])
    vol = df["volume_score"].fillna(0.0)

    has_scimago = sjr_norm.notna().astype(float)

    numer = (
        sjr_norm.fillna(0.0) * 0.35
        + h_norm.fillna(0.0) * 0.15
        + is_core * 0.30
        + vol * 0.20
    )
    # Sum of weights for signals actually available per row
    denom = has_scimago * 0.35 + has_scimago * 0.15 + 0.30 + 0.20

    return (numer / denom).clip(0.0, 1.0)


def compute_b5(df: pd.DataFrame) -> pd.Series:
    """B5 - Cost control (0-1).

    apc_score (80%) + has_waiver (20%)
    """
    apc = df["apc_score"].fillna(0.5)   # unknown APC cost -> neutral
    waiver = to_float(df["has_waiver"])
    return (apc * 0.80 + waiver * 0.20).clip(0.0, 1.0)


def compute_final(b3: pd.Series, b4: pd.Series, b5: pd.Series) -> pd.Series:
    """Weighted composite normalized to 0-1."""
    return ((b3 * B3_WEIGHT + b4 * B4_WEIGHT + b5 * B5_WEIGHT) / WEIGHT_TOTAL).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

# Feature display names for cluster labeling (ordered by priority)
_CLUSTER_FEATURE_NAMES = {
    "apc_score":    "Low Cost",
    "review_score": "Rigorous Review",
    "sjr_norm":     "High Impact",
    "is_core":      "Core Indexed",
    "volume_score": "High Volume",
    "weeks_score":  "Fast Publishing",
}


def run_clustering(df: pd.DataFrame, eligible: pd.Series) -> tuple[np.ndarray, list]:
    """Fit KMeans on eligible journals; return (cluster_indices, cluster_labels).

    Features are log-scaled where heavily right-skewed (volume_score),
    then standardized before clustering.
    """
    feat = pd.DataFrame({
        "apc_score":    df["apc_score"].fillna(0.5),
        "review_score": df["review_score"].fillna(0.0),
        "sjr_norm":     minmax_norm(df["sjr_score"]).fillna(0.0),
        "is_core":      to_float(df["is_core"]),
        "volume_score": np.log1p(df["volume_score"].fillna(0.0)),  # log-scale skewed dist
        "weeks_score":  df["weeks_score"].fillna(0.5),
    })

    X = feat.loc[eligible].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    raw_labels = km.predict(X_scaled)

    # Label each cluster by its most distinctive feature (highest standardized centroid)
    centroid_df = pd.DataFrame(km.cluster_centers_, columns=feat.columns)
    name_map: dict[int, str] = {}
    used: set[str] = set()
    # Process clusters in order of their max standardized value (most distinctive first)
    prominence = centroid_df.max(axis=1).sort_values(ascending=False)
    for cluster_idx in prominence.index:
        ranked = centroid_df.loc[cluster_idx].sort_values(ascending=False)
        for feat_name in ranked.index:
            name = _CLUSTER_FEATURE_NAMES[feat_name]
            if name not in used:
                name_map[int(cluster_idx)] = name
                used.add(name)
                break

    # Map back to full dataframe length
    cluster_arr = np.full(len(df), -1, dtype=np.int8)
    label_arr: list = [None] * len(df)
    eligible_positions = np.where(eligible.values)[0]
    for pos, lbl in zip(eligible_positions, raw_labels):
        cluster_arr[pos] = int(lbl)
        label_arr[pos] = name_map[int(lbl)]

    return cluster_arr, label_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {INPUT_PARQUET} ...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  {len(df):,} journals, {len(df.columns)} columns")

    # --- Block scores ---
    print("\nComputing block scores ...")
    df["b3_score"] = compute_b3(df)
    df["b4_score"] = compute_b4(df)
    df["b5_score"] = compute_b5(df)
    df["final_score"] = compute_final(df["b3_score"], df["b4_score"], df["b5_score"])

    # Hard exclusion
    predatory_mask = df["is_predatory"].fillna(False).astype(bool)
    n_excl = int(predatory_mask.sum())
    df.loc[predatory_mask, "final_score"] = 0.0
    print(f"  Hard-excluded {n_excl:,} predatory journals (final_score = 0.0)")

    # Score distribution (non-predatory only)
    eligible_mask = ~predatory_mask
    s = df.loc[eligible_mask, "final_score"]
    print(f"\nScore distribution ({eligible_mask.sum():,} eligible journals):")
    for block, col in [("B3 editorial", "b3_score"), ("B4 quality  ", "b4_score"), ("B5 cost     ", "b5_score")]:
        v = df[col]
        print(f"  {block}: mean={v.mean():.3f}  std={v.std():.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}")
    print(f"  Final:       mean={s.mean():.3f}  std={s.std():.3f}  "
          f"min={s.min():.3f}  max={s.max():.3f}")
    for label, q in [("Top 10%", 0.90), ("Top 25%", 0.75), ("Bottom 25%", 0.25)]:
        print(f"  {label} threshold: {s.quantile(q):.3f}")

    # --- Clustering ---
    print(f"\nRunning KMeans (k={N_CLUSTERS}) on {eligible_mask.sum():,} eligible journals ...")
    cluster_arr, label_arr = run_clustering(df, eligible_mask)
    df["cluster"] = pd.array(cluster_arr, dtype="Int8")
    df["cluster_label"] = label_arr

    print("\nCluster summary:")
    for c in range(N_CLUSTERS):
        mask = df["cluster"] == c
        if not mask.any():
            continue
        name = df.loc[mask, "cluster_label"].iloc[0]
        fs = df.loc[mask, "final_score"]
        apc = df.loc[mask, "apc_usd"]
        q_counts = df.loc[mask, "sjr_quartile"].value_counts()
        q1 = q_counts.get("Q1", 0)
        print(f"  [{c}] {name:<20}  n={mask.sum():>5,}  "
              f"score={fs.mean():.3f}  "
              f"median_apc=${apc.median():.0f}  "
              f"Q1={q1:,}")

    # --- Final summary ---
    print(f"\n--- Output summary ---")
    print(f"  Journals scored:   {eligible_mask.sum():,}")
    print(f"  Predatory excluded:{n_excl:,}")
    print(f"  Clusters assigned: {(df['cluster'] >= 0).sum():,}")
    print(f"  New columns:       b3_score, b4_score, b5_score, final_score, cluster, cluster_label")

    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nWrote {len(df):,} rows -> {OUTPUT_PARQUET}  "
          f"({OUTPUT_PARQUET.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
