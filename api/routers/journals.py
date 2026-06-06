"""Public journal endpoints: GET /journal/{issn}, GET /explore."""

from typing import Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_df
from .. import schemas

router = APIRouter(tags=["journals"])


@router.get("/journal/{issn}", response_model=schemas.JournalSummary)
def get_journal(issn: str, df: pd.DataFrame = Depends(get_df)):
    """Retrieve a single journal by print or online ISSN."""
    issn = issn.strip().upper()
    mask = (
        df["issn_print"].str.upper().eq(issn)
        | df["issn_online"].str.upper().eq(issn)
    )
    row = df[mask]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Journal with ISSN {issn} not found")
    return _row_to_summary(row.iloc[0])


@router.get("/explore", response_model=schemas.ExploreResponse)
def explore(
    country:  Optional[str] = Query(None),
    quartile: Optional[str] = Query(None, description="Q1 / Q2 / Q3 / Q4"),
    apc_max:  Optional[float] = Query(None, ge=0),
    language: Optional[str] = Query(None),
    cluster:  Optional[str] = Query(None),
    q:        Optional[str] = Query(None, description="Keyword search in title/subjects"),
    limit:    int = Query(50, ge=1, le=200),
    offset:   int = Query(0, ge=0),
    df: pd.DataFrame = Depends(get_df),
):
    """Filtered, paginated journal list sorted by final_score descending."""
    mask = pd.Series(True, index=df.index)

    if country:
        mask &= df["country"].str.lower().eq(country.lower())
    if quartile:
        mask &= df["sjr_quartile"].str.upper().eq(quartile.upper())
    if apc_max is not None:
        mask &= df["apc_usd"] <= apc_max
    if language:
        mask &= df["languages"].str.contains(language, case=False, na=False)
    if cluster:
        mask &= df["cluster_label"].str.lower().eq(cluster.lower())
    if q:
        ql = q.lower()
        mask &= (
            df["title"].str.lower().str.contains(ql, na=False)
            | df["subjects"].str.lower().str.contains(ql, na=False)
        )
    # Always exclude predatory journals from public explore
    mask &= ~df["is_predatory"]

    filtered = df[mask].nlargest(offset + limit, "final_score")
    total = int(mask.sum())
    page = filtered.iloc[offset: offset + limit]

    return schemas.ExploreResponse(
        total=total,
        offset=offset,
        limit=limit,
        results=[_row_to_summary(row) for _, row in page.iterrows()],
    )


def _row_to_summary(row: pd.Series) -> schemas.JournalSummary:
    return schemas.JournalSummary(
        title=str(row.get("title") or ""),
        publisher=_str(row.get("publisher")),
        country=_str(row.get("country")),
        sjr_quartile=_str(row.get("sjr_quartile")),
        apc_usd=float(row.get("apc_usd") or 0),
        weeks_to_pub=_float(row.get("weeks_to_pub")),
        final_score=float(row.get("final_score") or 0),
        cluster_label=_str(row.get("cluster_label")),
        license=_str(row.get("license")),
        is_core=bool(row.get("is_core") or False),
        doaj_url=_str(row.get("doaj_url")),
    )


def _str(val) -> Optional[str]:
    import pandas as pd
    return None if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val)

def _float(val) -> Optional[float]:
    import pandas as pd
    return None if (val is None or (isinstance(val, float) and pd.isna(val))) else float(val)
