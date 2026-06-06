"""POST /match — NLP matching with rate limiting."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

# nlp.py lives in the project root (parent of this package)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import nlp as nlp_engine

from ..deps import get_db, get_current_user, get_df
from .. import models, schemas

router = APIRouter(tags=["match"])

CLIENT_MONTHLY_LIMIT = 50
ADJUSTABLE = 0.35   # speed + prestige + cost share this pool of the 100%


@router.post("/match", response_model=schemas.MatchResponse)
def match(
    req: schemas.MatchRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
    df: pd.DataFrame = Depends(get_df),
):
    # --- Rate limiting (client role only) ---
    month = datetime.utcnow().strftime("%Y-%m")
    usage = db.query(models.Usage).filter(
        models.Usage.user_id == user.id,
        models.Usage.month == month,
    ).first()

    queries_used = usage.query_count if usage else 0

    if user.role == "client" and queries_used >= CLIENT_MONTHLY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly query limit ({CLIENT_MONTHLY_LIMIT}) reached. "
                   f"Resets on the 1st of next month.",
        )

    # --- NLP matching ---
    clean_df = df[~df["is_predatory"]].copy()
    try:
        nlp_results = nlp_engine.match(
            title=req.title,
            area=req.area or None,
            abstract=req.abstract or None,
            df=clean_df,
            top_k=200,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # --- Apply B2 formula with priority weights ---
    total_pref = (req.speed + req.prestige + req.cost) or 1
    speed_w    = req.speed    / total_pref
    prestige_w = req.prestige / total_pref
    cost_w     = req.cost     / total_pref

    nlp_results["publication_score"] = (
        nlp_results["nlp_score"]                    * 0.40
        + nlp_results["b3_score"].fillna(0.0)        * 0.25
        + nlp_results["weeks_score"].fillna(0.5)     * speed_w    * ADJUSTABLE
        + nlp_results["b4_score"].fillna(0.0)        * prestige_w * ADJUSTABLE
        + nlp_results["b5_score"].fillna(0.5)        * cost_w     * ADJUSTABLE
    ).clip(0, 1)

    top_n = nlp_results.nlargest(req.top_k, "publication_score").reset_index(drop=True)

    # --- Increment usage counter ---
    if usage:
        usage.query_count += 1
    else:
        db.add(models.Usage(user_id=user.id, month=month, query_count=1))

    # --- Save search history ---
    results_list = [_row_to_result(row, i + 1) for i, (_, row) in enumerate(top_n.iterrows())]
    db.add(models.Search(
        user_id=user.id,
        title=req.title,
        area=req.area,
        abstract=req.abstract,
        results_json=json.dumps([r.model_dump() for r in results_list], default=str),
    ))
    db.commit()

    queries_used_now = queries_used + 1
    remaining = (CLIENT_MONTHLY_LIMIT - queries_used_now) if user.role == "client" else None

    return schemas.MatchResponse(
        query_title=req.title,
        query_area=req.area,
        results=results_list,
        queries_used=queries_used_now,
        queries_remaining=remaining,
    )


def _row_to_result(row: pd.Series, rank: int) -> schemas.JournalResult:
    def _s(v) -> Optional[str]:
        return None if (v is None or pd.isna(v)) else str(v)
    def _f(v, default=0.0) -> float:
        return default if (v is None or pd.isna(v)) else float(v)

    return schemas.JournalResult(
        rank=rank,
        title=str(row.get("title") or ""),
        publisher=_s(row.get("publisher")),
        country=_s(row.get("country")),
        sjr_quartile=_s(row.get("sjr_quartile")),
        apc_usd=_f(row.get("apc_usd")),
        weeks_to_pub=_f(row.get("weeks_to_pub")) or None,
        publication_score=_f(row.get("publication_score")),
        nlp_score=_f(row.get("nlp_score")),
        b3_score=_f(row.get("b3_score")),
        b4_score=_f(row.get("b4_score")),
        b5_score=_f(row.get("b5_score")),
        confidence=str(row.get("confidence") or "Low"),
        license=_s(row.get("license")),
        is_core=bool(row.get("is_core") or False),
        has_waiver=bool(row.get("has_waiver") or False),
        doaj_url=_s(row.get("doaj_url")),
        url=_s(row.get("url")),
        authors_url=_s(row.get("authors_url")),
        aims_url=_s(row.get("aims_url")),
        apc_url=_s(row.get("apc_url")),
        waiver_url=_s(row.get("waiver_url")),
    )
