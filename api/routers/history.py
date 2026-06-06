"""GET /history — authenticated user's search history."""

import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..deps import get_db, get_current_user
from .. import models, schemas

router = APIRouter(tags=["history"])


@router.get("/history", response_model=list[schemas.SearchHistoryItem])
def get_history(
    limit: int = 20,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Return the authenticated user's last `limit` searches, newest first."""
    searches = (
        db.query(models.Search)
        .filter(models.Search.user_id == user.id)
        .order_by(models.Search.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        schemas.SearchHistoryItem(
            id=s.id,
            title=s.title,
            area=s.area,
            abstract=s.abstract,
            created_at=s.created_at,
            result_count=_count_results(s.results_json),
        )
        for s in searches
    ]


@router.get("/history/{search_id}/results")
def get_history_results(
    search_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Return the full result list for a past search."""
    s = db.query(models.Search).filter(
        models.Search.id == search_id,
        models.Search.user_id == user.id,
    ).first()
    if not s:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Search not found")
    return json.loads(s.results_json or "[]")


def _count_results(results_json: str | None) -> int:
    if not results_json:
        return 0
    try:
        return len(json.loads(results_json))
    except Exception:
        return 0
