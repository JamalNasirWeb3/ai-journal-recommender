"""Pydantic request / response schemas."""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    """Public registration — always creates a client account.
    Role promotion requires an existing internal user via POST /auth/promote/{id}.
    """
    email: str
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if v.isdigit():
            raise ValueError("Password must not be all digits")
        return v


class PromoteRequest(BaseModel):
    """Body for POST /auth/promote/{user_id} — internal use only."""
    role: str = "internal"

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ("client", "internal"):
            raise ValueError("role must be 'client' or 'internal'")
        return v


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    title: str
    area: Optional[str] = None
    abstract: Optional[str] = None
    top_k: int = 10
    speed: int = 33
    prestige: int = 34
    cost: int = 33


class JournalResult(BaseModel):
    rank: int
    title: str
    publisher: Optional[str]
    country: Optional[str]
    sjr_quartile: Optional[str]
    apc_usd: float
    weeks_to_pub: Optional[float]
    publication_score: float
    nlp_score: float
    b3_score: float
    b4_score: float
    b5_score: float
    confidence: str
    license: Optional[str]
    is_core: bool
    has_waiver: bool
    doaj_url: Optional[str]
    url: Optional[str]           # journal homepage
    authors_url: Optional[str]   # submission / author instructions
    aims_url: Optional[str]      # aims & scope
    apc_url: Optional[str]       # APC details
    waiver_url: Optional[str]    # waiver info


class MatchResponse(BaseModel):
    query_title: str
    query_area: Optional[str]
    results: List[JournalResult]
    queries_used: int
    queries_remaining: Optional[int]   # None for internal role


# ---------------------------------------------------------------------------
# Explore
# ---------------------------------------------------------------------------

class JournalSummary(BaseModel):
    title: str
    publisher: Optional[str]
    country: Optional[str]
    sjr_quartile: Optional[str]
    apc_usd: float
    weeks_to_pub: Optional[float]
    final_score: float
    cluster_label: Optional[str]
    license: Optional[str]
    is_core: bool
    doaj_url: Optional[str]


class ExploreResponse(BaseModel):
    total: int
    offset: int
    limit: int
    results: List[JournalSummary]


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class SearchHistoryItem(BaseModel):
    id: int
    title: str
    area: Optional[str]
    abstract: Optional[str]
    created_at: datetime
    result_count: int

    model_config = {"from_attributes": True}
