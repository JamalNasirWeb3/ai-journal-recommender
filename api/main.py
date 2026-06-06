"""Sprint 7: FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload --port 8000

Environment variables:
    JWT_SECRET_KEY      Required in production. Ephemeral random key used if absent (tokens
                        invalidated on restart — dev only).
    ADMIN_EMAIL         Seed an internal admin user at startup.
    ADMIN_PASSWORD      Required with ADMIN_EMAIL.

Interactive docs: http://localhost:8000/docs
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
import nlp as nlp_engine

from .db import engine, SessionLocal
from . import models
from .auth import KEY_IS_EPHEMERAL, hash_password, utcnow
from .routers import auth, journals, match, history

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------
    # 1. Warn about insecure JWT key
    # ------------------------------------------------------------------
    if KEY_IS_EPHEMERAL:
        logger.warning(
            "JWT_SECRET_KEY is not set — using an ephemeral random key. "
            "All tokens will be invalidated on restart. "
            "Set JWT_SECRET_KEY before deploying to production."
        )

    # ------------------------------------------------------------------
    # 2. Create / migrate DB tables
    # ------------------------------------------------------------------
    models.Base.metadata.create_all(bind=engine)

    # ------------------------------------------------------------------
    # 3. Seed admin user from env vars (if no internal user exists yet)
    # ------------------------------------------------------------------
    admin_email    = os.getenv("ADMIN_EMAIL", "").strip()
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()

    db = SessionLocal()
    try:
        has_admin = db.query(models.User).filter(
            models.User.role == "internal"
        ).first()

        if not has_admin:
            if admin_email and admin_password:
                existing = db.query(models.User).filter(
                    models.User.email == admin_email
                ).first()
                if existing:
                    existing.role = "internal"
                    logger.info(f"Promoted existing user to internal: {admin_email}")
                else:
                    db.add(models.User(
                        email=admin_email,
                        hashed_password=hash_password(admin_password),
                        role="internal",
                    ))
                    logger.info(f"Seeded internal admin: {admin_email}")
                db.commit()
            else:
                logger.warning(
                    "No internal user exists. "
                    "Set ADMIN_EMAIL + ADMIN_PASSWORD env vars to seed one, "
                    "or register a user and promote via POST /auth/promote/{id}."
                )

        # ------------------------------------------------------------------
        # 4. Purge expired blacklist entries
        # ------------------------------------------------------------------
        now = utcnow()
        deleted = db.query(models.TokenBlacklist).filter(
            models.TokenBlacklist.expires_at < now
        ).delete()
        if deleted:
            db.commit()
            logger.info(f"Purged {deleted} expired blacklist entries")

    finally:
        db.close()

    # ------------------------------------------------------------------
    # 5. Load parquet + pre-warm NLP model
    # ------------------------------------------------------------------
    parquet_path = Path(__file__).parent.parent / "journals_scored.parquet"
    df = pd.read_parquet(parquet_path)
    for col in ("is_predatory", "is_core", "has_doi", "has_waiver", "plagiarism_check"):
        df[col] = df[col].fillna(False).astype(bool)
    app.state.df = df
    logger.info(f"Loaded {len(df):,} journals from {parquet_path.name}")

    logger.info("Pre-warming NLP model...")
    nlp_engine.load_model()
    logger.info("API ready.")

    yield
    # Shutdown — nothing to clean up


app = FastAPI(
    title="AI Powered Journal Recommender — API",
    description="REST API for the AI Powered Journal Recommender Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(journals.router)
app.include_router(match.router)
app.include_router(history.router)


@app.get("/", tags=["meta"])
def root():
    return {"service": "AI Powered Journal Recommender API", "version": "1.0.0"}


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
