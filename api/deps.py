"""FastAPI dependency functions."""

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import pandas as pd

from .db import SessionLocal
from .auth import decode_token
from . import models

_bearer          = HTTPBearer(auto_error=True)
_bearer_optional = HTTPBearer(auto_error=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_df(request: Request) -> pd.DataFrame:
    return request.app.state.df


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> models.User:
    payload = decode_token(credentials.credentials)

    user_id_str = payload.get("sub")
    try:
        uid = int(user_id_str)              # Fix: catch non-numeric sub → 401, not 500
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.query(models.User).filter(
        models.User.id == uid,
        models.User.is_active == True,
    ).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    # Fix: check revoked tokens
    jti = payload.get("jti")
    if jti:
        revoked = db.query(models.TokenBlacklist).filter(
            models.TokenBlacklist.jti == jti
        ).first()
        if revoked:
            raise HTTPException(
                status_code=401,
                detail="Token has been revoked. Please log in again.",
            )

    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_optional),
    db: Session = Depends(get_db),
) -> models.User | None:
    """Return the current user if a valid token is provided, else None.
    Inlined to avoid calling get_current_user.__wrapped__ which bypasses DI.
    """
    if credentials is None:
        return None
    try:
        payload = decode_token(credentials.credentials)
    except HTTPException:
        return None

    user_id_str = payload.get("sub")
    try:
        uid = int(user_id_str)
    except (TypeError, ValueError):
        return None

    jti = payload.get("jti")
    if jti:
        revoked = db.query(models.TokenBlacklist).filter(
            models.TokenBlacklist.jti == jti
        ).first()
        if revoked:
            return None

    return db.query(models.User).filter(
        models.User.id == uid,
        models.User.is_active == True,
    ).first()
