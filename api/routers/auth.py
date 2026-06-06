"""Authentication routes: register, login, logout, /me, promote."""

from datetime import timedelta, datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from ..deps import get_db, get_current_user
from .. import models, schemas
from ..auth import (
    hash_password, verify_password, create_access_token, decode_token, utcnow
)

router = APIRouter(prefix="/auth", tags=["auth"])

MAX_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
ATTEMPT_RESET_SECONDS = 3600   # reset counter after 1 h of no attempts


# ---------------------------------------------------------------------------
# Brute-force helpers
# ---------------------------------------------------------------------------

def _check_lockout(email: str, db: Session) -> None:
    record = db.query(models.LoginAttempt).filter(
        models.LoginAttempt.email == email
    ).first()
    if record and record.locked_until and record.locked_until > utcnow():
        wait = max(1, int((record.locked_until - utcnow()).total_seconds() / 60))
        raise HTTPException(
            status_code=429,
            detail=f"Account locked after {MAX_ATTEMPTS} failed attempts. "
                   f"Try again in {wait} minute(s).",
        )


def _record_failure(email: str, db: Session) -> None:
    now = utcnow()
    record = db.query(models.LoginAttempt).filter(
        models.LoginAttempt.email == email
    ).first()
    if record:
        # Auto-reset counter if last attempt was >1 hour ago
        if record.last_attempt and (now - record.last_attempt).total_seconds() > ATTEMPT_RESET_SECONDS:
            record.attempt_count = 0
            record.locked_until = None
        record.attempt_count += 1
        record.last_attempt = now
        if record.attempt_count >= MAX_ATTEMPTS:
            record.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
    else:
        db.add(models.LoginAttempt(email=email, attempt_count=1, last_attempt=now))
    db.commit()


def _clear_failures(email: str, db: Session) -> None:
    record = db.query(models.LoginAttempt).filter(
        models.LoginAttempt.email == email
    ).first()
    if record:
        record.attempt_count = 0
        record.locked_until = None
        db.commit()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/register", response_model=schemas.UserOut, status_code=201)
def register(req: schemas.RegisterRequest, db: Session = Depends(get_db)):
    """Register a new client account. Role is always 'client'.
    To create an internal user, register first then call POST /auth/promote/{id}.
    """
    if db.query(models.User).filter(models.User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = models.User(
        email=req.email,
        hashed_password=hash_password(req.password),
        role="client",   # always client — cannot be overridden by caller
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=schemas.Token)
def login(req: schemas.LoginRequest, db: Session = Depends(get_db)):
    _check_lockout(req.email, db)

    user = db.query(models.User).filter(models.User.email == req.email).first()

    if not user or not verify_password(req.password, user.hashed_password):
        _record_failure(req.email, db)
        # Generic message — don't reveal whether the email exists
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    _clear_failures(req.email, db)
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return schemas.Token(access_token=token)


@router.post("/logout")
def logout(
    credentials=Depends(HTTPBearer()),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Revoke the current token immediately."""
    payload = decode_token(credentials.credentials)
    jti = payload.get("jti")
    exp_ts = payload.get("exp")

    if jti:
        expires_at = (
            datetime.fromtimestamp(exp_ts, tz=timezone.utc).replace(tzinfo=None)
            if exp_ts else utcnow() + timedelta(hours=24)
        )
        db.add(models.TokenBlacklist(
            jti=jti,
            user_id=user.id,
            expires_at=expires_at,
        ))
        db.commit()
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=schemas.UserOut)
def me(user: models.User = Depends(get_current_user)):
    return user


@router.post("/promote/{target_id}", response_model=schemas.UserOut)
def promote_user(
    target_id: int,
    body: schemas.PromoteRequest = schemas.PromoteRequest(),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Change another user's role. Requires the caller to be internal."""
    if current_user.role != "internal":
        raise HTTPException(status_code=403, detail="Only internal users can change roles")
    if current_user.id == target_id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    target = db.query(models.User).filter(models.User.id == target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    target.role = body.role
    db.commit()
    db.refresh(target)
    return target
