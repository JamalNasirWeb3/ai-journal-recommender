"""Thin HTTP client for calling the FastAPI backend from Streamlit apps.

Usage:
    from api_client import APIClient
    client = APIClient("http://localhost:8000")
    token = client.login("user@example.com", "password")
    results = client.match(token, title="...", area="...")
"""

from __future__ import annotations
from typing import Optional
import requests

DEFAULT_BASE = "http://localhost:8000"
TIMEOUT = 60


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class APIClient:
    def __init__(self, base_url: str = DEFAULT_BASE):
        self.base = base_url.rstrip("/")

    def _get(self, path: str, token: str | None = None, **kwargs):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.get(f"{self.base}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
        if not resp.ok:
            raise APIError(resp.status_code, resp.json().get("detail", resp.text))
        return resp.json()

    def _post(self, path: str, token: str | None = None, **kwargs):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.post(f"{self.base}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
        if not resp.ok:
            raise APIError(resp.status_code, resp.json().get("detail", resp.text))
        return resp.json()

    # ------------------------------------------------------------------
    def health(self) -> bool:
        try:
            return self._get("/health").get("status") == "ok"
        except Exception:
            return False

    def login(self, email: str, password: str) -> str:
        """Returns JWT access token. Raises APIError on failure."""
        data = self._post("/auth/login", json={"email": email, "password": password})
        return data["access_token"]

    def register(self, email: str, password: str, role: str = "client") -> dict:
        return self._post("/auth/register", json={"email": email, "password": password, "role": role})

    def me(self, token: str) -> dict:
        return self._get("/auth/me", token=token)

    def match(
        self,
        token: str,
        title: str,
        area: Optional[str] = None,
        abstract: Optional[str] = None,
        top_k: int = 10,
        speed: int = 33,
        prestige: int = 34,
        cost: int = 33,
    ) -> dict:
        return self._post("/match", token=token, json={
            "title": title, "area": area, "abstract": abstract,
            "top_k": top_k, "speed": speed, "prestige": prestige, "cost": cost,
        })

    def explore(
        self,
        country: str | None = None,
        quartile: str | None = None,
        apc_max: float | None = None,
        language: str | None = None,
        cluster: str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        params = {k: v for k, v in {
            "country": country, "quartile": quartile, "apc_max": apc_max,
            "language": language, "cluster": cluster, "q": q,
            "limit": limit, "offset": offset,
        }.items() if v is not None}
        return self._get("/explore", params=params)

    def get_journal(self, issn: str) -> dict:
        return self._get(f"/journal/{issn}")

    def get_history(self, token: str, limit: int = 20) -> list:
        return self._get("/history", token=token, params={"limit": limit})

    def get_history_results(self, token: str, search_id: int) -> list:
        return self._get(f"/history/{search_id}/results", token=token)
