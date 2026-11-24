from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import requests
from fastapi import HTTPException, Request


@dataclass(frozen=True)
class Principal:
    user_id: str
    email: str | None = None


def auth_mode() -> str:
    return os.getenv(
        "FINAGENT_AUTH_MODE", "supabase" if os.getenv("SUPABASE_URL") else "token"
    )


def valid_service_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return bool(
            parts.hostname
            and not parts.username
            and not parts.password
            and not parts.query
            and not parts.fragment
            and (
                parts.scheme == "https"
                or parts.scheme == "http"
                and parts.hostname in {"127.0.0.1", "localhost", "::1"}
            )
        )
    except ValueError:
        return False


def supabase_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "") or os.getenv(
        "SUPABASE_ANON_KEY", ""
    )
    if not url or not key:
        raise HTTPException(503, "Authentication is not configured.")
    if not valid_service_url(url):
        raise HTTPException(503, "Supabase requires HTTPS.")
    headers = {"apikey": key, **kwargs.pop("headers", {})}
    try:
        return requests.request(
            method,
            f"{url}{path}",
            headers=headers,
            timeout=(5, 15),
            allow_redirects=False,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise HTTPException(503, "Authentication service is unavailable.") from exc


def require_principal(request: Request) -> Principal:
    verified = getattr(request.state, "principal", None)
    if isinstance(verified, Principal):
        return verified
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            401, "Sign in to continue.", headers={"WWW-Authenticate": "Bearer"}
        )
    if auth_mode() == "token":
        expected = os.getenv("FINAGENT_API_TOKEN", "")
        if not expected:
            raise HTTPException(503, "Worker access is not configured.")
        if not hmac.compare_digest(token.encode(), expected.encode()):
            raise HTTPException(401, "Invalid access token.")
        principal = Principal(
            "local-" + hashlib.sha256(expected.encode()).hexdigest()[:24]
        )
    elif auth_mode() == "supabase":
        response = supabase_request(
            "GET", "/auth/v1/user", headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code != 200:
            raise HTTPException(
                401 if response.status_code in {400, 401, 403} else 503,
                "Session verification failed.",
            )
        try:
            user = response.json()
            if not isinstance(user, dict) or not isinstance(user.get("id"), str):
                raise ValueError("Invalid identity")
            user_id = str(UUID(user["id"]))
        except ValueError as exc:
            raise HTTPException(503, "Session verification is unavailable.") from exc
        principal = Principal(
            user_id, user.get("email") if isinstance(user.get("email"), str) else None
        )
    else:
        raise HTTPException(503, "Unsupported authentication configuration.")
    request.state.principal = principal
    return principal
