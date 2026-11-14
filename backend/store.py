"""Supabase persistence client for CloudPilot using standard library urllib only."""

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def _cfg() -> Tuple[Optional[str], Optional[str]]:
    """Retrieves Supabase URL and Service Key from environment variables."""
    return os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")


def _headers(key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Builds headers for Supabase REST API requests.

    Note:
        Keys starting with 'eyJ' are JWTs and require the 'Authorization: Bearer <key>' header.
        Keys starting with 'sb_secret_' must NOT include the Authorization header.
    """
    h = {
        "apikey": key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    if key.startswith("eyJ"):
        h["Authorization"] = f"Bearer {key}"
    if extra:
        h.update(extra)
    return h


def _post(table: str, row: Dict[str, Any]) -> None:
    """Internal synchronous POST to Supabase table; fails silently on error."""
    url, key = _cfg()
    if not url or not key:
        return
    try:
        endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
        data = json.dumps(row).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers=_headers(key),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            resp.read()
    except Exception:
        # Silent no-op on network failure, missing table, or timeout
        pass


def save(table: str, row: Dict[str, Any]) -> None:
    """Asynchronously persists a row to a Supabase table via daemon thread."""
    url, key = _cfg()
    if not url or not key:
        return
    threading.Thread(target=_post, args=(table, row), daemon=True).start()


def recent(table: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Synchronously queries recent rows from a Supabase table ordered by created_at desc."""
    url, key = _cfg()
    if not url or not key:
        return []
    try:
        endpoint = f"{url.rstrip('/')}/rest/v1/{table}?order=created_at.desc&limit={int(limit)}"
        headers = _headers(key, extra={"Accept": "application/json"})
        req = urllib.request.Request(endpoint, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            content = resp.read().decode("utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

