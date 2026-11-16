"""Supabase persistence client for CloudPilot using standard library urllib only."""

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

_last_save_error: Optional[Dict[str, Any]] = None
_last_save_lock = threading.Lock()


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


def _scrub(text: str, secret: Optional[str]) -> str:
    """Removes secret key if it appears in error messages or response bodies."""
    if not text:
        return ""
    if secret and secret in text:
        text = text.replace(secret, "[REDACTED]")
    return text


def _record_save_error(table: str, status: Optional[int], raw_body: str, key: Optional[str]) -> None:
    """Logs non-fatal error JSON to stdout and tracks last_save_error in memory."""
    global _last_save_error
    clean_body = _scrub(raw_body, key)
    log_entry = {
        "store_error": True,
        "table": table,
        "status": status,
        "body": clean_body[:300],
    }
    try:
        sys.stdout.write(json.dumps(log_entry) + "\n")
        sys.stdout.flush()
    except Exception:
        pass

    with _last_save_lock:
        _last_save_error = {
            "status": status,
            "body": clean_body[:200],
        }


def _post(table: str, row: Dict[str, Any]) -> None:
    """Internal synchronous POST to Supabase table; fails safely with structured log on error."""
    url, key = _cfg()
    if not url or not key:
        return
    status: Optional[int] = None
    raw_body: str = ""
    try:
        endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
        data = json.dumps(row).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers=_headers(key),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            status = getattr(resp, "status", getattr(resp, "code", 200))
            raw_body = resp.read().decode("utf-8", errors="replace")
            if status < 200 or status >= 300:
                _record_save_error(table, status, raw_body, key)
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            raw_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw_body = str(exc)
        _record_save_error(table, status, raw_body, key)
    except Exception as exc:
        _record_save_error(table, None, str(exc), key)


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


def diagnose() -> Dict[str, Any]:
    """Runs a synchronous diagnostic check on the Supabase store connection.

    Never exposes the API key in the returned dictionary or logs.
    """
    url, key = _cfg()
    url_set = bool(url)
    key_set = bool(key)

    if key and key.startswith("sb_secret_"):
        key_type = "sb_secret"
    elif key and key.startswith("eyJ"):
        key_type = "jwt"
    else:
        key_type = "other"

    url_host: Optional[str] = None
    if url:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        url_host = parsed.hostname

    status_code: Optional[int] = None
    response_body: Optional[str] = None

    if not url or not key:
        response_body = "Missing SUPABASE_URL or SUPABASE_SERVICE_KEY"
    else:
        try:
            endpoint = f"{url.rstrip('/')}/rest/v1/simulation_runs"
            payload = {
                "policy": "diagnostic",
                "num_workloads": 0,
                "seed": 0,
                "result": {},
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=data,
                headers=_headers(key),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                status_code = getattr(resp, "status", getattr(resp, "code", 200))
                raw_body = resp.read().decode("utf-8", errors="replace")
                response_body = raw_body[:300]
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            try:
                raw_body = exc.read().decode("utf-8", errors="replace")
                response_body = raw_body[:300] if raw_body else str(exc)[:300]
            except Exception:
                response_body = str(exc)[:300]
        except Exception as exc:
            status_code = None
            response_body = str(exc)[:300]

    with _last_save_lock:
        last_err = dict(_last_save_error) if _last_save_error is not None else None

    return {
        "url_set": url_set,
        "key_set": key_set,
        "key_type": key_type,
        "url_host": url_host,
        "status_code": status_code,
        "http_status": status_code,
        "response_body": response_body,
        "last_save_error": last_err,
    }
