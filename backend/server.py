"""Zero-dependency HTTP REST API and live control-plane dashboard for CloudPilot."""

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import hmac
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from allocation.exceptions import (
    AllocationError,
    InvalidResourceError,
    ReleaseError,
    SimulationLimitError,
    SimulationTimeoutError,
)
import backend.store as store
from backend.service import CloudPilotService
from backend.validators import (
    ALLOWED_POLICIES,
    validate_allocate_payload,
    validate_policy_payload,
    validate_release_payload,
    validate_simulate_payload,
)

MAX_BODY = 64 * 1024  # 64 KB
SIM_SLOTS = threading.BoundedSemaphore(2)

DEFAULT_ALLOWED_ORIGIN = "https://cloud-pilot-gamma.vercel.app"

# Rate limits: (max_requests, window_seconds)
SIM_RATE_LIMIT = 10
SIM_RATE_WINDOW = 60.0
_sim_rate_lock = threading.Lock()
_sim_rate_history: Dict[str, List[float]] = {}

GENERAL_RATE_LIMIT = 120
GENERAL_RATE_WINDOW = 60.0
_general_rate_lock = threading.Lock()
_general_rate_history: Dict[str, List[float]] = {}

# All state-mutating and simulation endpoints are protected
PROTECTED_POST_PATHS = {
    "/api/v1/policy",
    "/api/v1/allocate",
    "/api/v1/release",
    "/api/v1/reset",
    "/api/v1/simulate",
}

PROTECTED_GET_PATHS = {
    "/api/v1/cluster",
    "/api/v1/history",
    "/api/v1/store/status",
    "/api/v1/health",
}


def is_production() -> bool:
    """Checks whether the server is running in a production environment (e.g. Render)."""
    return bool(
        os.environ.get("RENDER")
        or os.environ.get("ENVIRONMENT", "").strip().lower() == "production"
        or os.environ.get("NODE_ENV", "").strip().lower() == "production"
    )



def _check_rate_limit(client_ip: str, category: str) -> bool:
    """Checks per-IP rate limit for a given endpoint category."""
    now = time.time()
    if category == "simulate":
        with _sim_rate_lock:
            recent = [t for t in _sim_rate_history.get(client_ip, []) if now - t < SIM_RATE_WINDOW]
            if len(recent) >= SIM_RATE_LIMIT:
                _sim_rate_history[client_ip] = recent
                return False
            recent.append(now)
            _sim_rate_history[client_ip] = recent
            return True
    else:
        with _general_rate_lock:
            recent = [t for t in _general_rate_history.get(client_ip, []) if now - t < GENERAL_RATE_WINDOW]
            if len(recent) >= GENERAL_RATE_LIMIT:
                _general_rate_history[client_ip] = recent
                return False
            recent.append(now)
            _general_rate_history[client_ip] = recent
            return True


class PayloadTooLargeError(Exception):
    """Raised when incoming request payload exceeds MAX_BODY."""
    pass


def _sanitize_for_storage(obj: Any) -> Any:
    """Recursively cleans objects for JSON storage, rounding floats and dropping NaN/inf."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 4)
    if is_dataclass(obj):
        return _sanitize_for_storage(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_storage(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_storage(v) for v in obj]
    return str(obj)


def _prepare_simulation_row(
    policy_key: str,
    num_workloads: int,
    seed: int,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the exact database row structure expected by Supabase simulation_runs."""
    clean_result = _sanitize_for_storage(result)
    if not isinstance(clean_result, dict):
        clean_result = {"result": clean_result}

    # Capped at 50 KB
    MAX_BYTES = 50 * 1024
    serialized = json.dumps(clean_result)
    if len(serialized.encode("utf-8")) > MAX_BYTES:
        clean_result = {
            k: v for k, v in clean_result.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        clean_result["_truncated"] = True

    return {
        "policy": str(policy_key),
        "num_workloads": int(num_workloads),
        "seed": int(seed),
        "result": clean_result,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>CloudPilot Control Plane</title>
  <style>
    body { background: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; margin: 0; padding: 24px; }
    h1 { margin-top: 0; color: #38bdf8; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; }
    .metric { font-size: 28px; font-weight: 700; color: #38bdf8; }
    button, select, input { background: #0f172a; color: #f8fafc; border: 1px solid #475569; padding: 8px 12px; border-radius: 6px; }
    button { background: #0284c7; border-color: #0284c7; cursor: pointer; font-weight: 600; }
    pre { background: #020617; padding: 12px; border-radius: 8px; overflow-x: auto; border: 1px solid #1e293b; }
  </style>
</head>
<body>
  <h1>CloudPilot ML Resource Allocation Control Plane</h1>
  <div class="grid" id="metrics"></div>
  <div class="card" style="margin-bottom: 16px;">
    <h3>Run On-Demand Trace Simulation</h3>
    <select id="policy">
      <option value="predictive_ar">Predictive (LinearAR)</option>
      <option value="predictive_ewma">Predictive (EWMA)</option>
      <option value="ucb_bandit">UCB-Bandit (RL)</option>
      <option value="first_fit">FirstFit</option>
      <option value="best_fit">BestFit</option>
      <option value="worst_fit">WorstFit</option>
      <option value="round_robin">RoundRobin</option>
    </select>
    <input id="num_workloads" type="number" value="200" min="10" max="2000" />
    <button onclick="runSim()">Run Trace Simulation</button>
    <button onclick="resetCluster()" style="background:#dc2626;border-color:#dc2626;">Reset Cluster</button>
  </div>
  <pre id="output">Loading cluster telemetry...</pre>
  <script>
    async function refresh() {
      const res = await fetch('/api/v1/cluster');
      const data = await res.json();
      const metricsDiv = document.getElementById('metrics');
      metricsDiv.textContent = '';
      const items = [
        ['Active Policy', data.active_policy],
        ['CPU Utilization', data.cpu_utilization_pct + '%'],
        ['RAM Utilization', data.ram_utilization_pct + '%'],
        ['Active Servers', data.servers_in_use + ' / ' + data.total_servers]
      ];
      for (const [title, val] of items) {
        const card = document.createElement('div');
        card.className = 'card';
        const label = document.createElement('div');
        label.textContent = title;
        const metric = document.createElement('div');
        metric.className = 'metric';
        metric.textContent = String(val);
        card.appendChild(label);
        card.appendChild(metric);
        metricsDiv.appendChild(card);
      }
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    }
    async function runSim() {
      const policy = document.getElementById('policy').value;
      const num_workloads = parseInt(document.getElementById('num_workloads').value, 10);
      const res = await fetch('/api/v1/simulate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({policy, num_workloads, seed: 42})
      });
      document.getElementById('output').textContent = JSON.stringify(await res.json(), null, 2);
    }
    async function resetCluster() {
      await fetch('/api/v1/reset', {method: 'POST'});
      await refresh();
    }
    refresh();
  </script>
</body>
</html>
"""


class CloudPilotHTTPServer(ThreadingHTTPServer):
    """Custom ThreadingHTTPServer holding a reference to the CloudPilotService instance."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: type,
        service: Optional[CloudPilotService] = None,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.service: CloudPilotService = service if service is not None else CloudPilotService()


class CloudPilotRequestHandler(BaseHTTPRequestHandler):
    """REST request handler for CloudPilot API endpoints."""

    server: CloudPilotHTTPServer
    req_id: str = ""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default noisy apache stderr logging
        return

    def _has_valid_api_key(self) -> bool:
        """Checks if request carries a verified API key matching CLOUDPILOT_API_KEY."""
        configured_key = os.environ.get("CLOUDPILOT_API_KEY")
        if not configured_key:
            return False
        header_key = self.headers.get("X-API-Key")
        if not header_key:
            auth_hdr = self.headers.get("Authorization", "")
            if auth_hdr.startswith("Bearer "):
                header_key = auth_hdr[7:].strip()
            else:
                header_key = auth_hdr.strip()
        return bool(header_key and hmac.compare_digest(header_key, configured_key))

    def _get_client_ip(self) -> str:
        """Determines client IP behind trusted reverse proxy (e.g. Render).

        Reverse proxies like Render append the connecting client IP to the end of X-Forwarded-For.
        Using the last entry prevents IP spoofing by clients who prepend forged IPs.

        When the request carries a verified API key (e.g. from the Vercel proxy), the trusted
        proxy sets X-CloudPilot-Client-IP. Only requests with a valid API key may use this header;
        otherwise it falls back to the rightmost entry of X-Forwarded-For.
        """
        if self._has_valid_api_key():
            proxy_client_ip = self.headers.get("X-CloudPilot-Client-IP")
            if proxy_client_ip and proxy_client_ip.strip():
                return proxy_client_ip.strip()

        xff = self.headers.get("X-Forwarded-For")
        if xff:
            parts = [p.strip() for p in xff.split(",") if p.strip()]
            if parts:
                return parts[-1]
        if self.client_address:
            return self.client_address[0]
        return "127.0.0.1"

    def _get_request_id(self) -> str:
        req_id = self.headers.get("X-Request-ID")
        if req_id and req_id.strip():
            return req_id.strip()[:64]
        return f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    def _log_json(self, method: str, path: str, status: int, duration_ms: float, req_id: str, client_ip: str) -> None:
        """Emits structured JSON access log to stdout with hashed client IP (never raw IP)."""
        client_ip_hash = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()[:12]
        entry = {
            "timestamp": round(time.time(), 3),
            "request_id": req_id,
            "method": method,
            "path": path,
            "status": int(status),
            "duration_ms": round(duration_ms, 2),
            "client_ip_hash": client_ip_hash,
        }
        try:
            sys.stdout.write(json.dumps(entry) + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def _get_allowed_origin(self) -> str:
        """Determines allowed CORS origin from ALLOWED_ORIGIN env var without wildcard default."""
        configured = os.environ.get("ALLOWED_ORIGIN", "").strip()
        if not configured:
            return DEFAULT_ALLOWED_ORIGIN
        allowed_list = [o.strip() for o in configured.split(",") if o.strip()]
        client_origin = self.headers.get("Origin", "")
        if client_origin and client_origin in allowed_list:
            return client_origin
        return allowed_list[0] if allowed_list else DEFAULT_ALLOWED_ORIGIN

    def _send_cors_headers(self) -> None:
        """Emits standard CORS headers configurable via ALLOWED_ORIGIN env var. No wildcard."""
        origin = self._get_allowed_origin()
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization, X-Request-ID")
        self.send_header("Vary", "Origin")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.req_id:
            self.send_header("X-Request-ID", self.req_id)
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_error(
        self,
        status: int,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Returns structured JSON error response without leaking stack traces."""
        error_name = code or HTTPStatus(status).name.lower()
        payload: Dict[str, Any] = {
            "error": message,
            "status": int(status),
            "code": error_name,
        }
        if details:
            payload.update(details)
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.req_id:
            self.send_header("X-Request-ID", self.req_id)
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.req_id:
            self.send_header("X-Request-ID", self.req_id)
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_OPTIONS(self) -> None:
        """Handles CORS preflight requests."""
        self.req_id = self._get_request_id()
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.req_id:
            self.send_header("X-Request-ID", self.req_id)
        self._send_cors_headers()
        self.end_headers()

    def _authorized(self) -> bool:
        """Verifies X-API-Key or Authorization header against CLOUDPILOT_API_KEY if configured."""
        configured_key = os.environ.get("CLOUDPILOT_API_KEY")
        if not configured_key:
            return True
        return self._has_valid_api_key()

    def _read_json_body(self) -> Dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except (ValueError, TypeError):
            raise InvalidResourceError("Invalid Content-Length.")
        if length < 0:
            raise InvalidResourceError("Invalid Content-Length.")
        if length > MAX_BODY:
            try:
                self.rfile.read(min(length, 256 * 1024))
            except Exception:
                pass
            raise PayloadTooLargeError(
                f"Payload too large. Maximum allowed size is {MAX_BODY} bytes."
            )
        if length == 0:
            return {}

        try:
            raw_bytes = self.rfile.read(length)
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidResourceError(f"Malformed UTF-8 payload: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InvalidResourceError(f"Invalid JSON payload: {exc}") from exc
        if not isinstance(data, dict):
            raise InvalidResourceError("JSON body must be an object.")
        return data

    def do_GET(self) -> None:
        t0 = time.perf_counter()
        status = HTTPStatus.OK
        self.req_id = self._get_request_id()
        client_ip = self._get_client_ip()

        try:
            path = urlparse(self.path).path

            # Production Fail-Closed Gate: Reject non-health endpoints if key is missing in production
            if is_production() and not os.environ.get("CLOUDPILOT_API_KEY"):
                if path not in ("/health", "/healthz"):
                    status = HTTPStatus.SERVICE_UNAVAILABLE
                    self._send_error(
                        status,
                        "Service unavailable: CLOUDPILOT_API_KEY must be configured in production.",
                        code="production_auth_misconfigured",
                    )
                    return

            if not _check_rate_limit(client_ip, "default"):
                status = HTTPStatus.TOO_MANY_REQUESTS
                self._send_error(status, "Rate limit exceeded. Try again shortly.", code="too_many_requests")
                return

            if path in ("/", "/index.html"):
                self._send_html(HTTPStatus.OK, DASHBOARD_HTML)
            elif path in ("/health", "/healthz"):
                self._send_json(HTTPStatus.OK, {
                    "status": "ok",
                    "service": "CloudPilot Control Plane",
                    "uptime_sec": round(time.time() - self.server.service._started_at, 2),
                })
            elif path in ("/ready", "/api/v1/ready"):
                is_ready, _, _ = store.check_ready()
                if is_ready:
                    self._send_json(HTTPStatus.OK, {"status": "ok"})
                else:
                    self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "not ok"})
            elif path in PROTECTED_GET_PATHS:
                if not self._authorized():
                    status = HTTPStatus.UNAUTHORIZED
                    self._send_error(status, "Unauthorized. Valid X-API-Key required.", code="unauthorized")
                    return
                if path == "/api/v1/health":
                    self._send_json(HTTPStatus.OK, self.server.service.get_health())
                elif path == "/api/v1/cluster":
                    self._send_json(HTTPStatus.OK, self.server.service.get_cluster_status())
                elif path == "/api/v1/history":
                    self._send_json(HTTPStatus.OK, store.recent("simulation_runs"))
                elif path == "/api/v1/store/status":
                    self._send_json(HTTPStatus.OK, store.diagnose())
            else:
                status = HTTPStatus.NOT_FOUND
                self._send_error(status, f"Endpoint '{path}' not found.", code="not_found")
        except Exception:
            traceback.print_exc(file=sys.stderr)
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            self._send_error(status, "Internal server error.", code="internal_error")
        finally:
            self._log_json("GET", self.path, status, (time.perf_counter() - t0) * 1000.0, self.req_id, client_ip)

    def do_POST(self) -> None:
        t0 = time.perf_counter()
        status = HTTPStatus.OK
        acquired_sim = False
        self.req_id = self._get_request_id()
        client_ip = self._get_client_ip()

        try:
            path = urlparse(self.path).path

            # Production Fail-Closed Gate: Reject non-health endpoints if key is missing in production
            if is_production() and not os.environ.get("CLOUDPILOT_API_KEY"):
                status = HTTPStatus.SERVICE_UNAVAILABLE
                self._send_error(
                    status,
                    "Service unavailable: CLOUDPILOT_API_KEY must be configured in production.",
                    code="production_auth_misconfigured",
                )
                return

            # Category-based rate limiting
            category = "simulate" if path == "/api/v1/simulate" else "write"
            if not _check_rate_limit(client_ip, category):
                status = HTTPStatus.TOO_MANY_REQUESTS
                msg = (
                    "Rate limit exceeded. Max 10 simulations per minute."
                    if category == "simulate"
                    else "Rate limit exceeded. Try again shortly."
                )
                self._send_error(status, msg, code="too_many_requests")
                return

            # API key authentication on all write/simulate endpoints
            if path in PROTECTED_POST_PATHS and not self._authorized():
                status = HTTPStatus.UNAUTHORIZED
                self._send_error(status, "Unauthorized. Valid X-API-Key required.", code="unauthorized")
                return

            body = self._read_json_body()

            if path == "/api/v1/policy":
                clean_policy = validate_policy_payload(body)
                res = self.server.service.set_policy(clean_policy)
                status = HTTPStatus.OK
                self._send_json(status, res)

            elif path == "/api/v1/allocate":
                alloc_data = validate_allocate_payload(body)
                res = self.server.service.allocate_workload(
                    workload_id=alloc_data["workload_id"],
                    cpu=alloc_data["cpu"],
                    ram=alloc_data["ram"],
                    arrival_time=alloc_data["arrival_time"],
                    duration=alloc_data["duration"],
                )
                alloc_record = {
                    "workload_id": alloc_data["workload_id"],
                    "cpu": alloc_data["cpu"],
                    "ram": alloc_data["ram"],
                    "placed": bool(res.get("placed")),
                    "server_id": res.get("server_id"),
                    "policy": res.get("active_policy"),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                store.save("allocation_events", alloc_record)
                status = HTTPStatus.CREATED if res["placed"] else HTTPStatus.CONFLICT
                self._send_json(status, res)

            elif path == "/api/v1/release":
                workload_id = validate_release_payload(body)
                res = self.server.service.release_workload(workload_id)
                status = HTTPStatus.OK
                self._send_json(status, res)

            elif path == "/api/v1/reset":
                res = self.server.service.reset_cluster()
                status = HTTPStatus.OK
                self._send_json(status, res)

            elif path == "/api/v1/simulate":
                # Concurrency control via semaphore with immediate Retry-After
                if not SIM_SLOTS.acquire(blocking=False):
                    status = HTTPStatus.TOO_MANY_REQUESTS
                    self._send_error(
                        status,
                        "Simulation queue full. Try again shortly.",
                        code="simulation_queue_full",
                        extra_headers={"Retry-After": "5"},
                    )
                    return
                acquired_sim = True

                req_policy, num_workloads, seed = validate_simulate_payload(body)

                res = self.server.service.run_simulation(
                    policy=req_policy,
                    num_workloads=num_workloads,
                    seed=seed,
                )
                policy_key = str(res.get("policy_key") or req_policy or "predictive_ar")
                sim_row = _prepare_simulation_row(
                    policy_key=policy_key,
                    num_workloads=num_workloads,
                    seed=seed,
                    result=res,
                )

                # Persist with retry and timeout; if database is down, simulation still succeeds
                saved = store.save_sync("simulation_runs", sim_row, max_retries=1, timeout=2.0)
                if saved:
                    res["store_status"] = "saved"
                    res["persisted"] = True
                else:
                    res["store_status"] = "not saved"
                    res["persisted"] = False

                status = HTTPStatus.OK
                self._send_json(status, res)

            else:
                status = HTTPStatus.NOT_FOUND
                self._send_error(status, f"Endpoint '{path}' not found.", code="not_found")

        except PayloadTooLargeError as exc:
            status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            self._send_error(status, str(exc), code="request_entity_too_large")
        except InvalidResourceError as exc:
            status = HTTPStatus.BAD_REQUEST
            self._send_error(status, str(exc), code="bad_request")
        except ReleaseError as exc:
            status = HTTPStatus.NOT_FOUND
            self._send_error(status, str(exc), code="not_found")
        except SimulationTimeoutError as exc:
            status = HTTPStatus.GATEWAY_TIMEOUT
            self._send_error(status, str(exc), code="simulation_timeout")
        except SimulationLimitError as exc:
            status = HTTPStatus.BAD_REQUEST
            self._send_error(status, str(exc), code="simulation_limit_exceeded")
        except AllocationError as exc:
            status = HTTPStatus.CONFLICT
            self._send_error(status, str(exc), code="conflict")
        except Exception:
            traceback.print_exc(file=sys.stderr)
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            self._send_error(status, "Internal server error.", code="internal_error")
        finally:
            if acquired_sim:
                SIM_SLOTS.release()
            self._log_json("POST", self.path, status, (time.perf_counter() - t0) * 1000.0, self.req_id, client_ip)



def create_app_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    service: Optional[CloudPilotService] = None,
) -> CloudPilotHTTPServer:
    """Creates and binds a CloudPilotHTTPServer instance."""
    return CloudPilotHTTPServer((host, port), CloudPilotRequestHandler, service=service)


def main() -> None:
    if is_production() and not os.environ.get("CLOUDPILOT_API_KEY"):
        sys.stderr.write("FATAL: CLOUDPILOT_API_KEY must be configured in production (RENDER=true).\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run CloudPilot Production Control Plane Server.")
    default_port = int(os.environ.get("PORT", 8000))
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Bind port (default: {default_port} from PORT env)",
    )
    args = parser.parse_args()

    server = create_app_server(host=args.host, port=args.port)
    actual_host, actual_port = server.server_address
    print(f"CloudPilot Control Plane listening on http://{actual_host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
