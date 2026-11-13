"""Zero-dependency HTTP REST API and live control-plane dashboard for CloudPilot."""

import argparse
import hmac
import json
import os
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from allocation.exceptions import AllocationError, InvalidResourceError, ReleaseError
from backend.service import CloudPilotService

MAX_BODY = 64 * 1024  # 64 KB
SIM_SLOTS = threading.BoundedSemaphore(2)


class PayloadTooLargeError(Exception):
    """Raised when incoming request payload exceeds MAX_BODY."""
    pass


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

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default noisy apache stderr logging
        return

    def _log_json(self, method: str, path: str, status: int, duration_ms: float) -> None:
        """Emits structured JSON access log to stdout."""
        entry = {
            "timestamp": round(time.time(), 3),
            "method": method,
            "path": path,
            "status": int(status),
            "duration_ms": round(duration_ms, 2),
        }
        try:
            sys.stdout.write(json.dumps(entry) + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        """Verifies X-API-Key against CLOUDPILOT_API_KEY environment variable if configured."""
        configured_key = os.environ.get("CLOUDPILOT_API_KEY")
        if not configured_key:
            return True
        header_key = self.headers.get("X-API-Key", "")
        return hmac.compare_digest(header_key, configured_key)

    def _read_json_body(self) -> Dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except (ValueError, TypeError):
            raise InvalidResourceError("Invalid Content-Length.")
        if length < 0:
            raise InvalidResourceError("Invalid Content-Length.")
        if length > MAX_BODY:
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
        try:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_html(HTTPStatus.OK, DASHBOARD_HTML)
            elif path == "/healthz":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
            elif path == "/api/v1/health":
                self._send_json(HTTPStatus.OK, self.server.service.get_health())
            elif path == "/api/v1/cluster":
                self._send_json(HTTPStatus.OK, self.server.service.get_cluster_status())
            else:
                status = HTTPStatus.NOT_FOUND
                self._send_json(status, {"error": f"Endpoint '{path}' not found."})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            self._send_json(status, {"error": "internal"})
        finally:
            self._log_json("GET", self.path, status, (time.perf_counter() - t0) * 1000.0)

    def do_POST(self) -> None:
        t0 = time.perf_counter()
        status = HTTPStatus.OK
        acquired_sim = False
        try:
            if not self._authorized():
                status = HTTPStatus.UNAUTHORIZED
                self._send_json(status, {"error": "Unauthorized."})
                return

            path = urlparse(self.path).path
            body = self._read_json_body()

            if path == "/api/v1/policy":
                if "policy" not in body:
                    raise InvalidResourceError("Missing required field 'policy'.")
                res = self.server.service.set_policy(body["policy"])
                status = HTTPStatus.OK
                self._send_json(status, res)

            elif path == "/api/v1/allocate":
                for req_field in ("workload_id", "cpu", "ram"):
                    if req_field not in body:
                        raise InvalidResourceError(f"Missing required field '{req_field}'.")
                res = self.server.service.allocate_workload(
                    workload_id=body["workload_id"],
                    cpu=body["cpu"],
                    ram=body["ram"],
                    arrival_time=body.get("arrival_time", 0.0),
                    duration=body.get("duration"),
                )
                status = HTTPStatus.CREATED if res["placed"] else HTTPStatus.CONFLICT
                self._send_json(status, res)

            elif path == "/api/v1/release":
                if "workload_id" not in body:
                    raise InvalidResourceError("Missing required field 'workload_id'.")
                res = self.server.service.release_workload(body["workload_id"])
                status = HTTPStatus.OK
                self._send_json(status, res)

            elif path == "/api/v1/reset":
                res = self.server.service.reset_cluster()
                status = HTTPStatus.OK
                self._send_json(status, res)

            elif path == "/api/v1/simulate":
                if not SIM_SLOTS.acquire(blocking=False):
                    status = HTTPStatus.TOO_MANY_REQUESTS
                    self._send_json(
                        status, {"error": "Simulation queue full. Try again shortly."}
                    )
                    return
                acquired_sim = True
                res = self.server.service.run_simulation(
                    policy=body.get("policy"),
                    num_workloads=body.get("num_workloads", 200),
                    seed=body.get("seed", 42),
                )
                status = HTTPStatus.OK
                self._send_json(status, res)

            else:
                status = HTTPStatus.NOT_FOUND
                self._send_json(status, {"error": f"Endpoint '{path}' not found."})

        except PayloadTooLargeError as exc:
            status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            self._send_json(status, {"error": str(exc)})
        except InvalidResourceError as exc:
            status = HTTPStatus.BAD_REQUEST
            self._send_json(status, {"error": str(exc)})
        except ReleaseError as exc:
            status = HTTPStatus.NOT_FOUND
            self._send_json(status, {"error": str(exc)})
        except AllocationError as exc:
            status = HTTPStatus.CONFLICT
            self._send_json(status, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            self._send_json(status, {"error": "internal"})
        finally:
            if acquired_sim:
                SIM_SLOTS.release()
            self._log_json("POST", self.path, status, (time.perf_counter() - t0) * 1000.0)


def create_app_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    service: Optional[CloudPilotService] = None,
) -> CloudPilotHTTPServer:
    """Creates and binds a CloudPilotHTTPServer instance."""
    return CloudPilotHTTPServer((host, port), CloudPilotRequestHandler, service=service)


def main() -> None:
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
