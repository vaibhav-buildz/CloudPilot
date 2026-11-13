"""Standalone CLI smoke test for verifying the CloudPilot Control Plane REST API."""

import argparse
import json
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

from backend.server import create_app_server


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any], float]:
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        status = resp.status
        body = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return status, body, round(elapsed_ms, 2)


def run_smoke_suite(base_url: Optional[str] = None) -> bool:
    """Runs a 7-step production smoke test against a live or ephemeral server."""
    ephemeral_server = None
    server_thread = None

    if base_url is None:
        ephemeral_server = create_app_server(host="127.0.0.1", port=0)
        host, port = ephemeral_server.server_address
        base_url = f"http://{host}:{port}"
        server_thread = threading.Thread(target=ephemeral_server.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.05)

    try:
        print(f"Running CloudPilot Production Smoke Suite against {base_url} ...")

        # 1. Healthcheck
        st, body, ms = _request_json(base_url, "GET", "/api/v1/health")
        assert st == 200 and body["status"] == "healthy", f"Healthcheck failed: {body}"
        print(f"  [PASS] GET  /api/v1/health   -> {st} ({ms} ms)")

        # 2. Switch Policy
        st, body, ms = _request_json(base_url, "POST", "/api/v1/policy", {"policy": "predictive_ar"})
        assert st == 200 and body["active_policy"] == "predictive_ar"
        print(f"  [PASS] POST /api/v1/policy   -> {st} ({ms} ms) policy={body['active_policy']}")

        # 3. Allocate Workload
        st, body, ms = _request_json(
            base_url,
            "POST",
            "/api/v1/allocate",
            {"workload_id": "smoke-job-01", "cpu": 4.0, "ram": 8.0},
        )
        assert st == 201 and body["placed"] is True
        print(f"  [PASS] POST /api/v1/allocate -> {st} ({ms} ms) server={body['server_id']}")

        # 4. Cluster Telemetry Check
        st, body, ms = _request_json(base_url, "GET", "/api/v1/cluster")
        assert st == 200 and body["active_workloads_count"] == 1
        print(f"  [PASS] GET  /api/v1/cluster  -> {st} ({ms} ms) active_jobs={body['active_workloads_count']}")

        # 5. Release Workload
        st, body, ms = _request_json(
            base_url, "POST", "/api/v1/release", {"workload_id": "smoke-job-01"}
        )
        assert st == 200 and body["released"] is True
        print(f"  [PASS] POST /api/v1/release  -> {st} ({ms} ms)")

        # 6. On-Demand Trace Simulation
        st, body, ms = _request_json(
            base_url,
            "POST",
            "/api/v1/simulate",
            {"policy": "predictive_ar", "num_workloads": 100, "seed": 42},
        )
        assert st == 200 and body["placed_count"] > 0
        print(
            f"  [PASS] POST /api/v1/simulate -> {st} ({ms} ms) placed={body['placed_count']}/100, SLA={body['sla_efficiency_score']}"
        )

        # 7. Reset Cluster
        st, body, ms = _request_json(base_url, "POST", "/api/v1/reset", {})
        assert st == 200 and body["reset"] is True
        print(f"  [PASS] POST /api/v1/reset    -> {st} ({ms} ms)")

        print("Smoke Suite Status: ALL 7 CHECKS PASSED.")
        return True
    finally:
        if ephemeral_server is not None:
            ephemeral_server.shutdown()
            ephemeral_server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CloudPilot API Smoke Test.")
    parser.add_argument(
        "--url",
        default=None,
        help="Optional target URL (e.g. http://127.0.0.1:8000). Spawns ephemeral server if omitted.",
    )
    args = parser.parse_args()
    ok = run_smoke_suite(base_url=args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

