"""Unit and live HTTP integration tests for CloudPilot Control Plane API."""

import http.client
import io
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

from allocation.exceptions import InvalidResourceError, ReleaseError
from allocation.workload import Workload
from backend.server import SIM_SLOTS, _sim_rate_history, _sim_rate_lock, create_app_server
import backend.store as store
from backend.service import CloudPilotService
from backend.smoke_test import run_smoke_suite
from ml.optimizer import PredictiveAllocator, UCBBanditAllocator


class TestCloudPilotService(unittest.TestCase):
    """Unit tests for CloudPilotService business logic and state management."""

    def setUp(self) -> None:
        self.specs = [("s1", 8.0, 16.0), ("s2", 16.0, 32.0)]
        self.service = CloudPilotService(server_specs=self.specs, default_policy="predictive_ar")

    def test_initial_health_and_cluster_status(self) -> None:
        health = self.service.get_health()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["active_policy"], "predictive_ar")
        self.assertEqual(health["total_servers"], 2)
        self.assertEqual(health["active_workloads"], 0)
        self.assertGreaterEqual(health["uptime_sec"], 0.0)

        status = self.service.get_cluster_status()
        self.assertEqual(status["servers_in_use"], 0)
        self.assertEqual(status["cpu_utilization_pct"], 0.0)
        self.assertEqual(status["ram_utilization_pct"], 0.0)
        self.assertEqual(len(status["servers"]), 2)

    def test_set_policy_valid_and_invalid(self) -> None:
        for policy in self.service.available_policies:
            res = self.service.set_policy(policy)
            self.assertEqual(res["active_policy"], policy)
            self.assertEqual(self.service.active_policy, policy)

        with self.assertRaises(InvalidResourceError):
            self.service.set_policy("non_existent_policy")

    def test_allocate_and_release_lifecycle(self) -> None:
        alloc_res = self.service.allocate_workload(
            workload_id="job-1",
            cpu=4.0,
            ram=8.0,
        )
        self.assertTrue(alloc_res["placed"])
        self.assertEqual(alloc_res["workload_id"], "job-1")

        status = self.service.get_cluster_status()
        self.assertEqual(status["placed_total"], 1)
        self.assertEqual(status["servers_in_use"], 1)
        self.assertEqual(status["active_workloads_count"], 1)
        self.assertGreater(status["cpu_utilization_pct"], 0.0)

        # Duplicate ID rejection
        with self.assertRaises(InvalidResourceError):
            self.service.allocate_workload("job-1", cpu=1.0, ram=1.0)

        # Release
        rel_res = self.service.release_workload("job-1")
        self.assertTrue(rel_res["released"])
        self.assertEqual(rel_res["workload_id"], "job-1")

        post_status = self.service.get_cluster_status()
        self.assertEqual(post_status["servers_in_use"], 0)
        self.assertEqual(post_status["active_workloads_count"], 0)
        self.assertEqual(post_status["cpu_utilization_pct"], 0.0)

        # Release unknown ID
        with self.assertRaises(ReleaseError):
            self.service.release_workload("unknown-job")

    def test_allocate_sla_rejection_when_oversized(self) -> None:
        res = self.service.allocate_workload("huge-job", cpu=999.0, ram=999.0)
        self.assertFalse(res["placed"])
        self.assertIn("SLA rejection", res["reason"])
        status = self.service.get_cluster_status()
        self.assertEqual(status["rejected_total"], 1)

    def test_reset_cluster(self) -> None:
        self.service.allocate_workload("job-reset", cpu=2.0, ram=4.0)
        self.assertEqual(self.service.get_cluster_status()["active_workloads_count"], 1)
        res = self.service.reset_cluster()
        self.assertTrue(res["reset"])
        status = self.service.get_cluster_status()
        self.assertEqual(status["active_workloads_count"], 0)
        self.assertEqual(status["servers_in_use"], 0)
        self.assertEqual(status["placed_total"], 0)

    def test_run_simulation(self) -> None:
        sim_res = self.service.run_simulation(policy="predictive_ewma", num_workloads=30, seed=42)
        self.assertEqual(sim_res["policy_key"], "predictive_ewma")
        self.assertEqual(sim_res["total_workloads"], 30)
        self.assertGreaterEqual(sim_res["placed_count"], 1)
        self.assertGreaterEqual(sim_res["sla_efficiency_score"], 0.0)
        self.assertLessEqual(sim_res["sla_efficiency_score"], 100.0)

    def test_concurrent_allocations_thread_safety(self) -> None:
        service = CloudPilotService(
            server_specs=[(f"srv-{i}", 64.0, 128.0) for i in range(10)],
            default_policy="first_fit",
        )
        num_threads = 6
        workloads_per_thread = 10

        def worker(thread_idx: int) -> None:
            for w_idx in range(workloads_per_thread):
                wid = f"th-{thread_idx}-job-{w_idx}"
                service.allocate_workload(workload_id=wid, cpu=1.0, ram=2.0)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        status = service.get_cluster_status()
        self.assertEqual(status["placed_total"], num_threads * workloads_per_thread)
        self.assertEqual(status["active_workloads_count"], num_threads * workloads_per_thread)


class TestCloudPilotHTTPServer(unittest.TestCase):
    """Live HTTP integration tests on ephemeral port (127.0.0.1:0)."""

    server: Any
    server_thread: threading.Thread
    host: str
    port: int
    service: CloudPilotService

    @classmethod
    def setUpClass(cls) -> None:
        specs = [("srv-alpha", 16.0, 32.0), ("srv-beta", 32.0, 64.0)]
        cls.service = CloudPilotService(server_specs=specs, default_policy="predictive_ar")
        cls.server = create_app_server(host="127.0.0.1", port=0, service=cls.service)
        cls.host, cls.port = cls.server.server_address

        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        # Brief pause to ensure socket is accepting connections
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2.0)

    def setUp(self) -> None:
        # Reset cluster state before each HTTP test
        self.service.reset_cluster()

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Any]:
        url = f"http://{self.host}:{self.port}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req_headers = {"Content-Type": "application/json"} if body is not None else {}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8")
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return status, json.loads(raw)
                return status, raw
        except urllib.error.HTTPError as exc:
            err_raw = exc.read().decode("utf-8")
            content_type = exc.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    return exc.code, json.loads(err_raw)
                except Exception:
                    return exc.code, err_raw
            return exc.code, err_raw

    def test_get_dashboard_html(self) -> None:
        status, content = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("CloudPilot ML Resource Allocation Control Plane", content)
        self.assertIn("<!DOCTYPE html>", content)

    def test_get_health(self) -> None:
        status, data = self._request("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("predictive_ar", data["available_policies"])

    def test_get_cluster(self) -> None:
        status, data = self._request("GET", "/api/v1/cluster")
        self.assertEqual(status, 200)
        self.assertEqual(data["total_servers"], 2)
        self.assertEqual(data["servers_in_use"], 0)

    def test_post_policy_switch(self) -> None:
        status, data = self._request("POST", "/api/v1/policy", {"policy": "best_fit"})
        self.assertEqual(status, 200)
        self.assertEqual(data["active_policy"], "best_fit")

        # Invalid policy returns 400 Bad Request
        status_err, data_err = self._request("POST", "/api/v1/policy", {"policy": "invalid_policy"})
        self.assertEqual(status_err, 400)
        self.assertIn("error", data_err)

    def test_post_allocate_success_and_conflict(self) -> None:
        # Successful placement -> 201 Created
        status, data = self._request("POST", "/api/v1/allocate", {
            "workload_id": "test-job-1",
            "cpu": 4.0,
            "ram": 8.0,
        })
        self.assertEqual(status, 201)
        self.assertTrue(data["placed"])
        self.assertEqual(data["workload_id"], "test-job-1")

        # Oversized job beyond capacity -> 409 Conflict (SLA rejection)
        status_conflict, data_conflict = self._request("POST", "/api/v1/allocate", {
            "workload_id": "huge-job",
            "cpu": 999.0,
            "ram": 999.0,
        })
        self.assertEqual(status_conflict, 409)
        self.assertFalse(data_conflict["placed"])

        # Invalid inputs -> 400 Bad Request
        status_bad, data_bad = self._request("POST", "/api/v1/allocate", {
            "workload_id": "bad-job",
            "cpu": -5.0,
            "ram": 8.0,
        })
        self.assertEqual(status_bad, 400)
        self.assertIn("error", data_bad)

    def test_post_release_lifecycle(self) -> None:
        # Allocate first
        self._request("POST", "/api/v1/allocate", {
            "workload_id": "rel-job",
            "cpu": 4.0,
            "ram": 8.0,
        })

        # Successful release -> 200 OK
        status, data = self._request("POST", "/api/v1/release", {"workload_id": "rel-job"})
        self.assertEqual(status, 200)
        self.assertTrue(data["released"])

        # Releasing non-existent workload -> 404 Not Found
        status_404, data_404 = self._request("POST", "/api/v1/release", {"workload_id": "ghost-job"})
        self.assertEqual(status_404, 404)
        self.assertIn("error", data_404)

    def test_post_reset(self) -> None:
        self._request("POST", "/api/v1/allocate", {
            "workload_id": "reset-job",
            "cpu": 2.0,
            "ram": 4.0,
        })
        status, data = self._request("POST", "/api/v1/reset", {})
        self.assertEqual(status, 200)
        self.assertTrue(data["reset"])
        self.assertEqual(data["active_workloads_count"], 0)

    def test_post_simulate(self) -> None:
        status, data = self._request("POST", "/api/v1/simulate", {
            "policy": "first_fit",
            "num_workloads": 50,
            "seed": 42,
        })
        self.assertEqual(status, 200)
        self.assertEqual(data["policy_key"], "first_fit")
        self.assertEqual(data["total_workloads"], 50)
        self.assertIn("placed_count", data)
        self.assertIn("sla_efficiency_score", data)

    def test_not_found_routes(self) -> None:
        status_get, data_get = self._request("GET", "/api/v1/unknown")
        self.assertEqual(status_get, 404)
        self.assertIn("error", data_get)

        status_post, data_post = self._request("POST", "/api/v1/unknown", {})
        self.assertEqual(status_post, 404)
        self.assertIn("error", data_post)

    def test_smoke_suite_executes_successfully(self) -> None:
        # Verify run_smoke_suite against live server URL
        url = f"http://{self.host}:{self.port}"
        ok = run_smoke_suite(base_url=url)
        self.assertTrue(ok)

    def test_get_healthz(self) -> None:
        status, data = self._request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "ok")

    def test_payload_too_large_413(self) -> None:
        conn = http.client.HTTPConnection(self.host, self.port)
        big_body = b"{" + b'"k":"v",' * 10000 + b'"end":"val"}'
        self.assertGreater(len(big_body), 64 * 1024)
        conn.putrequest("POST", "/api/v1/allocate")
        conn.putheader("Content-Length", str(len(big_body)))
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()
        conn.send(big_body)
        resp = conn.getresponse()
        self.assertEqual(resp.status, 413)
        conn.close()

    def test_invalid_content_length_400(self) -> None:
        conn = http.client.HTTPConnection(self.host, self.port)
        conn.putrequest("POST", "/api/v1/allocate")
        conn.putheader("Content-Length", "not-a-number")
        conn.endheaders()
        resp = conn.getresponse()
        self.assertEqual(resp.status, 400)
        conn.close()

    def test_auth_401_when_api_key_configured(self) -> None:
        os.environ["CLOUDPILOT_API_KEY"] = "secret-test-token"
        try:
            # Missing X-API-Key -> 401
            st_missing, _ = self._request("POST", "/api/v1/reset", {})
            self.assertEqual(st_missing, 401)

            # Invalid X-API-Key -> 401
            st_wrong, _ = self._request(
                "POST", "/api/v1/reset", {}, headers={"X-API-Key": "wrong-token"}
            )
            self.assertEqual(st_wrong, 401)

            # Valid X-API-Key -> 200
            st_valid, res = self._request(
                "POST", "/api/v1/reset", {}, headers={"X-API-Key": "secret-test-token"}
            )
            self.assertEqual(st_valid, 200)
            self.assertTrue(res.get("reset"))
        finally:
            os.environ.pop("CLOUDPILOT_API_KEY", None)

    def test_cors_headers(self) -> None:
        conn = http.client.HTTPConnection(self.host, self.port)
        conn.request("GET", "/api/v1/health")
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 200)
        self.assertIn("Access-Control-Allow-Origin", resp.headers)
        self.assertIn("Access-Control-Allow-Methods", resp.headers)
        self.assertIn("Access-Control-Allow-Headers", resp.headers)
        # Default must be https://cloud-pilot-gamma.vercel.app with NO wildcard
        self.assertEqual(
            resp.headers.get("Access-Control-Allow-Origin"),
            "https://cloud-pilot-gamma.vercel.app",
        )
        conn.close()

    def test_options_204(self) -> None:
        conn = http.client.HTTPConnection(self.host, self.port)
        conn.request("OPTIONS", "/api/v1/simulate")
        resp = conn.getresponse()
        body = resp.read()
        self.assertEqual(resp.status, 204)
        self.assertEqual(len(body), 0)
        self.assertEqual(resp.headers.get("Content-Length"), "0")
        self.assertIn("Access-Control-Allow-Origin", resp.headers)
        self.assertIn("Access-Control-Allow-Methods", resp.headers)
        conn.close()

    def test_selective_auth_scope(self) -> None:
        os.environ["CLOUDPILOT_API_KEY"] = "test-secret-key"
        try:
            # Public GET endpoints do NOT require key
            st, data = self._request("GET", "/healthz")
            self.assertEqual(st, 200)
            st, data = self._request("GET", "/health")
            self.assertEqual(st, 200)
            self.assertEqual(data.get("status"), "ok")

            st, data = self._request("GET", "/ready")
            self.assertEqual(st, 200)
            self.assertEqual(data, {"status": "ok"})

            # Data-reading endpoints require key when CLOUDPILOT_API_KEY is configured
            st, _ = self._request("GET", "/api/v1/health")
            self.assertEqual(st, 401)
            st, _ = self._request("GET", "/api/v1/health", headers={"X-API-Key": "test-secret-key"})
            self.assertEqual(st, 200)

            st, _ = self._request("GET", "/api/v1/cluster")
            self.assertEqual(st, 401)
            st, _ = self._request("GET", "/api/v1/cluster", headers={"X-API-Key": "test-secret-key"})
            self.assertEqual(st, 200)

            st, _ = self._request("GET", "/api/v1/history")
            self.assertEqual(st, 401)
            st, _ = self._request("GET", "/api/v1/history", headers={"X-API-Key": "test-secret-key"})
            self.assertEqual(st, 200)

            # Protected diagnostic GET endpoint requires key
            st, _ = self._request("GET", "/api/v1/store/status")
            self.assertEqual(st, 401)
            st, diag = self._request(
                "GET", "/api/v1/store/status", headers={"X-API-Key": "test-secret-key"}
            )
            self.assertEqual(st, 200)
            self.assertIn("url_set", diag)
            self.assertIn("key_set", diag)
            self.assertIn("key_type", diag)
            self.assertIn("url_host", diag)

            # All write/simulate endpoints require key when CLOUDPILOT_API_KEY is configured
            st, _ = self._request(
                "POST", "/api/v1/simulate", {"policy": "first_fit", "num_workloads": 10}
            )
            self.assertEqual(st, 401)

            st, res = self._request(
                "POST",
                "/api/v1/simulate",
                {"policy": "first_fit", "num_workloads": 10},
                headers={"X-API-Key": "test-secret-key"},
            )
            self.assertEqual(st, 200)
            self.assertIn("placed_count", res)

            # Protected POST endpoints require key
            st, _ = self._request("POST", "/api/v1/policy", {"policy": "best_fit"})
            self.assertEqual(st, 401)
            st, _ = self._request(
                "POST", "/api/v1/allocate", {"workload_id": "w1", "cpu": 1.0, "ram": 2.0}
            )
            self.assertEqual(st, 401)
            st, _ = self._request("POST", "/api/v1/release", {"workload_id": "w1"})
            self.assertEqual(st, 401)
            st, _ = self._request("POST", "/api/v1/reset", {})
            self.assertEqual(st, 401)


            # Providing valid key unlocks protected endpoints
            st, _ = self._request(
                "POST",
                "/api/v1/policy",
                {"policy": "best_fit"},
                headers={"X-API-Key": "test-secret-key"},
            )
            self.assertEqual(st, 200)
        finally:
            os.environ.pop("CLOUDPILOT_API_KEY", None)

    def test_simulate_rate_limit_429(self) -> None:
        client_ip = "127.0.0.1"
        with _sim_rate_lock:
            # Pre-fill 10 recent timestamps
            _sim_rate_history[client_ip] = [time.time()] * 10
        try:
            status, data = self._request(
                "POST", "/api/v1/simulate", {"policy": "first_fit", "num_workloads": 10}
            )
            self.assertEqual(status, 429)
            self.assertIn("Rate limit exceeded", data.get("error", ""))
        finally:
            with _sim_rate_lock:
                _sim_rate_history.clear()

    def test_get_history(self) -> None:
        status, data = self._request("GET", "/api/v1/history")
        self.assertEqual(status, 200)
        self.assertIsInstance(data, list)

    def test_get_store_status_unprotected_when_no_api_key(self) -> None:
        status, data = self._request("GET", "/api/v1/store/status")
        self.assertEqual(status, 200)
        self.assertIn("url_set", data)
        self.assertIn("key_set", data)
        self.assertIn("key_type", data)
        self.assertIn("url_host", data)
        self.assertIn("last_save_error", data)

    def test_simulate_saves_to_supabase_exact_format(self) -> None:
        called_event = threading.Event()
        captured_requests = []
        original_urlopen = urllib.request.urlopen

        def mock_urlopen_fn(req, *args, **kwargs):
            url = getattr(req, "full_url", str(req))
            if "supabase" in url:
                captured_requests.append(req)
                called_event.set()
                mock_resp = MagicMock()
                mock_resp.status = 201
                mock_resp.read.return_value = b'{"status": "ok"}'
                return mock_resp
            return original_urlopen(req, *args, **kwargs)

        old_url = os.environ.get("SUPABASE_URL")
        old_key = os.environ.get("SUPABASE_SERVICE_KEY")
        secret_key = "sb_secret_exact_format_key_999"
        try:
            os.environ["SUPABASE_URL"] = "https://mock-supabase.supabase.co"
            os.environ["SUPABASE_SERVICE_KEY"] = secret_key

            with patch("urllib.request.urlopen", side_effect=mock_urlopen_fn):
                status, res = self._request(
                    "POST",
                    "/api/v1/simulate",
                    {"policy": "first_fit", "num_workloads": 50, "seed": 777},
                )
                self.assertEqual(status, 200)
                self.assertTrue(
                    called_event.wait(timeout=3.0),
                    "store.save urlopen was not called within timeout",
                )

            req = captured_requests[0]
            # 1. Assert exact request URL path
            parsed = urllib.parse.urlparse(req.full_url)
            self.assertEqual(parsed.path, "/rest/v1/simulation_runs")

            # 2. Assert headers: apikey present, Authorization absent for sb_secret_
            header_map = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("apikey", header_map)
            self.assertEqual(header_map["apikey"], secret_key)
            self.assertNotIn("authorization", header_map)

            # 3. Assert exact JSON body schema: {"policy", "num_workloads", "seed", "result"}
            body = json.loads(req.data.decode("utf-8"))
            self.assertEqual(set(body.keys()), {"policy", "num_workloads", "seed", "result"})
            self.assertEqual(body["policy"], "first_fit")
            self.assertEqual(body["num_workloads"], 50)
            self.assertEqual(body["seed"], 777)
            self.assertIsInstance(body["result"], dict)
            self.assertIn("placed_count", body["result"])
            self.assertIn("sla_efficiency_score", body["result"])

            # Assert NO extra top-level keys or flattened metrics
            self.assertNotIn("algorithm", body)
            self.assertNotIn("total_workloads", body)
            self.assertNotIn("placed_count", body)
            self.assertNotIn("created_at", body)
            self.assertNotIn("sla_efficiency_score", body)
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            else:
                os.environ.pop("SUPABASE_URL", None)
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key
            else:
                os.environ.pop("SUPABASE_SERVICE_KEY", None)

    def test_simulation_concurrency_429(self) -> None:
        self.assertTrue(SIM_SLOTS.acquire(blocking=False))
        self.assertTrue(SIM_SLOTS.acquire(blocking=False))
        try:
            status, data = self._request(
                "POST", "/api/v1/simulate", {"policy": "first_fit", "num_workloads": 20}
            )
            self.assertEqual(status, 429)
            self.assertIn("Simulation queue full", data.get("error", ""))
        finally:
            SIM_SLOTS.release()
            SIM_SLOTS.release()

    def test_ucb_arm_state_consistency(self) -> None:
        ucb = UCBBanditAllocator()
        pred_arm = ucb.arms[0]
        self.assertIsInstance(pred_arm, PredictiveAllocator)

        w1 = Workload("w-arm-consistent", 2.0, 4.0, arrival_time=0.0)
        ucb.observe(w1)
        self.assertEqual(pred_arm._last_observed_id, "w-arm-consistent")
        self.assertEqual(pred_arm._win_arrivals, 1)

        # Re-observing same workload does not duplicate count
        ucb.observe(w1)
        self.assertEqual(pred_arm._win_arrivals, 1)


class TestSupabaseStore(unittest.TestCase):
    """Unit tests for Supabase store helper (backend/store.py)."""

    def test_headers_with_jwt_token(self) -> None:
        jwt_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy.sig"
        headers = store._headers(jwt_key)
        self.assertEqual(headers["apikey"], jwt_key)
        self.assertEqual(headers["Authorization"], f"Bearer {jwt_key}")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_headers_with_sb_secret_token(self) -> None:
        sb_secret = "sb_secret_abcd1234efgh5678"
        headers = store._headers(sb_secret)
        self.assertEqual(headers["apikey"], sb_secret)
        # sb_secret_ must NOT include Authorization header
        self.assertNotIn("Authorization", headers)
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_silent_noop_when_env_vars_missing(self) -> None:
        old_url = os.environ.pop("SUPABASE_URL", None)
        old_key = os.environ.pop("SUPABASE_SERVICE_KEY", None)
        try:
            # save should silently no-op without raising
            store.save("simulation_runs", {"test": 123})
            # recent should return empty list
            rows = store.recent("simulation_runs")
            self.assertEqual(rows, [])
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key

    def test_diagnose_missing_env(self) -> None:
        old_url = os.environ.pop("SUPABASE_URL", None)
        old_key = os.environ.pop("SUPABASE_SERVICE_KEY", None)
        try:
            diag = store.diagnose()
            self.assertFalse(diag["url_set"])
            self.assertFalse(diag["key_set"])
            self.assertEqual(diag["key_type"], "other")
            self.assertIsNone(diag["url_host"])
            self.assertIsNone(diag["status_code"])
            self.assertIn("Missing", diag["response_body"])
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key

    @patch("urllib.request.urlopen")
    def test_diagnose_with_mocked_urlopen_success(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_resp.read.return_value = b'{"status": "ok", "id": 42}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        old_url = os.environ.get("SUPABASE_URL")
        old_key = os.environ.get("SUPABASE_SERVICE_KEY")
        secret_value = "sb_secret_super_secret_key_12345"
        try:
            os.environ["SUPABASE_URL"] = "https://my-supabase-db.supabase.co"
            os.environ["SUPABASE_SERVICE_KEY"] = secret_value

            diag = store.diagnose()

            self.assertTrue(diag["url_set"])
            self.assertTrue(diag["key_set"])
            self.assertEqual(diag["key_type"], "sb_secret")
            self.assertEqual(diag["url_host"], "my-supabase-db.supabase.co")
            self.assertEqual(diag["status_code"], 201)
            self.assertIn('"id": 42', diag["response_body"])

            # Verify request headers and URL
            mock_urlopen.assert_called_once()
            called_req = mock_urlopen.call_args[0][0]
            self.assertEqual(
                called_req.full_url,
                "https://my-supabase-db.supabase.co/rest/v1/simulation_runs",
            )
            # In urllib.request.Request, headers are capitalized
            self.assertEqual(called_req.headers.get("Apikey"), secret_value)
            # sb_secret_ should NOT have Authorization header
            self.assertNotIn("Authorization", called_req.headers)

            # CRITICAL: Verify the secret key is NEVER exposed anywhere in the output
            serialized = json.dumps(diag)
            self.assertNotIn(secret_value, serialized)
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            else:
                os.environ.pop("SUPABASE_URL", None)
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key
            else:
                os.environ.pop("SUPABASE_SERVICE_KEY", None)

    @patch("urllib.request.urlopen")
    def test_diagnose_with_mocked_urlopen_http_error(self, mock_urlopen: MagicMock) -> None:
        error_body = b'{"message": "Invalid JWT token"}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://my-supabase-db.supabase.co/rest/v1/simulation_runs",
            401,
            "Unauthorized",
            {},
            io.BytesIO(error_body),
        )

        old_url = os.environ.get("SUPABASE_URL")
        old_key = os.environ.get("SUPABASE_SERVICE_KEY")
        jwt_value = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy"
        try:
            os.environ["SUPABASE_URL"] = "https://my-supabase-db.supabase.co"
            os.environ["SUPABASE_SERVICE_KEY"] = jwt_value

            diag = store.diagnose()

            self.assertTrue(diag["url_set"])
            self.assertTrue(diag["key_set"])
            self.assertEqual(diag["key_type"], "jwt")
            self.assertEqual(diag["url_host"], "my-supabase-db.supabase.co")
            self.assertEqual(diag["status_code"], 401)
            self.assertIn("Invalid JWT token", diag["response_body"])

            # Verify key is never exposed
            serialized = json.dumps(diag)
            self.assertNotIn(jwt_value, serialized)
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            else:
                os.environ.pop("SUPABASE_URL", None)
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key
            else:
                os.environ.pop("SUPABASE_SERVICE_KEY", None)

    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("urllib.request.urlopen")
    def test_post_error_logs_json_and_records_last_save_error(
        self, mock_urlopen: MagicMock, mock_stdout: io.StringIO
    ) -> None:
        err_msg = b'{"code":"PGRST204","message":"Could not find column"}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://my-supabase-db.supabase.co/rest/v1/simulation_runs",
            400,
            "Bad Request",
            {},
            io.BytesIO(err_msg),
        )

        old_url = os.environ.get("SUPABASE_URL")
        old_key = os.environ.get("SUPABASE_SERVICE_KEY")
        secret_key = "sb_secret_super_secret_for_error_test"
        try:
            os.environ["SUPABASE_URL"] = "https://my-supabase-db.supabase.co"
            os.environ["SUPABASE_SERVICE_KEY"] = secret_key

            store._post("simulation_runs", {"test": 123})

            # Verify JSON line printed to stdout
            output = mock_stdout.getvalue()
            self.assertIn('"store_error": true', output)
            self.assertIn('"table": "simulation_runs"', output)
            self.assertIn('"status": 400', output)
            self.assertIn("PGRST204", output)
            self.assertNotIn(secret_key, output)

            # Verify diagnose() includes last_save_error
            diag = store.diagnose()
            self.assertIn("last_save_error", diag)
            self.assertIsNotNone(diag["last_save_error"])
            self.assertEqual(diag["last_save_error"]["status"], 400)
            self.assertIn("PGRST204", diag["last_save_error"]["body"])
            self.assertNotIn(secret_key, diag["last_save_error"]["body"])
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            else:
                os.environ.pop("SUPABASE_URL", None)
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key
            else:
                os.environ.pop("SUPABASE_SERVICE_KEY", None)

    def test_check_ready_standalone_and_connected(self) -> None:
        old_url = os.environ.pop("SUPABASE_URL", None)
        old_key = os.environ.pop("SUPABASE_SERVICE_KEY", None)
        try:
            # Standalone mode when env vars absent
            is_ready, msg, code = store.check_ready()
            self.assertTrue(is_ready)
            self.assertEqual(msg, "standalone")
            self.assertIsNone(code)
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key

    @patch("urllib.request.urlopen")
    def test_check_ready_failure_returns_false(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        old_url = os.environ.get("SUPABASE_URL")
        old_key = os.environ.get("SUPABASE_SERVICE_KEY")
        try:
            os.environ["SUPABASE_URL"] = "https://mock-db.supabase.co"
            os.environ["SUPABASE_SERVICE_KEY"] = "mock-key"
            is_ready, msg, code = store.check_ready()
            self.assertFalse(is_ready)
            self.assertIn("Connection refused", msg)
            self.assertIsNone(code)
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            else:
                os.environ.pop("SUPABASE_URL", None)
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key
            else:
                os.environ.pop("SUPABASE_SERVICE_KEY", None)


class TestProductionReliabilityAndSecurity(unittest.TestCase):
    """Tests for Production-Readiness pass: auth, validation, health/ready, db resilience."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_app_server(host="127.0.0.1", port=0)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        conn = http.client.HTTPConnection(self.host, self.port)
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        data = json.dumps(body) if body is not None else None
        conn.request(method, path, data, h)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return resp.status, parsed

    def test_health_endpoint_happy_path(self) -> None:
        st, data = self._request("GET", "/health")
        self.assertEqual(st, 200)
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("uptime_sec", data)

    def test_ready_endpoint_standalone(self) -> None:
        old_url = os.environ.pop("SUPABASE_URL", None)
        old_key = os.environ.pop("SUPABASE_SERVICE_KEY", None)
        try:
            st, data = self._request("GET", "/ready")
            self.assertEqual(st, 200)
            self.assertEqual(data, {"status": "ok"})
        finally:
            if old_url is not None:
                os.environ["SUPABASE_URL"] = old_url
            if old_key is not None:
                os.environ["SUPABASE_SERVICE_KEY"] = old_key

    @patch("backend.store.check_ready")
    def test_ready_endpoint_db_failure_returns_503(self, mock_check_ready: MagicMock) -> None:
        mock_check_ready.return_value = (False, "HTTP 500", 500)
        st, data = self._request("GET", "/ready")
        self.assertEqual(st, 503)
        self.assertEqual(data, {"status": "not ok"})

    def test_production_fail_closed_without_api_key(self) -> None:
        old_render = os.environ.get("RENDER")
        old_key = os.environ.get("CLOUDPILOT_API_KEY")
        try:
            os.environ["RENDER"] = "true"
            if "CLOUDPILOT_API_KEY" in os.environ:
                del os.environ["CLOUDPILOT_API_KEY"]

            # Health remains accessible
            st, data = self._request("GET", "/health")
            self.assertEqual(st, 200)

            # All non-health endpoints fail-closed with 503
            st, data = self._request("GET", "/ready")
            self.assertEqual(st, 503)
            self.assertEqual(data.get("code"), "production_auth_misconfigured")

            st, data = self._request("GET", "/api/v1/cluster")
            self.assertEqual(st, 503)
            self.assertEqual(data.get("code"), "production_auth_misconfigured")

            st, data = self._request("POST", "/api/v1/simulate", {"num_workloads": 10})
            self.assertEqual(st, 503)
            self.assertEqual(data.get("code"), "production_auth_misconfigured")
        finally:
            if old_render is not None:
                os.environ["RENDER"] = old_render
            else:
                os.environ.pop("RENDER", None)
            if old_key is not None:
                os.environ["CLOUDPILOT_API_KEY"] = old_key

    def test_spoofed_x_forwarded_for_cannot_bypass_rate_limit(self) -> None:
        trusted_proxy_ip = "198.51.100.99"
        headers_base = {"Content-Type": "application/json"}

        # Exhaust rate limit of 10 simulations per minute for trusted_proxy_ip
        for i in range(10):
            spoofed_xff = f"spoofed-{i}.example.com, {trusted_proxy_ip}"
            hdrs = dict(headers_base)
            hdrs["X-Forwarded-For"] = spoofed_xff
            st, _ = self._request("POST", "/api/v1/simulate", {"policy": "first_fit", "num_workloads": 5}, headers=hdrs)
            self.assertEqual(st, 200)

        # 11th request with a brand new spoofed client IP prepended must still be rejected
        hdrs = dict(headers_base)
        hdrs["X-Forwarded-For"] = f"completely-new-spoofed.com, {trusted_proxy_ip}"
        st_blocked, data_blocked = self._request(
            "POST",
            "/api/v1/simulate",
            {"policy": "first_fit", "num_workloads": 5},
            headers=hdrs,
        )
        self.assertEqual(st_blocked, 429)
        self.assertEqual(data_blocked.get("code"), "too_many_requests")

    def test_simulation_concurrency_slots_busy_immediate_503_retry_after(self) -> None:
        from backend.server import SIM_SLOTS
        # Acquire both slots to simulate complete saturation
        slot1 = SIM_SLOTS.acquire(blocking=False)
        slot2 = SIM_SLOTS.acquire(blocking=False)
        self.assertTrue(slot1)
        self.assertTrue(slot2)
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5.0)
            body = json.dumps({"policy": "first_fit", "num_workloads": 10})
            conn.request("POST", "/api/v1/simulate", body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            self.assertIn(resp.status, (429, 503))
            self.assertEqual(resp.getheader("Retry-After"), "5")
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("code"), "simulation_queue_full")
            conn.close()
        finally:
            SIM_SLOTS.release()
            SIM_SLOTS.release()

    def test_simulation_wall_clock_timeout_and_caps(self) -> None:
        from allocation.cluster import Cluster
        from allocation.config import build_heterogeneous_specs
        from allocation.heuristics import FirstFit
        from allocation.simulator import TraceSimulator
        from allocation.exceptions import SimulationTimeoutError, SimulationLimitError
        from datasets.trace_loader import generate_alibaba_style_trace

        cluster = Cluster(build_heterogeneous_specs(2))
        trace = generate_alibaba_style_trace(n=50, seed=42)
        sim = TraceSimulator(cluster=cluster, allocator=FirstFit())

        # Test wall-clock timeout
        with self.assertRaises(SimulationTimeoutError):
            sim.run(trace, reset_cluster=True, wall_clock_timeout_sec=0.000001)

        # Test step limit cap
        with self.assertRaises(SimulationLimitError):
            sim.run(trace, reset_cluster=True, max_steps=5)

        # Test simulated duration cap
        with self.assertRaises(SimulationLimitError):
            sim.run(trace, reset_cluster=True, max_duration_sec=0.1)

    def test_input_validation_bounds_num_workloads(self) -> None:
        # Exceeds maximum 1000 workloads cap
        st, data = self._request("POST", "/api/v1/simulate", {"num_workloads": 1001})
        self.assertEqual(st, 400)
        self.assertEqual(data.get("code"), "bad_request")
        self.assertIn("out of bounds", data.get("error", ""))

        # Zero or negative workloads
        st, data = self._request("POST", "/api/v1/simulate", {"num_workloads": 0})
        self.assertEqual(st, 400)
        self.assertEqual(data.get("code"), "bad_request")

        st, data = self._request("POST", "/api/v1/simulate", {"num_workloads": -10})
        self.assertEqual(st, 400)

    def test_input_validation_invalid_policy(self) -> None:
        st, data = self._request("POST", "/api/v1/simulate", {"policy": "magic_ai_unsupported"})
        self.assertEqual(st, 400)
        self.assertEqual(data.get("code"), "bad_request")
        self.assertIn("Invalid policy", data.get("error", ""))

    def test_input_validation_allocate_bounds(self) -> None:
        # Negative CPU
        st, data = self._request("POST", "/api/v1/allocate", {"workload_id": "w1", "cpu": -1.0, "ram": 2.0})
        self.assertEqual(st, 400)
        self.assertIn("cpu", data.get("error", "").lower())

        # Out-of-bounds RAM
        st, data = self._request("POST", "/api/v1/allocate", {"workload_id": "w1", "cpu": 1.0, "ram": 5000.0})
        self.assertEqual(st, 400)
        self.assertIn("ram", data.get("error", "").lower())

    @patch("backend.store.save_sync")
    def test_simulate_runs_and_returns_not_saved_when_db_down(self, mock_save_sync: MagicMock) -> None:
        mock_save_sync.return_value = False
        st, data = self._request("POST", "/api/v1/simulate", {"policy": "first_fit", "num_workloads": 10})
        self.assertEqual(st, 200)
        self.assertIn("placed_count", data)
        # Resilience verification: simulation runs successfully even if DB fails
        self.assertEqual(data.get("store_status"), "not saved")
        self.assertFalse(data.get("persisted"))

    @patch("backend.store.save_sync")
    def test_simulate_returns_saved_when_db_succeeds(self, mock_save_sync: MagicMock) -> None:
        mock_save_sync.return_value = True
        st, data = self._request("POST", "/api/v1/simulate", {"policy": "first_fit", "num_workloads": 10})
        self.assertEqual(st, 200)
        self.assertEqual(data.get("store_status"), "saved")
        self.assertTrue(data.get("persisted"))

    def test_structured_error_responses_no_stack_traces(self) -> None:
        st, data = self._request("GET", "/non_existent_endpoint")
        self.assertEqual(st, 404)
        self.assertIn("error", data)
        self.assertEqual(data.get("status"), 404)
        self.assertEqual(data.get("code"), "not_found")
        self.assertNotIn("Traceback", json.dumps(data))

    @unittest.skipUnless(os.environ.get("RUN_LIVE_TESTS") == "1", "Live Render tests gated behind RUN_LIVE_TESTS=1")
    def test_live_render_end_to_end_simulation_and_history(self) -> None:
        """Integration test against live Render backend: verifies simulation saves, appears in history, and cleans up."""
        render_url = os.environ.get("RENDER_BACKEND_URL", "https://cloudpilot-2whk.onrender.com")
        api_key = os.environ.get("CLOUDPILOT_API_KEY", "")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        try:
            # 1. Health check to ensure remote backend is awake
            req_h = urllib.request.Request(f"{render_url}/health")
            with urllib.request.urlopen(req_h, timeout=10.0) as resp:
                if resp.status != 200:
                    self.skipTest("Live Render backend not returning 200 on /health")
        except Exception as exc:
            self.skipTest(f"Live Render backend unreachable ({exc}); skipping live integration test.")

        # 2. Trigger on-demand simulation with tagged seed
        tagged_seed = 999000 + (int(time.time()) % 900)
        sim_payload = json.dumps({
            "policy": "round_robin",
            "num_workloads": 12,
            "seed": tagged_seed,
        }).encode("utf-8")

        req_sim = urllib.request.Request(
            f"{render_url}/api/v1/simulate",
            data=sim_payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req_sim, timeout=15.0) as resp:
            self.assertEqual(resp.status, 200)
            sim_res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(sim_res.get("placed_count"), 12)

        # 3. Allow time for Supabase record propagation
        time.sleep(2.0)

        # 4. Verify tagged run appears in History
        req_hist = urllib.request.Request(f"{render_url}/api/v1/history", headers=headers)
        with urllib.request.urlopen(req_hist, timeout=10.0) as resp:
            self.assertEqual(resp.status, 200)
            hist = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(hist, list)
            self.assertGreater(len(hist), 0)
            found_run = any(
                entry.get("seed") == tagged_seed or (entry.get("result", {}).get("seed") == tagged_seed)
                for entry in hist
            )
            self.assertTrue(found_run, f"Tagged run (seed {tagged_seed}) not found in live history.")

        # 5. Clean up / delete row afterwards
        if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
            store.delete_sync("simulation_runs", {"seed": f"eq.{tagged_seed}"})



if __name__ == "__main__":
    unittest.main()


