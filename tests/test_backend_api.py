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
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
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
            st, _ = self._request("GET", "/healthz")
            self.assertEqual(st, 200)
            st, _ = self._request("GET", "/api/v1/health")
            self.assertEqual(st, 200)
            st, _ = self._request("GET", "/api/v1/cluster")
            self.assertEqual(st, 200)
            st, _ = self._request("GET", "/api/v1/history")
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

            # Public POST /simulate does NOT require key
            st, res = self._request(
                "POST", "/api/v1/simulate", {"policy": "first_fit", "num_workloads": 10}
            )
            self.assertEqual(st, 200)
            self.assertIn("placed_count", res)

            # Protected POST endpoints DO require key
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


if __name__ == "__main__":
    unittest.main()

