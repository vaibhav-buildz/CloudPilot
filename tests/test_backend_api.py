"""Unit and live HTTP integration tests for CloudPilot Control Plane API."""

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from allocation.exceptions import InvalidResourceError, ReleaseError
from backend.server import create_app_server
from backend.service import CloudPilotService
from backend.smoke_test import run_smoke_suite


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
    ) -> Tuple[int, Any]:
        url = f"http://{self.host}:{self.port}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
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


if __name__ == "__main__":
    unittest.main()

