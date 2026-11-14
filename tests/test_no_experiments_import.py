"""Test ensuring backend and production modules have zero dependency on experiments/ or tests/."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class TestNoExperimentsImport(unittest.TestCase):
    """Verifies production code runs in an isolated environment without experiments/."""

    def test_backend_imports_and_runs_without_experiments(self) -> None:
        project_root = Path(__file__).resolve().parent.parent

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            # Copy only production directories: allocation, datasets, ml, backend
            for pkg in ("allocation", "datasets", "ml", "backend"):
                src = project_root / pkg
                dst = temp_dir / pkg
                shutil.copytree(src, dst)

            # Assert experiments/ and tests/ are strictly absent in temp_dir
            self.assertFalse((temp_dir / "experiments").exists())
            self.assertFalse((temp_dir / "tests").exists())

            # Inline test script executed in a subprocess
            test_script = (
                "import sys\n"
                "temp_root = sys.argv[1]\n"
                "# Clean sys.path to strictly isolate environment\n"
                "sys.path = [temp_root] + [p for p in sys.path if 'CloudPilot' not in p and p != '']\n"
                "import backend.server\n"
                "import backend.service\n"
                "svc = backend.service.CloudPilotService()\n"
                "res = svc.run_simulation(policy='first_fit', num_workloads=10, seed=42)\n"
                "assert 'sla_efficiency_score' in res, 'Missing sla_efficiency_score in simulation result'\n"
                "print('ISOLATED_BACKEND_SUCCESS')\n"
            )

            # Run in isolated mode with clean environment
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)

            proc = subprocess.run(
                [sys.executable, "-c", test_script, str(temp_dir)],
                cwd=str(temp_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )

            if proc.returncode != 0:
                self.fail(
                    f"Subprocess failed with code {proc.returncode} in isolated environment.\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}"
                )

            self.assertIn("ISOLATED_BACKEND_SUCCESS", proc.stdout)


if __name__ == "__main__":
    unittest.main()

