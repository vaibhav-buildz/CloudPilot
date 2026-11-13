"""Backend control plane package for CloudPilot."""

from backend.server import (
    CloudPilotHTTPServer,
    CloudPilotRequestHandler,
    create_app_server,
)
from backend.service import CloudPilotService


def run_smoke_suite(*args, **kwargs):
    """Lazy import wrapper to avoid runpy circular warning on python -m backend.smoke_test."""
    from backend.smoke_test import run_smoke_suite as _run
    return _run(*args, **kwargs)


__all__ = [
    "CloudPilotService",
    "CloudPilotHTTPServer",
    "CloudPilotRequestHandler",
    "create_app_server",
    "run_smoke_suite",
]
