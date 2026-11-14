"""SLA efficiency scoring for CloudPilot simulation evaluations."""

from typing import Any


def compute_sla_efficiency_score(sim: Any) -> float:
    """Computes a composite [0, 100] research score balancing SLA acceptance (60%),
    RAM utilization (25%), and load balance (15%).
    """
    acceptance_rate = 100.0 - sim.rejection_rate_pct
    balance_score = max(0.0, 100.0 - sim.avg_load_imbalance * 2.5)
    score = (
        0.60 * acceptance_rate
        + 0.25 * sim.avg_ram_utilization_pct
        + 0.15 * balance_score
    )
    return round(score, 2)

