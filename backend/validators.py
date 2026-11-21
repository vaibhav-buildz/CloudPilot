"""Input validation and bounds verification for CloudPilot API endpoints."""

from typing import Any, Dict, Optional, Set, Tuple

from allocation.exceptions import InvalidResourceError

ALLOWED_POLICIES: Set[str] = {
    "first_fit",
    "best_fit",
    "worst_fit",
    "round_robin",
    "predictive_ewma",
    "predictive_ar",
    "ucb_bandit",
}

# Bounds definitions for Render free-tier resilience and cluster sanity
MAX_SIMULATION_WORKLOADS = 1000
MIN_SIMULATION_WORKLOADS = 1
MAX_SEED = 2_147_483_647
MIN_SEED = 0

MIN_CPU = 0.01
MAX_CPU = 1024.0
MIN_RAM = 0.01
MAX_RAM = 4096.0
MIN_ARRIVAL = 0.0
MAX_ARRIVAL = 1e9
MIN_DURATION = 0.001
MAX_DURATION = 1e8


def validate_simulate_payload(data: Dict[str, Any]) -> Tuple[Optional[str], int, int]:
    """Validates simulate request payload and returns typed (policy, num_workloads, seed)."""
    if not isinstance(data, dict):
        raise InvalidResourceError("Request body must be a JSON object.")

    policy = data.get("policy")
    if policy is not None:
        if not isinstance(policy, str):
            raise InvalidResourceError("Field 'policy' must be a string.")
        policy_clean = policy.strip().lower()
        if policy_clean not in ALLOWED_POLICIES:
            raise InvalidResourceError(
                f"Invalid policy '{policy}'. Allowed policies: {sorted(list(ALLOWED_POLICIES))}."
            )
        policy = policy_clean

    num_workloads_raw = data.get("num_workloads", 200)
    if isinstance(num_workloads_raw, bool) or not isinstance(num_workloads_raw, int):
        raise InvalidResourceError(
            f"Field 'num_workloads' must be an integer between {MIN_SIMULATION_WORKLOADS} and {MAX_SIMULATION_WORKLOADS}."
        )
    if not (MIN_SIMULATION_WORKLOADS <= num_workloads_raw <= MAX_SIMULATION_WORKLOADS):
        raise InvalidResourceError(
            f"'num_workloads' out of bounds ({num_workloads_raw}). Permitted range: {MIN_SIMULATION_WORKLOADS} to {MAX_SIMULATION_WORKLOADS}."
        )

    seed_raw = data.get("seed", 42)
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise InvalidResourceError(f"Field 'seed' must be an integer between {MIN_SEED} and {MAX_SEED}.")
    if not (MIN_SEED <= seed_raw <= MAX_SEED):
        raise InvalidResourceError(f"'seed' out of bounds ({seed_raw}). Permitted range: {MIN_SEED} to {MAX_SEED}.")

    return policy, num_workloads_raw, seed_raw


def validate_allocate_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates allocate request payload with strict bounds on every numeric parameter."""
    if not isinstance(data, dict):
        raise InvalidResourceError("Request body must be a JSON object.")

    for field in ("workload_id", "cpu", "ram"):
        if field not in data:
            raise InvalidResourceError(f"Missing required field '{field}'.")

    workload_id = data["workload_id"]
    if not isinstance(workload_id, str) or not workload_id.strip() or len(workload_id) > 128:
        raise InvalidResourceError("Field 'workload_id' must be a non-empty string under 128 characters.")

    cpu = data["cpu"]
    if isinstance(cpu, bool) or not isinstance(cpu, (int, float)):
        raise InvalidResourceError(f"Field 'cpu' must be a numeric value between {MIN_CPU} and {MAX_CPU}.")
    cpu_float = float(cpu)
    if not (MIN_CPU <= cpu_float <= MAX_CPU):
        raise InvalidResourceError(f"'cpu' out of bounds ({cpu_float}). Permitted range: {MIN_CPU} to {MAX_CPU}.")

    ram = data["ram"]
    if isinstance(ram, bool) or not isinstance(ram, (int, float)):
        raise InvalidResourceError(f"Field 'ram' must be a numeric value between {MIN_RAM} and {MAX_RAM}.")
    ram_float = float(ram)
    if not (MIN_RAM <= ram_float <= MAX_RAM):
        raise InvalidResourceError(f"'ram' out of bounds ({ram_float}). Permitted range: {MIN_RAM} to {MAX_RAM}.")

    arrival_time = data.get("arrival_time", 0.0)
    if isinstance(arrival_time, bool) or not isinstance(arrival_time, (int, float)):
        raise InvalidResourceError(f"Field 'arrival_time' must be a numeric value >= {MIN_ARRIVAL}.")
    arr_float = float(arrival_time)
    if not (MIN_ARRIVAL <= arr_float <= MAX_ARRIVAL):
        raise InvalidResourceError(
            f"'arrival_time' out of bounds ({arr_float}). Permitted range: {MIN_ARRIVAL} to {MAX_ARRIVAL}."
        )

    duration = data.get("duration")
    dur_float = None
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise InvalidResourceError("Field 'duration' must be a positive numeric value.")
        dur_float = float(duration)
        if not (MIN_DURATION <= dur_float <= MAX_DURATION):
            raise InvalidResourceError(
                f"'duration' out of bounds ({dur_float}). Permitted range: {MIN_DURATION} to {MAX_DURATION}."
            )

    return {
        "workload_id": workload_id.strip(),
        "cpu": cpu_float,
        "ram": ram_float,
        "arrival_time": arr_float,
        "duration": dur_float,
    }


def validate_release_payload(data: Dict[str, Any]) -> str:
    """Validates release request payload."""
    if not isinstance(data, dict):
        raise InvalidResourceError("Request body must be a JSON object.")
    if "workload_id" not in data:
        raise InvalidResourceError("Missing required field 'workload_id'.")
    wid = data["workload_id"]
    if not isinstance(wid, str) or not wid.strip() or len(wid) > 128:
        raise InvalidResourceError("Field 'workload_id' must be a non-empty string under 128 characters.")
    return wid.strip()


def validate_policy_payload(data: Dict[str, Any]) -> str:
    """Validates policy update request payload."""
    if not isinstance(data, dict):
        raise InvalidResourceError("Request body must be a JSON object.")
    if "policy" not in data:
        raise InvalidResourceError("Missing required field 'policy'.")
    pol = data["policy"]
    if not isinstance(pol, str):
        raise InvalidResourceError("Field 'policy' must be a string.")
    clean = pol.strip().lower()
    if clean not in ALLOWED_POLICIES:
        raise InvalidResourceError(
            f"Invalid policy '{pol}'. Allowed policies: {sorted(list(ALLOWED_POLICIES))}."
        )
    return clean
