"""CloudPilot Simulator Entry Point.

Demonstrates:
1. Instantiating cloud servers with finite capacities.
2. Defining workloads with explicit CPU and RAM demands.
3. Allocating workloads to servers with atomic validation.
4. Displaying server resource utilization metrics.
5. Releasing resources and restoring capacity.
"""

from allocation.exceptions import InsufficientCapacityError
from allocation.resource import CloudServer
from allocation.workload import Workload


def display_server_status(title: str, servers: list[CloudServer]) -> None:
    """Pretty-print server status and resource utilization."""
    print(f"\n--- {title} ---")
    for server in servers:
        print(
            f"  {server.server_id:15} | "
            f"CPU: {server.used_cpu:4.1f}/{server.total_cpu:4.1f} vCPU ({server.cpu_utilization:5.1f}%) | "
            f"RAM: {server.used_ram:4.1f}/{server.total_ram:4.1f} GB ({server.ram_utilization:5.1f}%) | "
            f"Available: {server.available_cpu():.1f} vCPU, {server.available_ram():.1f} GB"
        )


def main() -> None:
    print("=" * 70)
    print(" CloudPilot - Cloud Resource Allocation Simulator")
    print("=" * 70)

    # 1. Create two cloud servers
    server_1 = CloudServer(server_id="Server-Alpha", total_cpu=8, total_ram=16)
    server_2 = CloudServer(server_id="Server-Beta", total_cpu=16, total_ram=32)
    servers = [server_1, server_2]

    display_server_status("1. Initial Server State (Idle)", servers)

    # 2. Create sample workloads
    workload_web = Workload(workload_id="WL-Web-Frontend", cpu_requirement=4, ram_requirement=8)
    workload_db = Workload(workload_id="WL-Database", cpu_requirement=8, ram_requirement=16)
    workload_analytics = Workload(workload_id="WL-Analytics-Worker", cpu_requirement=4, ram_requirement=8)

    print("\n--- 2. Sample Workloads Created ---")
    for wl in [workload_web, workload_db, workload_analytics]:
        print(f"  {wl}")

    # 3. Allocate resources
    print("\n--- 3. Allocating Workloads ---")
    print(f"  Allocating {workload_web.workload_id} -> {server_1.server_id}...")
    server_1.allocate(workload_web)

    print(f"  Allocating {workload_db.workload_id} -> {server_2.server_id}...")
    server_2.allocate(workload_db)

    print(f"  Allocating {workload_analytics.workload_id} -> {server_2.server_id}...")
    server_2.allocate(workload_analytics)

    # Demonstrate constraint validation / over-allocation prevention
    oversized_job = Workload(workload_id="WL-Oversized", cpu_requirement=8, ram_requirement=16)
    print(f"\n  Attempting to allocate oversized {oversized_job.workload_id} (8 vCPU) -> {server_1.server_id} (available: {server_1.available_cpu()} vCPU)...")
    try:
        server_1.allocate(oversized_job)
    except InsufficientCapacityError as err:
        print(f"  [Rejected Expectedly] {err}")

    # 4. Display server resource utilization
    display_server_status("4. Server Utilization Post-Allocation", servers)

    # 5. Release resources
    print("\n--- 5. Releasing Resources ---")
    print(f"  Releasing {workload_web.workload_id} from {server_1.server_id}...")
    server_1.release(workload_web)

    print(f"  Releasing {workload_analytics.workload_id} from {server_2.server_id}...")
    server_2.release(workload_analytics)

    display_server_status("6. Server Utilization Post-Release", servers)

    print("\n" + "=" * 70)
    print(" Simulation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
