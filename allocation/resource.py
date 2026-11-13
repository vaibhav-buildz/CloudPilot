"""Resource model representing cloud physical or virtual servers."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Union

from allocation.exceptions import (
    InsufficientCPUError,
    InsufficientRAMError,
    InvalidResourceError,
    ReleaseError,
)
from allocation.workload import Workload, _validate_identifier, _validate_positive_number


class CloudServer:
    """Represents a cloud compute node with finite CPU and RAM capacity.

    Attributes:
        server_id: Unique string identifier for the server.
        total_cpu: Total CPU capacity (in vCPUs or cores).
        total_ram: Total RAM capacity (in GB).
        used_cpu: Currently allocated CPU.
        used_ram: Currently allocated RAM.
    """

    def __init__(
        self,
        server_id: str,
        total_cpu: Union[int, float],
        total_ram: Union[int, float],
    ) -> None:
        self.server_id = _validate_identifier(server_id, "Server ID")
        self.total_cpu = _validate_positive_number(total_cpu, "Total CPU")
        self.total_ram = _validate_positive_number(total_ram, "Total RAM")

        self.used_cpu: Union[int, float] = 0
        self.used_ram: Union[int, float] = 0
        self._allocated_workloads: List[Workload] = []

    def available_cpu(self) -> Union[int, float]:
        """Calculate the remaining unallocated CPU capacity."""
        remaining = self.total_cpu - self.used_cpu
        return 0 if abs(remaining) < 1e-9 else remaining

    def available_ram(self) -> Union[int, float]:
        """Calculate the remaining unallocated RAM capacity."""
        remaining = self.total_ram - self.used_ram
        return 0 if abs(remaining) < 1e-9 else remaining

    @property
    def cpu_utilization(self) -> float:
        """Calculate the CPU utilization as a percentage (0.0 to 100.0)."""
        return (self.used_cpu / self.total_cpu) * 100.0

    @property
    def ram_utilization(self) -> float:
        """Calculate the RAM utilization as a percentage (0.0 to 100.0)."""
        return (self.used_ram / self.total_ram) * 100.0

    @property
    def allocated_workloads(self) -> List[Workload]:
        """Return a copy of the list of workloads allocated to this server."""
        return list(self._allocated_workloads)

    def _resolve_resource_args(
        self,
        cpu_or_workload: Union[int, float, Workload],
        ram: Optional[Union[int, float]],
        operation: str,
    ) -> tuple[Union[int, float], Union[int, float], Optional[Workload]]:
        """Normalize arguments whether provided as a Workload or numeric pair."""
        if isinstance(cpu_or_workload, Workload):
            return (
                cpu_or_workload.cpu_requirement,
                cpu_or_workload.ram_requirement,
                cpu_or_workload,
            )

        if ram is None:
            raise InvalidResourceError(
                f"RAM must be provided when {operation} by numeric CPU capacity"
            )

        validated_cpu = _validate_positive_number(cpu_or_workload, "CPU")
        validated_ram = _validate_positive_number(ram, "RAM")
        return validated_cpu, validated_ram, None

    def can_allocate(
        self,
        cpu_or_workload: Union[int, float, Workload],
        ram: Optional[Union[int, float]] = None,
    ) -> bool:
        """Check whether the requested resources can be allocated without modifying state."""
        try:
            req_cpu, req_ram, _ = self._resolve_resource_args(
                cpu_or_workload, ram, "checking allocation"
            )
        except (InvalidResourceError, TypeError, ValueError):
            return False

        return (req_cpu <= self.available_cpu() + 1e-9) and (
            req_ram <= self.available_ram() + 1e-9
        )

    def allocate(
        self,
        cpu_or_workload: Union[int, float, Workload],
        ram: Optional[Union[int, float]] = None,
    ) -> bool:
        """Allocate compute resources to the server atomically.

        Args:
            cpu_or_workload: A Workload instance or numeric CPU demand.
            ram: Required RAM demand when numeric CPU is supplied.

        Returns:
            True if allocation succeeded.

        Raises:
            InvalidResourceError: If input values are negative, zero, or non-numeric.
            InsufficientCPUError: If available CPU cannot satisfy demand.
            InsufficientRAMError: If available RAM cannot satisfy demand.
        """
        req_cpu, req_ram, workload_obj = self._resolve_resource_args(
            cpu_or_workload, ram, "allocating"
        )

        avail_cpu = self.available_cpu()
        avail_ram = self.available_ram()

        # Check capacity constraints before altering any state (atomic guarantee)
        if req_cpu > avail_cpu + 1e-9:
            raise InsufficientCPUError(
                f"Insufficient CPU on server '{self.server_id}': "
                f"requested {req_cpu}, available {avail_cpu} (total {self.total_cpu})"
            )

        if req_ram > avail_ram + 1e-9:
            raise InsufficientRAMError(
                f"Insufficient RAM on server '{self.server_id}': "
                f"requested {req_ram}, available {avail_ram} (total {self.total_ram})"
            )

        # Apply state changes atomically
        self.used_cpu += req_cpu
        self.used_ram += req_ram

        # Normalize clean boundaries against floating point artifacts
        if math.isclose(self.used_cpu, self.total_cpu, abs_tol=1e-9):
            self.used_cpu = self.total_cpu
        if math.isclose(self.used_ram, self.total_ram, abs_tol=1e-9):
            self.used_ram = self.total_ram

        if workload_obj is not None:
            self._allocated_workloads.append(workload_obj)

        return True

    def allocate_workload(self, workload: Workload) -> bool:
        """Convenience method to allocate a Workload instance."""
        return self.allocate(workload)

    def release(
        self,
        cpu_or_workload: Union[int, float, Workload],
        ram: Optional[Union[int, float]] = None,
    ) -> bool:
        """Release previously allocated resources atomically.

        Args:
            cpu_or_workload: A Workload instance or numeric CPU capacity to release.
            ram: Numeric RAM capacity to release when numeric CPU is supplied.

        Returns:
            True if release succeeded.

        Raises:
            InvalidResourceError: If input values are negative, zero, or non-numeric.
            ReleaseError: If requested release exceeds currently used CPU or RAM.
        """
        rel_cpu, rel_ram, workload_obj = self._resolve_resource_args(
            cpu_or_workload, ram, "releasing"
        )

        # Validate that we are not releasing more than currently allocated (atomic check)
        if rel_cpu > self.used_cpu + 1e-9:
            raise ReleaseError(
                f"Cannot release {rel_cpu} CPU from server '{self.server_id}': "
                f"only {self.used_cpu} currently allocated"
            )

        if rel_ram > self.used_ram + 1e-9:
            raise ReleaseError(
                f"Cannot release {rel_ram} RAM from server '{self.server_id}': "
                f"only {self.used_ram} currently allocated"
            )

        # Apply state updates atomically
        self.used_cpu -= rel_cpu
        self.used_ram -= rel_ram

        # Clean zero threshold against floating point residuals
        if abs(self.used_cpu) < 1e-9:
            self.used_cpu = 0
        if abs(self.used_ram) < 1e-9:
            self.used_ram = 0

        if workload_obj is not None and workload_obj in self._allocated_workloads:
            self._allocated_workloads.remove(workload_obj)

        return True

    def release_workload(self, workload: Workload) -> bool:
        """Convenience method to release a Workload instance."""
        return self.release(workload)

    def __repr__(self) -> str:
        return (
            f"CloudServer(server_id='{self.server_id}', "
            f"total_cpu={self.total_cpu}, total_ram={self.total_ram}, "
            f"used_cpu={self.used_cpu}, used_ram={self.used_ram})"
        )

    def __str__(self) -> str:
        return (
            f"Server[{self.server_id}] | "
            f"CPU: {self.used_cpu}/{self.total_cpu} ({self.cpu_utilization:.1f}%) | "
            f"RAM: {self.used_ram}/{self.total_ram} GB ({self.ram_utilization:.1f}%)"
        )
