"""Workload model representing cloud compute demands."""

from __future__ import annotations

import math
from typing import Any, Union

from allocation.exceptions import InvalidResourceError


def _validate_positive_number(value: Any, name: str) -> Union[int, float]:
    """Validate that a value is a finite positive number (> 0).

    Args:
        value: Numeric value to validate.
        name: Name of the parameter for clear error messaging.

    Returns:
        The validated numeric value.

    Raises:
        InvalidResourceError: If value is not a number, is a bool, is NaN/inf, or is <= 0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidResourceError(
            f"{name} must be a valid number, got {type(value).__name__} ({value!r})"
        )
    if math.isnan(value) or math.isinf(value):
        raise InvalidResourceError(f"{name} must be a finite number, got {value}")
    if value <= 0:
        raise InvalidResourceError(f"{name} must be strictly positive (> 0), got {value}")
    return value


def _validate_identifier(identifier: Any, name: str) -> str:
    """Validate that an identifier is a non-empty string.

    Args:
        identifier: Identifier to validate.
        name: Name of the parameter for error messaging.

    Returns:
        The stripped string identifier.

    Raises:
        InvalidResourceError: If identifier is not a string or is empty.
    """
    if not isinstance(identifier, str):
        raise InvalidResourceError(
            f"{name} must be a string, got {type(identifier).__name__}"
        )
    cleaned = identifier.strip()
    if not cleaned:
        raise InvalidResourceError(f"{name} cannot be empty or whitespace only")
    return cleaned


class Workload:
    """Represents a discrete compute job or service requiring CPU and RAM resources.

    Attributes:
        workload_id: Unique identifier for the workload.
        cpu_requirement: Required CPU capacity (in vCPUs or cores).
        ram_requirement: Required RAM capacity (in GB).
    """

    def __init__(
        self,
        workload_id: str,
        cpu_requirement: Union[int, float],
        ram_requirement: Union[int, float],
    ) -> None:
        self.workload_id = _validate_identifier(workload_id, "Workload ID")
        self.cpu_requirement = _validate_positive_number(cpu_requirement, "CPU requirement")
        self.ram_requirement = _validate_positive_number(ram_requirement, "RAM requirement")

    @property
    def cpu(self) -> Union[int, float]:
        """Convenience property for cpu_requirement."""
        return self.cpu_requirement

    @property
    def ram(self) -> Union[int, float]:
        """Convenience property for ram_requirement."""
        return self.ram_requirement

    def __repr__(self) -> str:
        return (
            f"Workload(workload_id='{self.workload_id}', "
            f"cpu_requirement={self.cpu_requirement}, "
            f"ram_requirement={self.ram_requirement})"
        )

    def __str__(self) -> str:
        return (
            f"Workload[{self.workload_id}] "
            f"(CPU: {self.cpu_requirement}, RAM: {self.ram_requirement}GB)"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Workload):
            return False
        return (
            self.workload_id == other.workload_id
            and math.isclose(self.cpu_requirement, other.cpu_requirement, rel_tol=1e-9)
            and math.isclose(self.ram_requirement, other.ram_requirement, rel_tol=1e-9)
        )

    def __hash__(self) -> int:
        return hash((self.workload_id, self.cpu_requirement, self.ram_requirement))

