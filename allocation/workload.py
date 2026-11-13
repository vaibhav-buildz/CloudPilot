"""Workload model representing cloud compute demands."""

from __future__ import annotations

import math
from typing import Any, Optional, Union

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


def _validate_non_negative_number(value: Any, name: str) -> float:
    """Validate that a value is a finite non-negative number (>= 0.0).

    Args:
        value: Numeric value to validate.
        name: Name of the parameter for clear error messaging.

    Returns:
        The validated float value.

    Raises:
        InvalidResourceError: If value is not a number, is a bool, is NaN/inf, or is < 0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidResourceError(
            f"{name} must be a valid number, got {type(value).__name__} ({value!r})"
        )
    if math.isnan(value) or math.isinf(value):
        raise InvalidResourceError(f"{name} must be a finite number, got {value}")
    if value < 0:
        raise InvalidResourceError(f"{name} must be non-negative (>= 0.0), got {value}")
    return float(value)


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
        arrival_time: Timestamp (in seconds) when the workload arrives.
        duration: Optional runtime duration (in seconds) of the workload.
    """

    def __init__(
        self,
        workload_id: str,
        cpu_requirement: Union[int, float],
        ram_requirement: Union[int, float],
        arrival_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> None:
        self.workload_id = _validate_identifier(workload_id, "Workload ID")
        self.cpu_requirement = _validate_positive_number(cpu_requirement, "CPU requirement")
        self.ram_requirement = _validate_positive_number(ram_requirement, "RAM requirement")
        self.arrival_time: float = _validate_non_negative_number(arrival_time, "Arrival time")
        self.duration: Optional[float] = (
            float(_validate_positive_number(duration, "Duration"))
            if duration is not None
            else None
        )

    @property
    def cpu(self) -> Union[int, float]:
        """Convenience property for cpu_requirement."""
        return self.cpu_requirement

    @property
    def ram(self) -> Union[int, float]:
        """Convenience property for ram_requirement."""
        return self.ram_requirement

    @property
    def completion_time(self) -> Optional[float]:
        """Timestamp when the workload completes, or None if duration is indefinite."""
        return (self.arrival_time + self.duration) if self.duration is not None else None

    def __repr__(self) -> str:
        base = (
            f"Workload(workload_id='{self.workload_id}', "
            f"cpu_requirement={self.cpu_requirement}, "
            f"ram_requirement={self.ram_requirement}"
        )
        if self.arrival_time > 0.0 or self.duration is not None:
            base += f", arrival_time={self.arrival_time}, duration={self.duration}"
        return base + ")"

    def __str__(self) -> str:
        details = f"Workload[{self.workload_id}] (CPU: {self.cpu_requirement}, RAM: {self.ram_requirement}GB"
        if self.arrival_time > 0.0:
            details += f", arrived={self.arrival_time}s"
        if self.duration is not None:
            details += f", duration={self.duration}s"
        return details + ")"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Workload):
            return False
        base_eq = (
            self.workload_id == other.workload_id
            and math.isclose(self.cpu_requirement, other.cpu_requirement, rel_tol=1e-9)
            and math.isclose(self.ram_requirement, other.ram_requirement, rel_tol=1e-9)
            and math.isclose(self.arrival_time, other.arrival_time, rel_tol=1e-9, abs_tol=1e-9)
        )
        if not base_eq:
            return False
        if (self.duration is None) != (other.duration is None):
            return False
        if self.duration is not None and other.duration is not None:
            return math.isclose(self.duration, other.duration, rel_tol=1e-9, abs_tol=1e-9)
        return True

    def __hash__(self) -> int:
        return hash((
            self.workload_id,
            self.cpu_requirement,
            self.ram_requirement,
            self.arrival_time,
            self.duration,
        ))
