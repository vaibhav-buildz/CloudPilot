"""Custom exceptions for CloudPilot resource allocation.

All domain exceptions inherit from AllocationError and standard Python exceptions
(such as ValueError) to ensure interoperability and intuitive error handling.
"""


class AllocationError(Exception):
    """Base exception for all resource allocation errors in CloudPilot."""
    pass


class InvalidResourceError(ValueError, AllocationError):
    """Raised when resource parameters are invalid (non-positive, non-numeric, NaN, inf, or empty identifiers)."""
    pass


class InsufficientCapacityError(AllocationError, ValueError):
    """Base exception when a server cannot satisfy an allocation request due to capacity constraints."""
    pass


class InsufficientCPUError(InsufficientCapacityError):
    """Raised when available server CPU is insufficient for the requested allocation."""
    pass


class InsufficientRAMError(InsufficientCapacityError):
    """Raised when available server RAM is insufficient for the requested allocation."""
    pass


class ReleaseError(AllocationError, ValueError):
    """Raised when attempting to release more resources than currently allocated or invalid release parameters."""
    pass

