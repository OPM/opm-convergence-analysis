"""
Generic well failure data structures for multi-simulator support.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WellFailure:
    """
    Generic well failure information for a single well at a single iteration.

    Attributes:
        well_name: Name of the well
        failure_type: Type of failure (simulator-specific string, e.g., "MassBalance", "Pressure")
        phase: Phase affected (e.g., "Oil", "Water", "Gas"), if applicable
        details: Additional simulator-specific details
    """

    well_name: str
    failure_type: str
    phase: Optional[str] = None
    details: Optional[str] = None

    def get_display_reason(self) -> str:
        """Get human-readable failure reason."""
        if self.phase:
            return f"{self.failure_type} ({self.phase})"
        return self.failure_type
