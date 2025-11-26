"""
OPM-specific well failure parser.
"""

import re
from typing import List
from ...core.well_data import WellFailure


class OPMWellFailureParser:
    """Parses OPM Flow well failure strings into generic WellFailure objects."""

    @staticmethod
    def parse_well_status_string(status_string: str) -> List[WellFailure]:
        """
        Parse OPM well status string into list of WellFailure objects.

        Example input: "FAIL { WELLA ControlRate } { WELLB MassBalance Phase=2 }"

        Args:
            status_string: OPM well status string

        Returns:
            List of WellFailure objects
        """
        failures = []

        if not isinstance(status_string, str) or "FAIL" not in status_string:
            return failures

        # Extract failure blocks like "{ WELLA ControlRate }"
        failure_blocks = re.findall(r"\{([^}]+)\}", status_string)

        for block in failure_blocks:
            parts = block.strip().split()
            if not parts:
                continue

            well_name = parts[0]
            failure_info = " ".join(parts[1:]) if len(parts) > 1 else ""

            # Parse failure type and phase
            failure_type, phase = OPMWellFailureParser._parse_failure_info(failure_info)

            failures.append(
                WellFailure(
                    well_name=well_name,
                    failure_type=failure_type,
                    phase=phase,
                    details=failure_info,  # Keep original for reference
                )
            )

        return failures

    @staticmethod
    def _parse_failure_info(failure_info: str) -> tuple:
        """Parse OPM failure info string to determine type and phase."""
        upper = failure_info.upper()
        phase = None

        # Extract phase if present
        if "PHASE=" in upper:
            phase_match = re.search(r"PHASE=(\d+)", upper)
            if phase_match:
                phase_num = int(phase_match.group(1))
                # OPM phase numbering: 0=Water, 1=Oil, 2=Gas
                phase_map = {0: "Water", 1: "Oil", 2: "Gas"}
                phase = phase_map.get(phase_num)

        # Map OPM failure type strings directly
        # These match the OPM ConvergenceReport::WellFailure::Type enum
        if "MASSBALANCE" in upper:
            return "MassBalance", phase
        elif "CONTROLRATE" in upper:
            return "ControlRate", None
        elif "CONTROLBHP" in upper:
            return "ControlBHP", None
        elif "CONTROLTHP" in upper:
            return "ControlTHP", None
        elif "PRESSURE" in upper:
            return "Pressure", None
        elif "UNSOLVABLE" in upper:
            return "Unsolvable", None
        elif "WRONGFLOWDIRECTION" in upper:
            return "WrongFlowDirection", None
        elif "INVALID" in upper:
            return "Invalid", None
        else:
            # Keep unknown types as-is
            return failure_info if failure_info else "Unknown", None
