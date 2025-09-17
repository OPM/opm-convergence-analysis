"""
Helper functions and utilities for convergence monitoring.

This module provides utility functions that replicate MATLAB functionality
used throughout the convergence monitoring codebase.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Union, Optional


def compute_start_positions(iterations: np.ndarray) -> np.ndarray:
    """
    Compute start positions for iteration curves.

    Identifies where new timesteps begin by looking for decreases in iteration numbers.

    Args:
        iterations: Array of iteration numbers

    Returns:
        Array of start positions (curve_pos)
    """
    if len(iterations) == 0:
        return np.array([0])

    # Find where iteration decreases (new timestep starts)
    decreases = np.concatenate([[True], iterations[1:] < iterations[:-1], [True]])
    return np.where(decreases)[0]


def process_well_status(well_status: List[str]) -> np.ndarray:
    """
    Process well status to identify failed wells.

    Args:
        well_status: List of well status strings

    Returns:
        Boolean array indicating failed wells
    """
    return np.array(["FAIL" in status for status in well_status])


def collect_columns(
    values: List[np.ndarray], header: List[str], pattern: str
) -> Dict[str, Any]:
    """
    Collect columns matching a pattern into a structure.

    Args:
        values: List of column value arrays
        header: List of column names
        pattern: Regex pattern to match column names

    Returns:
        Dictionary with 'value' and 'label' keys
    """
    import re

    # Find matching columns
    matching_indices = []
    matching_labels = []

    for i, col_name in enumerate(header):
        if re.match(pattern, col_name):
            matching_indices.append(i)
            # Replace underscores with dots for labels
            label = col_name.replace("_", ".")
            matching_labels.append(label)

    if not matching_indices:
        # Return empty structure if no matches
        return {
            "value": np.array([]).reshape(len(values[0]) if values else 0, 0),
            "label": [],
        }

    # Collect matching values
    collected_values = np.column_stack([values[i] for i in matching_indices])

    return {"value": collected_values, "label": matching_labels}
