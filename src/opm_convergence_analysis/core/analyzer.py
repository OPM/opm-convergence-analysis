"""
Analyzer class for calculating error metrics from OPM INFOITER data for convergence analysis.

This module provides the Analyzer class which analyzes convergence behavior
from OPM simulation data.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, List, Tuple


class Analyzer:
    """
    Analyzer for OPM convergence behavior.

    This class provides functionality to analyze convergence behavior from
    INFOITER data, calculating error metrics and convergence indicators.
    """

    def __init__(self):
        """Initialize the Analyzer."""
        pass

    def analyze(
        self, data: Dict[str, Any], **kwargs
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Calculate error metrics from OPM's INFOITER data for convergence analysis.

        Args:
            data: Data structure from DataReader containing mb, cnv, curve_pos, raw
            **kwargs: Optional parameters:
                - tol: Convergence tolerances (dict with 'mb' and 'cnv' fields)

        Returns:
            Tuple of (errors, labels, metrics) where:
            - errors: m-by-k array of error measures
            - labels: List of convergence metric labels
            - metrics: Dictionary with convergence indicators
        """
        opt = {}
        opt["tol"] = {}
        opt["inferFailure"] = lambda e: np.sum(e > 0, axis=1)
        opt["calcDistance"] = lambda e: np.sum(e, axis=1)

        # Merge passed tolerances
        if "tol" in kwargs:
            opt["tol"].update(kwargs["tol"])

        # Extract data components
        mb = data["mb"]
        cnv = data["cnv"]
        curve_pos = data["curve_pos"]
        raw = data["raw"]

        # Compute errors and labels
        errors, labels = self._compute_errors(mb, cnv, opt)

        # Calculate basic failure and distance metrics
        fail = opt["inferFailure"](errors)
        dist = opt["calcDistance"](errors)

        # Ensure these are 1D arrays
        fail = np.atleast_1d(fail)
        dist = np.atleast_1d(dist)

        # Identify successful steps
        conv = self._identify_successful_steps(curve_pos, raw)

        # Create metrics structure with core analysis metrics
        metrics = {
            "fail": fail,
            "dist": dist,
            "conv": conv,
            "raw": raw,
        }

        return errors, labels, metrics

    def _compute_errors(
        self, mb: Dict[str, Any], cnv: Dict[str, Any], opt: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute error measures from material balance and convergence data.

        Args:
            mb: Material balance data structure
            cnv: Convergence data structure
            opt: Options dictionary with tolerance information

        Returns:
            Tuple of (errors, labels)
        """
        # Merge tolerances with defaults
        tol = opt["tol"]

        n_cnv = len(cnv["label"]) if cnv and "label" in cnv else 0
        n_mb = len(mb["label"]) if mb and "label" in mb else 0

        # Create tolerance array matching data structure
        # CNV columns come first, then MB columns in the combined array
        if "cnv" not in tol:
            raise ValueError("Missing required tolerance 'cnv' in tolerance parameters")
        if "mb" not in tol:
            raise ValueError("Missing required tolerance 'mb' in tolerance parameters")

        tol_array = np.concatenate(
            [np.full(n_cnv, tol["cnv"]), np.full(n_mb, tol["mb"])]
        )

        # Combine CNV and MB values
        combined_values = np.column_stack([cnv["value"], mb["value"]])

        # Calculate log10 errors, clipped to be non-negative
        # error = max(log10(value/tolerance), 0)
        errors = np.maximum(np.log10(combined_values / tol_array[np.newaxis, :]), 0.0)

        # Combine labels
        labels = list(cnv["label"]) + list(mb["label"])

        return errors, labels

    def _identify_successful_steps(
        self, curve_pos: np.ndarray, raw: Dict[str, Any]
    ) -> np.ndarray:
        """
        Identify which timesteps were successful (converged to next timestep).

        Args:
            curve_pos: Curve position array
            raw: Raw data dictionary

        Returns:
            Boolean array indicating successful steps
        """
        n_steps = len(curve_pos) - 1 if curve_pos is not None else 0

        if "ReportStep" not in raw or "TimeStep" not in raw:
            # If we don't have step information, assume all steps succeeded
            return np.ones(n_steps, dtype=bool)

        # Get step information
        report_step = raw["ReportStep"]
        time_step = raw["TimeStep"]

        # Combine into step array
        step = np.column_stack([report_step, time_step])

        # Get indices of last and first iterations of consecutive timesteps
        last_iter_indices = curve_pos[1:-1] - 1  # Last iteration of previous step
        first_iter_indices = curve_pos[1:-1]  # First iteration of current step

        # Step succeeded if we advanced to next report step or time step
        success = []
        for i in range(len(last_iter_indices)):
            last_step = step[last_iter_indices[i]]
            first_step = step[first_iter_indices[i]]

            # Step is successful if any component (report step or time step) increased
            step_successful = np.any(first_step > last_step)
            success.append(step_successful)

        # Add final step as successful (always true for last step)
        success.append(True)

        return np.array(success, dtype=bool)
