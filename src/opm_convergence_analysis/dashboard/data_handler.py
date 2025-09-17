"""
Data handling functionality for the dashboard.

Contains data loading, processing, and state management for the dashboard.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np

from ..core.dbg_reader import load_case_data
import opm_convergence_analysis as oca


class DataHandler:
    """
    Handles data loading, processing, and state management for the dashboard.
    """

    def __init__(self):
        """Initialize the data handler."""
        self.current_data: Optional[Dict[str, Any]] = None
        self.current_errors: Optional[np.ndarray] = None
        self.current_labels: Optional[List[str]] = None
        self.current_metrics: Optional[Dict[str, Any]] = None
        self.current_savings: Optional[Dict[str, Any]] = None
        self.loaded_case_params: Dict[str, Any] = {}
        self.loaded_case_info: Dict[str, Any] = {}
        self.current_step_index: int = 0
        self.available_steps: List[int] = []

        # Performance optimization: cache plots that don't change between steps
        self._cached_plots: Dict[str, Any] = {}
        self._cache_invalidated: bool = True

    def load_case_from_path(self, input_path: str) -> bool:
        """
        Load case data from flexible input path.

        Args:
            input_path: Path to INFOITER, DBG, folder, or DATA file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load case data using the reader
            print(f"load_case_data() called for: {input_path}")
            case_data = load_case_data(input_path)

            if case_data["data"] is None:
                print(f"No INFOITER data found in {input_path}")
                return False

            # Store the loaded data
            self.current_data = case_data["data"]
            self.loaded_case_params = case_data["convergence_params"]
            self.loaded_case_info = case_data["case_info"]

            # Analyze convergence
            success = self._analyze_convergence()

            if success:
                self._update_available_steps()
                self._print_load_info()

            return success

        except Exception as e:
            print(f"Error loading case from {input_path}: {e}")
            return False

    def _analyze_convergence(self) -> bool:
        """
        Analyze convergence with current data and parameters.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get tolerances from DBG file (should always be available with defaults)
            tol_cnv = self.loaded_case_params.get("cnv_tolerance")
            tol_mb = self.loaded_case_params.get("mb_tolerance")

            # Analyze convergence
            mb_str = f"{tol_mb:.0e}" if tol_mb is not None else "None"
            cnv_str = f"{tol_cnv:.0e}" if tol_cnv is not None else "None"
            print(f"Analyzing convergence (MB: {mb_str}, CNV: {cnv_str})...")
            self.current_errors, self.current_labels, self.current_metrics = (
                oca.analyze_convergence(
                    self.current_data,
                    tol={"mb": tol_mb, "cnv": tol_cnv},
                )
            )
            # Invalidate plot cache since analysis data changed
            self._cache_invalidated = True

            return True

        except Exception as e:
            print(f"Error analyzing convergence: {e}")
            return False

    def _update_available_steps(self):
        """Update the list of available steps."""
        if self.current_metrics is not None:
            flagged_steps = list(self.current_metrics.get("flaggedSteps", []))
            if not flagged_steps and self.current_data is not None:
                n_steps = len(self.current_data["curve_pos"]) - 1
                flagged_steps = list(range(min(10, n_steps)))
            self.available_steps = flagged_steps

            # Ensure current_step_index is valid
            if self.current_step_index >= len(self.available_steps):
                self.current_step_index = 0

    def _print_load_info(self):
        """Print information about loaded case and parameters."""
        if self.loaded_case_params:
            print("Loaded parameters from DBG file:")
            for key, value in self.loaded_case_params.items():
                print(f"   {key}: {value}")

        if self.loaded_case_info.get("deck_filename"):
            print(f"Case: {Path(self.loaded_case_info['deck_filename']).name}")

    def navigate_step(self, direction: str) -> Tuple[int, bool, bool]:
        """
        Navigate to next/previous step.

        Args:
            direction: 'next' or 'prev'

        Returns:
            Tuple of (current_step, prev_disabled, next_disabled)
        """
        if not self.available_steps:
            return 0, True, True

        if direction == "prev" and self.current_step_index > 0:
            self.current_step_index -= 1
        elif (
            direction == "next"
            and self.current_step_index < len(self.available_steps) - 1
        ):
            self.current_step_index += 1

        current_step = (
            self.available_steps[self.current_step_index] if self.available_steps else 0
        )
        prev_disabled = self.current_step_index <= 0
        next_disabled = self.current_step_index >= len(self.available_steps) - 1

        return current_step, prev_disabled, next_disabled

    def get_current_step(self) -> int:
        """
        Get the current step number.

        Returns:
            Current step number
        """
        if self.available_steps and self.current_step_index < len(self.available_steps):
            return self.available_steps[self.current_step_index]
        return 0

    def get_step_display_text(self) -> str:
        """
        Get display text for current step including report step and timestep.

        Returns:
            Formatted step display text with report step and timestep info
        """
        if not self.available_steps:
            return "Step 0 of 0"

        current_step = self.available_steps[self.current_step_index]

        # Try to get report step and timestep information
        if (
            self.current_data
            and "raw" in self.current_data
            and "ReportStep" in self.current_data["raw"]
            and "TimeStep" in self.current_data["raw"]
            and "curve_pos" in self.current_data
        ):

            curve_pos = self.current_data["curve_pos"]
            if current_step < len(curve_pos) - 1:
                # Get the first iteration index for this step
                step_start_idx = curve_pos[current_step]
                report_step = self.current_data["raw"]["ReportStep"][step_start_idx]
                time_step = self.current_data["raw"]["TimeStep"][step_start_idx]

                return f"Step {current_step} of {len(self.available_steps)} total (Report: {report_step}, Time: {time_step})"

        # Fallback to original format if no step info available
        return f"Step {current_step} of {len(self.available_steps)} total"

    def get_progress_percentage(self) -> float:
        """
        Get progress percentage for current step.

        Returns:
            Progress percentage (0-100)
        """
        if not self.available_steps:
            return 0

        return (self.current_step_index / max(1, len(self.available_steps) - 1)) * 100

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        """Get current data."""
        return self.current_data

    @property
    def analysis_results(self) -> Optional[Tuple]:
        """Get current analysis results as tuple of (errors, labels, metrics)."""
        has_results = all(
            [
                self.current_errors is not None,
                self.current_labels is not None,
                self.current_metrics is not None,
            ]
        )
        if has_results:
            return self.current_errors, self.current_labels, self.current_metrics
        return None

    def has_data(self) -> bool:
        """
        Check if data is loaded.

        Returns:
            True if data is available, False otherwise
        """
        return all(
            [
                self.current_data is not None,
                self.current_errors is not None,
                self.current_metrics is not None,
            ]
        )

    def get_case_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the current case.

        Returns:
            Dictionary with case summary information
        """
        if not self.has_data():
            return {}

        n_steps = len(self.current_data["curve_pos"]) - 1
        conv_rate = np.mean(self.current_metrics.get("conv", []))
        current_step = self.get_current_step()

        return {
            "n_steps": n_steps,
            "convergence_rate": f"{conv_rate:.1%}" if conv_rate else "N/A",
            "current_step": current_step,
            "case_name": Path(
                self.loaded_case_info.get("deck_filename", "Unknown")
            ).name,
        }

    def get_header_status(self) -> Dict[str, str]:
        """
        Get header status information.

        Returns:
            Dictionary with status color and text
        """
        if self.has_data():
            summary = self.get_case_summary()
            return {
                "color": "#27ae60",
                "text": f"Case Loaded ({summary['n_steps']} steps)",
            }
        else:
            return {"color": "#e74c3c", "text": "No Data"}

    def get_cached_plot(self, plot_name: str):
        """Get cached plot if available and cache is valid."""
        if self._cache_invalidated:
            return None
        return self._cached_plots.get(plot_name)

    def cache_plot(self, plot_name: str, plot_figure):
        """Cache a plot figure."""
        self._cached_plots[plot_name] = plot_figure
        self._cache_invalidated = False

    def invalidate_cache(self):
        """Invalidate the plot cache (call when data changes)."""
        self._cache_invalidated = True
        self._cached_plots.clear()
