"""
Data handling functionality for the dashboard.

Contains data loading, processing, and state management for the dashboard.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np

# Import the new Reader and Model
from ..simulators import get_reader
from ..core.models import SimulationData
from ..core.analyzer import Analyzer


class DataHandler:
    """
    Handles data loading, processing, and state management for the dashboard.
    """

    def __init__(self):
        """Initialize the data handler."""
        self.current_model: Optional[SimulationData] = None
        self.current_errors: Optional[np.ndarray] = None
        self.current_labels: Optional[List[str]] = None
        self.current_metrics: Optional[Dict[str, Any]] = None
        self.current_savings: Optional[Dict[str, Any]] = None

        self.loaded_case_params: Dict[str, Any] = {}
        self.loaded_case_info: Dict[str, Any] = {}
        self.current_step_index: int = 0
        self.available_steps: List[int] = []

        # Flag to indicate if data has been loaded (for initial display)
        self._data_loaded: bool = False

    def load_case_from_path(self, input_path: str) -> bool:
        """
        Load case data from flexible input path using auto-detected reader.

        Args:
            input_path: Path to INFOITER, DBG, folder, or DATA file

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Loading case from: {input_path}")

            try:
                reader = get_reader(input_path)
                print(f"Detected format: {reader.__class__.__name__}")
            except ValueError as e:
                print(f"Reader detection failed: {e}")
                return False

            # Read data into generic model
            self.current_model = reader.read(input_path)

            # Backward compatibility: set current_data to None as we use current_model
            self.current_data = None

            # Update loaded info
            # Extract tolerances for display from metric_meta
            tols = {}
            for meta in self.current_model.metric_meta.values():
                if "group" in meta and "tolerance" in meta:
                    tols[meta["group"]] = meta["tolerance"]

            self.loaded_case_params = tols
            self.loaded_case_info = {
                "deck_filename": self.current_model.case_name,
                "simulator": self.current_model.simulator_name,
            }

            # Analyze convergence
            success = self._analyze_convergence()

            if success:
                self._update_available_steps()
                self._print_load_info()
                self._data_loaded = True

            return success

        except Exception as e:
            print(f"Error loading case from {input_path}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _analyze_convergence(self) -> bool:
        """
        Analyze convergence with current data and parameters.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.current_model is None:
                return False

            print(
                f"Analyzing convergence for {self.current_model.simulator_name} case..."
            )

            # Use the Analyzer (which now handles SimulationData)
            # We don't need to extract tolerances manually as they are in the model,
            # but we can pass overrides if we had UI controls for them.

            analyzer = Analyzer()
            self.current_errors, self.current_labels, self.current_metrics = (
                analyzer.analyze(self.current_model)
            )
            # Invalidate plot cache since analysis data changed
            self._cache_invalidated = True

            return True

        except Exception as e:
            print(f"Error analyzing convergence: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _update_available_steps(self):
        """Update the list of available steps."""
        if self.current_metrics is not None:
            flagged_steps = list(self.current_metrics.get("flaggedSteps", []))
            if not flagged_steps and self.current_model is not None:
                n_steps = self.current_model.n_steps
                flagged_steps = list(range(min(10, n_steps)))
            self.available_steps = flagged_steps

            # Ensure current_step_index is valid
            if self.current_step_index >= len(self.available_steps):
                self.current_step_index = 0

    def _print_load_info(self):
        """Print information about loaded case and parameters."""
        if self.loaded_case_params:
            print("Loaded parameters:")
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
        total_steps = len(self.available_steps)

        # Try to get report step and timestep information from model
        if self.current_model is not None and hasattr(self.current_model, "steps"):
            step_data = self.current_model.steps

            # Check if we have report/time step info
            # Note: step_data arrays are length n_steps
            if current_step < len(step_data):
                parts = []
                if "report_step" in step_data:
                    parts.append(
                        f"Report: {step_data['report_step'].iloc[current_step]}"
                    )
                if "time_step" in step_data:
                    parts.append(f"Time: {step_data['time_step'].iloc[current_step]}")

                if parts:
                    return f"Step {current_step} of {total_steps} total ({', '.join(parts)})"

        # Fallback to generic text
        return f"Step {current_step} of {total_steps} total"

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
    def data(self) -> Optional[SimulationData]:
        """Get current data (Model)."""
        return self.current_model

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
        return self.current_model is not None and self.current_errors is not None

    def get_case_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the current case.

        Returns:
            Dictionary with case summary information
        """
        if not self.has_data():
            return {}

        n_steps = self.current_model.n_steps if self.current_model else 0

        # Calculate convergence rate from metrics if available
        conv_rate = 0.0
        if self.current_metrics and "conv" in self.current_metrics:
            conv_rate = np.mean(self.current_metrics["conv"])

        current_step = self.get_current_step()
        case_name = self.current_model.case_name if self.current_model else "Unknown"

        return {
            "n_steps": n_steps,
            "convergence_rate": f"{conv_rate:.1%}" if conv_rate else "N/A",
            "current_step": current_step,
            "case_name": case_name,
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
