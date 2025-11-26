import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from .well_data import WellFailure


@dataclass
class SimulationData:
    """
    Generic container for convergence data from any reservoir simulator.

    Attributes:
        iterations (pd.DataFrame):
            Main data table with one row per iteration.
            Must contain columns:
            - 'step_index' (int): Index of the time step (0-based, contiguous integers corresponding to steps DataFrame index)
            - 'iteration_index' (int): Iteration number within the step (usually 1-based for Newton iterations)
            - Any number of metric columns (e.g., 'pressure_residual', 'mb_error_oil')

        steps (pd.DataFrame):
            Metadata table with one row per time step.
            Index should correspond to 'step_index' from iterations table.
            Common columns:
            - 'time': Simulation time (e.g. days)
            - 'date': Simulation date (datetime)
            - 'converged': Boolean status
            - 'report_step': Report step index (optional)
            - 'time_step': Simulator time step index (optional)

        metric_meta (Dict[str, Dict[str, Any]]):
            Metadata for metrics found in iterations DataFrame.
            Key is the column name in iterations DataFrame.
            Value is a dict containing:
            - 'display_name': Human readable name
            - 'tolerance': Convergence threshold (optional)
            - 'group': Group name (e.g., 'Convergence', 'Material Balance') (optional)

        simulator_name (str): Name of the simulator (e.g., "OPM Flow")
        case_name (str): Name of the simulation case
    """

    iterations: pd.DataFrame
    steps: pd.DataFrame
    metric_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    simulator_name: str = "Generic"
    case_name: str = "Unknown"
    well_failures: Optional[List[List[WellFailure]]] = None  # List per iteration

    def __post_init__(self):
        """Validate data upon initialization."""
        self.validate()

    def validate(self):
        """
        Validate the consistency and structure of the simulation data.

        Raises:
            ValueError: If data is invalid or inconsistent.
        """
        # 1. Check required columns in iterations
        required_cols = ["step_index", "iteration_index"]
        for col in required_cols:
            if col not in self.iterations.columns:
                raise ValueError(
                    f"Iterations DataFrame missing required column: '{col}'"
                )

        # 2. Check step_index integrity
        if not self.iterations.empty:
            # Ensure step_index is integer
            if not pd.api.types.is_integer_dtype(self.iterations["step_index"]):
                raise ValueError("'step_index' column must be of integer type")

            # Check bounds
            # steps DataFrame defines the universe of steps 0..N-1
            n_steps = len(self.steps)
            max_step_idx = self.iterations["step_index"].max()
            min_step_idx = self.iterations["step_index"].min()

            if max_step_idx >= n_steps:
                raise ValueError(
                    f"Max step_index ({max_step_idx}) in iterations exceeds "
                    f"number of steps defined ({n_steps})"
                )

            if min_step_idx < 0:
                raise ValueError("step_index cannot be negative")

        # 3. Check metadata consistency
        for col_name in self.metric_meta:
            if col_name not in self.iterations.columns:
                raise ValueError(
                    f"Metric '{col_name}' defined in metadata but not found in iterations DataFrame"
                )

    @property
    def n_steps(self) -> int:
        """Number of time steps defined."""
        return len(self.steps)

    @property
    def total_rows(self) -> int:
        """Total number of rows in the iterations DataFrame."""
        return len(self.iterations)

    @property
    def total_newton_iterations(self) -> int:
        """
        Total number of Newton iterations.
        This typically excludes the initial residual check (iteration 0).
        Assumes 'iteration_index' column exists where 0 = initial, 1+ = Newton.
        """
        if "iteration_index" in self.iterations.columns:
            return (self.iterations["iteration_index"] > 0).sum()
        return 0

    def get_curve_pos(self) -> List[int]:
        """
        Get start positions of steps in the iterations dataframe.
        Useful for low-level plotting optimization.
        """
        # Calculate start indices of each step
        # We assume iterations are sorted by step_index

        # Get counts per step, ensure all steps are present even if empty
        if self.iterations.empty:
            return [0] * (self.n_steps + 1)

        counts = self.iterations["step_index"].value_counts().sort_index()

        # Ensure we cover all steps from 0 to max_step defined in steps table
        # Use n_steps instead of max_step to cover full range if steps are empty?
        # The steps DataFrame defines the number of steps.
        n_defined_steps = self.n_steps

        full_counts = counts.reindex(range(n_defined_steps), fill_value=0)
        cumulative = full_counts.cumsum()

        return [0] + cumulative.tolist()
