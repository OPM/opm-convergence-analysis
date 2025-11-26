"""
Unit tests for SimulationData model.
"""

import pytest
import pandas as pd
import numpy as np
from opm_convergence_analysis.core.models import SimulationData


class TestSimulationData:
    @pytest.fixture
    def valid_data(self):
        return {
            "iterations": pd.DataFrame(
                {
                    "step_index": [0, 0, 1, 1],
                    "iteration_index": [1, 2, 1, 2],
                    "Reservoir.Oil.CNV": [0.5, 0.0005, 0.4, 0.0001],
                }
            ),
            "steps": pd.DataFrame(
                {"time": [1.0, 2.0], "converged": [True, True]}, index=[0, 1]
            ),
            "metric_meta": {
                "Reservoir.Oil.CNV": {
                    "display_name": "Reservoir.Oil.CNV",
                    "group": "Convergence",
                    "tolerance": 1e-3,
                }
            },
        }

    def test_initialization(self, valid_data):
        model = SimulationData(**valid_data)
        assert model.n_steps == 2
        assert model.total_rows == 4
        assert model.total_newton_iterations == 4

    def test_validation_missing_columns(self, valid_data):
        # Missing step_index
        data = valid_data.copy()
        data["iterations"] = data["iterations"].drop(columns=["step_index"])

        with pytest.raises(ValueError, match="missing required column: 'step_index'"):
            SimulationData(**data)

        # Missing iteration_index
        data = valid_data.copy()
        data["iterations"] = valid_data["iterations"].drop(columns=["iteration_index"])

        with pytest.raises(
            ValueError, match="missing required column: 'iteration_index'"
        ):
            SimulationData(**data)

    def test_validation_invalid_types(self, valid_data):
        valid_data["iterations"]["step_index"] = valid_data["iterations"][
            "step_index"
        ].astype(float)

        with pytest.raises(
            ValueError, match="'step_index' column must be of integer type"
        ):
            SimulationData(**valid_data)

    def test_validation_bounds(self, valid_data):
        # Negative index
        data = valid_data.copy()
        data["iterations"] = data["iterations"].copy()
        data["iterations"].loc[0, "step_index"] = -1

        with pytest.raises(ValueError, match="step_index cannot be negative"):
            SimulationData(**data)

        # Index out of bounds
        data["iterations"].loc[0, "step_index"] = 2  # n_steps is 2, so max index is 1

        with pytest.raises(ValueError, match="exceeds number of steps"):
            SimulationData(**data)

    def test_validation_metadata(self, valid_data):
        valid_data["metric_meta"]["NonExistent"] = {}

        with pytest.raises(ValueError, match="defined in metadata but not found"):
            SimulationData(**valid_data)
