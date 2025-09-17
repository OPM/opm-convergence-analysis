"""
Test suite that validates Python implementation against reference values.

This module tests the convergence analysis functionality to ensure
proper calculation of error metrics and convergence indicators.
"""

import pytest
import numpy as np
from pathlib import Path

from ..core.data_reader import DataReader
from ..core.analyzer import Analyzer
from ..core.dbg_reader import DBGReader


class TestReferenceValues:
    """Test cases that validate convergence analysis functionality."""

    @pytest.fixture
    def test_file_path(self):
        """Path to test INFOITER file."""
        test_dir = Path(__file__).parent / "reference_data"
        return test_dir / "NORNE_ATW2013.INFOITER"

    @pytest.fixture
    def test_dbg_path(self):
        """Path to test DBG file."""
        test_dir = Path(__file__).parent / "reference_data"
        return test_dir / "NORNE_ATW2013.DBG"

    @pytest.fixture
    def loaded_data(self, test_file_path):
        """Load INFOITER data using DataReader."""
        reader = DataReader()
        return reader.read_infoiter(str(test_file_path))

    @pytest.fixture
    def dbg_tolerances(self, test_dbg_path):
        """Load tolerances from DBG file."""
        dbg_reader = DBGReader(str(test_dbg_path))
        return {
            "mb": dbg_reader.parameters.get("ToleranceMb", 1e-7),
            "cnv": dbg_reader.parameters.get("ToleranceCnv", 1e-3),
        }

    @pytest.fixture
    def analysis_results(self, loaded_data, dbg_tolerances):
        """Run analysis using Analyzer with tolerances from DBG file."""
        analyzer = Analyzer()
        errors, labels, metrics = analyzer.analyze(
            loaded_data,
            tol=dbg_tolerances,
        )
        return errors, labels, metrics

    def test_data_loading_reference_values(self, loaded_data):
        """Test that data loading produces expected reference values."""
        # From MATLAB: should have 1989 total iterations
        curve_pos = loaded_data["curve_pos"]
        n_timesteps = len(curve_pos) - 1

        total_iterations = 0
        for i in range(n_timesteps):
            step_iterations = curve_pos[i + 1] - curve_pos[i] - 1
            total_iterations += step_iterations

        assert (
            total_iterations == 1989
        ), f"Expected 1989 total iterations, got {total_iterations}"

        # Should have 3 MB and 3 CNV columns for NORNE case
        assert len(loaded_data["mb"]["label"]) == 3
        assert len(loaded_data["cnv"]["label"]) == 3
        assert loaded_data["mb"]["label"] == ["MB.Oil", "MB.Water", "MB.Gas"]
        assert loaded_data["cnv"]["label"] == ["CNV.Oil", "CNV.Water", "CNV.Gas"]

    def test_dbg_tolerance_loading(self, dbg_tolerances):
        """Test that tolerances are correctly loaded from DBG file."""
        # Check that tolerances are loaded and have reasonable values
        assert "mb" in dbg_tolerances
        assert "cnv" in dbg_tolerances
        assert dbg_tolerances["mb"] > 0
        assert dbg_tolerances["cnv"] > 0

        # For NORNE case, we expect specific tolerance values from the DBG file
        # These should match the values in the DBG file (1e-7 for MB, 1e-3 for CNV)
        assert (
            dbg_tolerances["mb"] == 1e-7
        ), f"Expected MB tolerance 1e-7, got {dbg_tolerances['mb']}"
        assert (
            dbg_tolerances["cnv"] == 1e-3
        ), f"Expected CNV tolerance 1e-3, got {dbg_tolerances['cnv']}"

    def test_analysis_reference_values(
        self, loaded_data, analysis_results, dbg_tolerances
    ):
        """Test that analysis produces expected reference values."""
        errors, labels, metrics = analysis_results

        # Check basic structure
        expected_rows = len(loaded_data["raw"]["Iteration"])
        expected_cols = len(loaded_data["cnv"]["label"]) + len(
            loaded_data["mb"]["label"]
        )

        assert errors.shape == (expected_rows, expected_cols)
        assert len(labels) == expected_cols
        assert labels == [
            "CNV.Oil",
            "CNV.Water",
            "CNV.Gas",
            "MB.Oil",
            "MB.Water",
            "MB.Gas",
        ]

        # Check metrics structure
        assert len(metrics["fail"]) == expected_rows
        assert len(metrics["dist"]) == expected_rows
        assert len(metrics["conv"]) == len(loaded_data["curve_pos"]) - 1

        # Check data types and ranges
        assert errors.dtype == np.float64
        assert np.all(errors >= 0), "All errors should be non-negative"
        assert metrics["fail"].dtype in [np.int64, np.int32]
        assert np.all(metrics["fail"] >= 0), "All failure counts should be non-negative"
        assert metrics["dist"].dtype == np.float64
        assert np.all(metrics["dist"] >= 0), "All distances should be non-negative"

    def test_convergence_statistics_reference(self, analysis_results):
        """Test convergence statistics match expected ranges."""
        _, _, metrics = analysis_results

        conv_rate = np.mean(metrics["conv"])

        # For NORNE case, we expect high convergence rate (>95%)
        assert conv_rate > 0.95, f"Expected convergence rate > 95%, got {conv_rate:.1%}"

        # Should have 269 timesteps
        assert (
            len(metrics["conv"]) == 269
        ), f"Expected 269 timesteps, got {len(metrics['conv'])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
