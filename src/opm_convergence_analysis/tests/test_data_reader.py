"""
Test suite for DataReader class.

This module tests the DataReader class against the reference MATLAB implementation
to ensure exact numerical compatibility.
"""

import numpy as np
import pytest
from pathlib import Path

from ..core.data_reader import DataReader


class TestDataReader:
    """Test cases for DataReader class."""

    @pytest.fixture
    def test_file_path(self):
        """Path to test INFOITER file."""
        test_dir = Path(__file__).parent / "reference_data"
        return test_dir / "NORNE_ATW2013.INFOITER"

    @pytest.fixture
    def data_reader(self):
        """DataReader instance for testing."""
        return DataReader()

    def test_file_exists(self, test_file_path):
        """Test that the reference data file exists."""
        assert test_file_path.exists(), f"Test file not found: {test_file_path}"

    def test_read_infoiter_basic(self, data_reader, test_file_path):
        """Test basic INFOITER file reading functionality."""
        result = data_reader.read_infoiter(str(test_file_path))

        # Verify return structure
        assert isinstance(result, dict)
        assert "mb" in result
        assert "cnv" in result
        assert "curve_pos" in result
        assert "raw" in result
        assert "penalty" in result

        # Verify mb structure
        mb = result["mb"]
        assert "value" in mb
        assert "label" in mb
        assert isinstance(mb["value"], np.ndarray)
        assert isinstance(mb["label"], list)

        # Verify cnv structure
        cnv = result["cnv"]
        assert "value" in cnv
        assert "label" in cnv
        assert isinstance(cnv["value"], np.ndarray)
        assert isinstance(cnv["label"], list)

        # Verify curve_pos
        curve_pos = result["curve_pos"]
        assert isinstance(curve_pos, np.ndarray)
        assert len(curve_pos) >= 2  # Should have at least start and end

        # Verify raw data
        raw = result["raw"]
        assert isinstance(raw, dict)
        assert "FailedWells" in raw
        assert isinstance(raw["FailedWells"], np.ndarray)

    def test_reference_values(self, data_reader, test_file_path):
        """Test against known reference values from MATLAB implementation."""
        result = data_reader.read_infoiter(str(test_file_path))

        # These values should match the MATLAB test reference
        # From test_reference_values.m, we know there should be 1989 total iterations
        # across all timesteps

        curve_pos = result["curve_pos"]
        n_timesteps = len(curve_pos) - 1

        # Count total iterations (excluding iteration 0 for each timestep)
        total_iterations = 0
        for i in range(n_timesteps):
            step_iterations = (
                curve_pos[i + 1] - curve_pos[i] - 1
            )  # -1 to exclude iteration 0
            total_iterations += step_iterations

        # This should match the MATLAB reference value
        assert (
            total_iterations == 1989
        ), f"Expected 1989 total iterations, got {total_iterations}"

        # Verify we have the expected number of MB and CNV columns
        mb_labels = result["mb"]["label"]
        cnv_labels = result["cnv"]["label"]

        # NORNE case should have 3 components: Oil, Water, Gas
        expected_mb_labels = ["MB.Oil", "MB.Water", "MB.Gas"]
        expected_cnv_labels = ["CNV.Oil", "CNV.Water", "CNV.Gas"]

        assert mb_labels == expected_mb_labels, f"MB labels mismatch: {mb_labels}"
        assert cnv_labels == expected_cnv_labels, f"CNV labels mismatch: {cnv_labels}"

        # Verify data shapes
        n_rows = len(result["raw"]["Iteration"])
        assert result["mb"]["value"].shape == (n_rows, 3)
        assert result["cnv"]["value"].shape == (n_rows, 3)

    def test_validate_infoiter_format(self, data_reader, test_file_path):
        """Test INFOITER format validation."""
        assert data_reader.validate_infoiter_format(str(test_file_path))

        # Test with non-existent file
        assert not data_reader.validate_infoiter_format("nonexistent.file")

    def test_get_file_info(self, data_reader, test_file_path):
        """Test file information extraction."""
        info = data_reader.get_file_info(str(test_file_path))

        assert "filename" in info
        assert "total_rows" in info
        assert "total_columns" in info
        assert "n_timesteps" in info
        assert "total_iterations" in info
        assert "mb_columns" in info
        assert "cnv_columns" in info

        # Verify expected values
        assert info["mb_columns"] == 3  # Oil, Water, Gas
        assert info["cnv_columns"] == 3  # Oil, Water, Gas
        assert info["total_iterations"] == 1989  # Reference value

    def test_file_not_found(self, data_reader):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            data_reader.read_infoiter("nonexistent.file")

    def test_well_status_processing(self, data_reader, test_file_path):
        """Test well status processing for failed wells."""
        result = data_reader.read_infoiter(str(test_file_path))

        # Verify FailedWells array
        failed_wells = result["raw"]["FailedWells"]
        assert isinstance(failed_wells, np.ndarray)
        assert failed_wells.dtype == bool
        assert len(failed_wells) == len(result["raw"]["Iteration"])

        # For NORNE test case, most wells should be CONV (not failed)
        # This is just a sanity check - most entries should be False
        failure_rate = np.mean(failed_wells)
        assert failure_rate < 0.5, f"Unexpectedly high failure rate: {failure_rate}"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])
