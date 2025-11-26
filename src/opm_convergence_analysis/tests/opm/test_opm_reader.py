"""
Test suite for OPM Flow support (Reader and SimulationData).

This module validates that the OPMReader correctly parses OPM Flow output files
(INFOITER/DBG) and converts them into the generic SimulationData format.
It serves as a reference for how a Simulator Reader should be implemented and tested.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from opm_convergence_analysis.simulators import OPMReader
from opm_convergence_analysis.core.models import SimulationData
from opm_convergence_analysis.core.analyzer import Analyzer


class TestOPMReader:
    """Test cases for OPMReader and its generic SimulationData output."""

    @pytest.fixture
    def test_file_path(self):
        """Path to test INFOITER file (Norne case)."""
        test_dir = Path(__file__).parent / "data"
        return test_dir / "NORNE_ATW2013.INFOITER"

    @pytest.fixture
    def reader(self):
        """OPMReader instance."""
        return OPMReader()

    @pytest.fixture
    def loaded_data(self, reader, test_file_path) -> SimulationData:
        """Load data using OPMReader."""
        return reader.read(str(test_file_path))

    def test_can_read(self, reader, test_file_path):
        """Test file format detection."""
        assert reader.can_read(test_file_path)
        assert reader.can_read(str(test_file_path))
        assert reader.can_read("case.INFOITER")
        assert reader.can_read("case.DBG")
        assert reader.can_read("case.DATA")

        # Should not read unrelated files
        assert not reader.can_read("image.png")
        assert not reader.can_read("script.py")

    def test_read_returns_simulation_data(self, loaded_data):
        """Test that read returns a valid SimulationData object."""
        assert isinstance(loaded_data, SimulationData)
        # Verify basic validity via the model's own validation
        loaded_data.validate()

    def test_reference_values_dimensions(self, loaded_data):
        """Test against known reference values for the Norne case."""
        # Total rows in iterations dataframe
        # (269 time steps, some with multiple iterations)
        # Reference: 2258 rows total (including iteration 0s)
        assert loaded_data.total_rows == 2258

        # Total Newton iterations
        # TODO: This mathces reference data,
        # but not the total iterations in the DBG file. Check if this is correct.
        # Reference: 1989 iterations
        assert loaded_data.total_newton_iterations == 1989

        # Number of time steps
        assert loaded_data.n_steps == 269

    def test_column_renaming_and_metadata(self, loaded_data):
        """
        Test that raw OPM columns (CNV.Oil) are correctly renamed to
        generic format (Reservoir.Oil.CNV) and have metadata.
        """
        df = loaded_data.iterations
        meta = loaded_data.metric_meta

        expected_columns = [
            "Reservoir.Oil.CNV",
            "Reservoir.Water.CNV",
            "Reservoir.Gas.CNV",
            "Reservoir.Oil.MB",
            "Reservoir.Water.MB",
            "Reservoir.Gas.MB",
        ]

        for col in expected_columns:
            # 1. Check column exists in DataFrame
            assert col in df.columns, f"Missing expected column: {col}"

            # 2. Check metadata exists
            assert col in meta, f"Missing metadata for: {col}"

            # 3. Check grouping
            expected_group = "Convergence" if "CNV" in col else "Material Balance"
            assert meta[col]["group"] == expected_group

    def test_tolerances_from_dbg(self, loaded_data):
        """Test that tolerances are correctly loaded from the sibling DBG file."""
        meta = loaded_data.metric_meta

        # Check Oil CNV tolerance (should be 1e-3 from DBG)
        cnv_meta = meta["Reservoir.Oil.CNV"]
        assert cnv_meta["tolerance"] == 1e-3

        # Check Oil MB tolerance (should be 1e-7 from DBG)
        mb_meta = meta["Reservoir.Oil.MB"]
        assert mb_meta["tolerance"] == 1e-7

    def test_well_status_columns(self, loaded_data):
        """Test that OPM-specific well status columns are preserved."""
        df = loaded_data.iterations

        assert "FailedWells" in df.columns
        assert "WellStatus" in df.columns

        # Check types
        assert df["FailedWells"].dtype == bool
        # WellStatus might be object (string) or int depending on parsing,
        # but for this case likely object/string representation or specific codes

    def test_steps_metadata(self, loaded_data):
        """Test steps DataFrame content."""
        steps = loaded_data.steps

        # Standard generic columns
        assert "time" in steps.columns
        assert "date" in steps.columns
        assert "converged" in steps.columns

        # OPM-specific/Standard metadata
        assert "report_step" in steps.columns
        assert "time_step" in steps.columns

        # Check convergence
        # In this successful run, most/all steps should be converged
        assert steps["converged"].all() or steps["converged"].mean() > 0.95

    def test_analyzer_compatibility(self, loaded_data):
        """
        Test that the produced SimulationData is compatible with the Analyzer.
        This serves as an integration test ensuring the generic contract is met.
        """
        analyzer = Analyzer()
        errors, labels, metrics = analyzer.analyze(loaded_data)

        # Check error calculation
        assert errors.shape == (loaded_data.total_rows, 6)  # 6 metrics
        assert metrics["dist"].shape == (loaded_data.total_rows,)

        # Check that we have the renamed labels
        assert "Reservoir.Oil.CNV" in labels

    def test_missing_file_error(self, reader):
        """Test error handling for missing files."""
        with pytest.raises(ValueError, match="No INFOITER file found"):
            reader.read("nonexistent_file.INFOITER")

    def test_well_failures_generic_format(self, loaded_data):
        """Test that well failures are parsed into generic format."""
        if loaded_data.well_failures is None:
            # Well failures might not be present in test data
            return

        assert isinstance(loaded_data.well_failures, list)

        # Check structure of well failures
        for iteration_failures in loaded_data.well_failures:
            assert isinstance(iteration_failures, list)
            for failure in iteration_failures:
                # Check it's the generic WellFailure type
                assert hasattr(failure, "well_name")
                assert hasattr(failure, "failure_type")
                assert hasattr(failure, "get_display_reason")

                # Verify well_name is a string
                assert isinstance(failure.well_name, str)

                # Verify we can get display reason
                reason = failure.get_display_reason()
                assert isinstance(reason, str)
