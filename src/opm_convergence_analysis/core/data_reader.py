"""
DataReader class for reading and parsing INFOITER files from OPM Flow simulations.

This module provides the DataReader class which replicates the functionality of
the MATLAB readInfoIter.m function.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from ..utils.helpers import (
    compute_start_positions,
    process_well_status,
    collect_columns,
)


class DataReader:
    """
    Reader for OPM Flow INFOITER files.

    This class provides functionality to read and parse INFOITER files from
    OPM Flow simulations, extracting material balance (MB_*), convergence (CNV_*),
    and penalty information.
    """

    def __init__(self):
        """Initialize the DataReader."""
        pass

    def read_infoiter(self, filename: str) -> Dict[str, Any]:
        """
        Read INFOITER file and return parsed data structure.

        Args:
            filename: Path to INFOITER file

        Returns:
            Dictionary containing:
                - mb: Material balance data structure
                - cnv: Convergence data structure
                - curve_pos: Array of curve start positions
                - raw: Raw data from file
                - penalty: Penalty data structure

        Raises:
            FileNotFoundError: If file cannot be opened
            ValueError: If file format is invalid
        """
        try:
            # Read the file using a custom parser to handle lines with extra well failure info
            # Some lines have additional well failure details beyond the standard column count
            df = self._parse_infoiter_file(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Failed to open INFOITER file '{filename}'")
        except Exception as e:
            raise ValueError(f"Failed to parse INFOITER file '{filename}': {str(e)}")

        # Get column headers and values
        header = list(df.columns)
        values = [df[col].values for col in header]

        # Process categorical columns (WellStatus)
        categorical_columns = ["WellStatus"]
        for col in categorical_columns:
            if col in header:
                idx = header.index(col)
                # Convert to string array for processing
                values[idx] = df[col].astype(str).values

        # Create raw data structure
        raw = self._create_raw_structure(df, header, values)

        # Extract material balance, convergence, and penalty data
        mb = collect_columns(values, header, r"^MB_")
        cnv = collect_columns(values, header, r"^CNV_")
        penalty = collect_columns(values, header, r"^Penalty")

        # Compute curve positions from iteration data
        if "Iteration" not in df.columns:
            raise ValueError("INFOITER file must contain 'Iteration' column")

        curve_pos = compute_start_positions(df["Iteration"].values)

        # Process well status to identify failed wells
        if "WellStatus" in df.columns:
            well_status_strings = df["WellStatus"].astype(str).values
            raw["WellStatus"] = well_status_strings  # Store raw well status strings
            raw["FailedWells"] = process_well_status(well_status_strings.tolist())
        else:
            raw["FailedWells"] = np.zeros(len(df), dtype=bool)

        # Look for corresponding DBG file
        dbg_file = None
        if filename:
            from pathlib import Path

            infoiter_path = Path(filename)
            stem = infoiter_path.stem
            dbg_path = infoiter_path.parent / f"{stem}.DBG"
            if dbg_path.exists():
                dbg_file = str(dbg_path)

        # Pre-calculate all step dates for performance optimization
        # This avoids reading the DBG file repeatedly during navigation
        step_dates = self._precalculate_step_dates(raw, curve_pos, dbg_file)

        return {
            "mb": mb,
            "cnv": cnv,
            "curve_pos": curve_pos,
            "raw": raw,
            "penalty": penalty,
            "dbg_file": dbg_file,
            "step_dates": step_dates,  # Pre-calculated dates for all steps
        }

    def _parse_infoiter_file(self, filename: str) -> pd.DataFrame:
        """
        Custom parser for INFOITER files that handles lines with extra well failure information.

        Some lines have additional columns beyond the standard column count, containing well failure details
        in the format: { WellName FailureType Severity=X Phase=Y }

        This method takes only the expected number of columns from each line to match MATLAB behavior:
        - 18 columns for 2-phase models (Oil, Gas)
        - 20 columns for 3-phase models (Oil, Water, Gas)

        Args:
            filename: Path to INFOITER file

        Returns:
            DataFrame with consistent column count (18 or 20 columns)
        """
        data_rows = []
        header = None
        lines_processed = 0

        with open(filename, "r") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                cols = line.split()
                lines_processed += 1

                if line_num == 0:
                    # Header line - expect 18 columns (2-phase) or 20 columns (3-phase)
                    if len(cols) not in [18, 20]:
                        raise ValueError(
                            f"Expected 18 or 20 columns in header, got {len(cols)}"
                        )
                    header = cols
                    expected_cols = len(cols)  # Store the expected number of columns
                else:
                    # Data line - take only the expected number of columns if more are present
                    if len(cols) >= expected_cols:
                        # For WellStatus (last column), preserve all remaining text as one field
                        if len(cols) > expected_cols:
                            # Join all remaining columns into the last column (WellStatus)
                            row_data = cols[
                                : expected_cols - 1
                            ]  # All columns except last
                            well_status = " ".join(
                                cols[expected_cols - 1 :]
                            )  # Join remaining as WellStatus
                            row_data.append(well_status)
                            data_rows.append(row_data)
                        else:
                            data_rows.append(cols[:expected_cols])
                    elif len(cols) > 0:
                        # This shouldn't happen in a well-formed INFOITER file
                        raise ValueError(
                            f"Line {line_num + 1} has only {len(cols)} columns, expected at least {expected_cols}"
                        )

        if header is None:
            raise ValueError("No header found in INFOITER file")

        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=header)

        # Convert numeric columns
        numeric_cols = header[:-1]  # All except WellStatus
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _create_raw_structure(
        self, df: pd.DataFrame, header: List[str], values: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Create raw data structure from DataFrame.

        Args:
            df: Pandas DataFrame with file data
            header: List of column names
            values: List of column value arrays

        Returns:
            Dictionary with column names as keys and arrays as values
        """
        raw = {}

        for col in header:
            if col in df.columns:
                raw[col] = df[col].values

        return raw

    def validate_infoiter_format(self, filename: str) -> bool:
        """
        Validate that the file has the expected INFOITER format.

        Args:
            filename: Path to file to validate

        Returns:
            True if file appears to be valid INFOITER format
        """
        try:
            # Try to read first few lines
            df = pd.read_csv(filename, sep=r"\s+", nrows=10)

            # Check for required columns
            required_cols = ["ReportStep", "TimeStep", "Iteration"]
            for col in required_cols:
                if col not in df.columns:
                    return False

            # Check for MB_ or CNV_ columns
            has_mb = any(col.startswith("MB_") for col in df.columns)
            has_cnv = any(col.startswith("CNV_") for col in df.columns)

            if not (has_mb or has_cnv):
                return False

            return True

        except Exception:
            return False

    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """
        Get basic information about an INFOITER file.

        Args:
            filename: Path to INFOITER file

        Returns:
            Dictionary with file information
        """
        try:
            # Use the same custom parser as read_infoiter to handle extra columns
            df = self._parse_infoiter_file(filename)

            # Count columns by type
            mb_cols = [col for col in df.columns if col.startswith("MB_")]
            cnv_cols = [col for col in df.columns if col.startswith("CNV_")]
            penalty_cols = [col for col in df.columns if col.startswith("Penalty")]

            # Count timesteps and iterations
            if "Iteration" in df.columns:
                curve_pos = compute_start_positions(df["Iteration"].values)
                n_timesteps = len(curve_pos) - 1
                total_iterations = (
                    len(df) - n_timesteps
                )  # Subtract initial iterations (iteration 0)
            else:
                n_timesteps = 0
                total_iterations = 0

            return {
                "filename": filename,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "n_timesteps": n_timesteps,
                "total_iterations": total_iterations,
                "mb_columns": len(mb_cols),
                "cnv_columns": len(cnv_cols),
                "penalty_columns": len(penalty_cols),
                "columns": list(df.columns),
                "mb_column_names": mb_cols,
                "cnv_column_names": cnv_cols,
                "penalty_column_names": penalty_cols,
            }

        except Exception as e:
            return {"filename": filename, "error": str(e)}

    def calculate_date_from_time_step(
        self, initial_date: datetime, time_days: float
    ) -> datetime:
        """
        Calculate the actual date from initial date and time step in days.

        Args:
            initial_date: Initial simulation date
            time_days: Time step in days (from INFOITER file)

        Returns:
            Calculated datetime
        """
        return initial_date + timedelta(days=time_days)

    def get_step_date_info(
        self, data: Dict[str, Any], step_idx: int
    ) -> Tuple[Optional[datetime], Optional[str]]:
        """
        Get date information for a specific step using pre-calculated dates.

        This method is fast because it uses pre-calculated dates from the data loading phase,
        avoiding the need to read DBG files or perform date calculations during navigation.

        Args:
            data: Raw data dictionary from read_infoiter (must contain 'step_dates')
            step_idx: Step index (0-based)

        Returns:
            Tuple of (datetime object, formatted date string)
        """
        if not data or "step_dates" not in data:
            return None, None

        step_dates = data["step_dates"]
        if step_idx >= len(step_dates):
            return None, None

        formatted_date = step_dates[step_idx]
        if formatted_date is None:
            return None, None

        # Convert formatted date back to datetime object
        try:
            calculated_date = datetime.strptime(formatted_date, "%d-%b-%Y")
            return calculated_date, formatted_date
        except ValueError:
            return None, None

    def _precalculate_step_dates(
        self, raw_data: Dict[str, Any], curve_pos: List[int], dbg_file: Optional[str]
    ) -> List[Optional[str]]:
        """
        Pre-calculate all step dates for performance optimization.

        Args:
            raw_data: Raw data dictionary
            curve_pos: Curve positions array
            dbg_file: Path to DBG file

        Returns:
            List of formatted date strings for each step
        """
        step_dates = []

        # Get initial date once
        initial_date = None
        if dbg_file:
            try:
                from .dbg_reader import DBGReader

                dbg_reader = DBGReader(dbg_file)
                initial_date = dbg_reader.get_initial_date()
            except Exception:
                pass

        # Pre-calculate dates for all steps
        for step_idx in range(len(curve_pos) - 1):
            if step_idx >= len(curve_pos) - 1:
                step_dates.append(None)
                continue

            step_start_idx = curve_pos[step_idx]

            # Get time information from INFOITER data
            if "Time" not in raw_data or step_start_idx >= len(raw_data["Time"]):
                step_dates.append(None)
                continue

            time_days = raw_data["Time"][step_start_idx]

            if initial_date is not None:
                calculated_date = self.calculate_date_from_time_step(
                    initial_date, time_days
                )
                formatted_date = calculated_date.strftime("%d-%b-%Y")
                step_dates.append(formatted_date)
            else:
                step_dates.append(None)

        return step_dates


def read_infoiter(filename: str) -> Dict[str, Any]:
    """
    Convenience function to read INFOITER file.

    Args:
        filename: Path to INFOITER file

    Returns:
        Parsed data structure
    """
    reader = DataReader()
    return reader.read_infoiter(filename)
