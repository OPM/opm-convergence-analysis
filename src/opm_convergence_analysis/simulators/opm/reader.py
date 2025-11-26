from typing import Union, Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
import re
from datetime import datetime

from ...core.base_reader import BaseReader
from ...core.models import SimulationData
from .infoiter import InfoIterReader
from .dbg_reader import DBGReader, find_case_files
from .well_parser import OPMWellFailureParser


class OPMReader(BaseReader):
    """
    Reader for OPM Flow simulations (INFOITER/DBG files) that produces generic SimulationData.
    """

    def can_read(self, source: Union[str, Path]) -> bool:
        source = Path(source)
        if source.suffix.upper() in [".INFOITER", ".DBG", ".DATA"]:
            return True
        if source.is_dir():
            return any(source.glob("*.INFOITER"))
        return False

    def read(self, source: Union[str, Path], **kwargs) -> SimulationData:
        source = Path(source)
        files = find_case_files(source)

        if not files["infoiter"]:
            raise ValueError(f"No INFOITER file found for {source}")

        # Use infoiter reader to parse text
        infoiter_reader = InfoIterReader()
        raw_data_dict = infoiter_reader.read_infoiter(str(files["infoiter"]))

        # Load parameters from DBG if available
        tolerances = {}
        case_info = {}
        initial_date = None

        if files["dbg"]:
            dbg_reader = DBGReader(files["dbg"])
            params = dbg_reader.get_convergence_parameters()
            case_info = dbg_reader.get_case_info()
            initial_date = dbg_reader.get_initial_date()

            if "cnv_tolerance" in params:
                tolerances["Convergence"] = params["cnv_tolerance"]
            if "mb_tolerance" in params:
                tolerances["Material Balance"] = params["mb_tolerance"]

        return self._convert_to_simulation_data(
            raw_data_dict, tolerances, case_info, initial_date
        )

    def _convert_to_simulation_data(
        self,
        data: Dict[str, Any],
        group_tolerances: Dict[str, float],
        case_info: Dict[str, Any],
        initial_date: Any,
    ) -> SimulationData:

        # 1. Construct Iterations DataFrame
        iter_dict = {}
        raw = data["raw"]

        # Basic columns
        iter_dict["iteration_index"] = raw.get("Iteration", [])

        # Well Status columns if available
        if "FailedWells" in raw:
            iter_dict["FailedWells"] = raw["FailedWells"]
        if "WellStatus" in raw:
            iter_dict["WellStatus"] = raw["WellStatus"]

        # We need step_index for each row.
        curve_pos = data["curve_pos"]
        n_rows = len(raw.get("Iteration", []))
        n_steps = len(curve_pos) - 1

        # Create step_index array efficiently
        step_indices = np.zeros(n_rows, dtype=int)
        for i in range(n_steps):
            start = curve_pos[i]
            end = curve_pos[i + 1]
            step_indices[start:end] = i

        iter_dict["step_index"] = step_indices

        # Metric Meta collection
        metric_meta = {}

        # Process Metrics (CNV and MB)
        def add_metrics(source_dict, group_name):
            if source_dict and "value" in source_dict:
                vals = source_dict["value"]  # Shape (rows, cols)
                labels = source_dict["label"]

                for i, label in enumerate(labels):
                    col_name = label

                    # Rename to structured format: Reservoir.{Phase}.{Type}
                    if label.startswith("CNV."):
                        phase = label.split(".")[1]
                        col_name = f"Reservoir.{phase}.CNV"
                    elif label.startswith("MB."):
                        phase = label.split(".")[1]
                        col_name = f"Reservoir.{phase}.MB"

                    iter_dict[col_name] = vals[:, i]

                    meta = {"display_name": col_name, "group": group_name}
                    if group_name in group_tolerances:
                        meta["tolerance"] = group_tolerances[group_name]

                    metric_meta[col_name] = meta

        if "cnv" in data:
            add_metrics(data["cnv"], "Convergence")
        if "mb" in data:
            add_metrics(data["mb"], "Material Balance")

        # Create Iterations DF
        iterations_df = pd.DataFrame(iter_dict)

        # Parse well failures into generic format
        well_failures = None
        if "WellStatus" in raw:
            well_failures = []
            for status_str in raw["WellStatus"]:
                failures_at_iter = OPMWellFailureParser.parse_well_status_string(
                    str(status_str)
                )
                well_failures.append(failures_at_iter)

        # 2. Construct Steps DataFrame
        step_data = {}
        first_indices = curve_pos[:-1]

        if "Time" in raw:
            step_data["time"] = raw["Time"][first_indices]

        if "ReportStep" in raw:
            step_data["report_step"] = raw["ReportStep"][first_indices]

        if "TimeStep" in raw:
            step_data["time_step"] = raw["TimeStep"][first_indices]

        # Calculate Dates
        if initial_date and "Time" in raw:
            from datetime import timedelta

            times = raw["Time"][first_indices]
            dates = []
            for t in times:
                try:
                    d = initial_date + timedelta(days=float(t))
                    dates.append(d)
                except:
                    dates.append(None)
            step_data["date"] = dates

        # Determine Convergence
        last_indices = curve_pos[1:] - 1
        converged = np.ones(n_steps, dtype=bool)
        if n_steps > 0 and "TimeStep" in raw and "ReportStep" in raw:
            steps_arr = np.column_stack([raw["ReportStep"], raw["TimeStep"]])

            for i in range(n_steps - 1):
                last_val = steps_arr[last_indices[i]]
                next_start_val = steps_arr[curve_pos[i + 1]]

                if np.any(next_start_val > last_val):
                    converged[i] = True
                else:
                    converged[i] = False

        step_data["converged"] = converged

        steps_df = pd.DataFrame(
            step_data, index=pd.RangeIndex(n_steps, name="step_index")
        )

        return SimulationData(
            iterations=iterations_df,
            steps=steps_df,
            metric_meta=metric_meta,
            simulator_name="OPM Flow",
            case_name=case_info.get("deck_filename", "Unknown"),
            well_failures=well_failures,
        )
