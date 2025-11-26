# Generic Simulator Support

This library is designed to be simulator-agnostic. While it provides built-in support for OPM Flow, it can be extended to support any reservoir simulator (e.g., Eclipse, Intersect, JutulDarcy) by implementing a custom Reader.

## Data Requirements

To perform the analysis, the reader must produce a `SimulationData` object containing the following:

### 1. Iterations DataFrame (`iterations`)
A pandas DataFrame with one row per Newton iteration.

**Required Columns:**
*   `step_index` (int): 0-based index linking to the `steps` DataFrame. Must be contiguous integers (0, 1, 2, ...).
*   `iteration_index` (int): 1-based index of the Newton iteration within the step.
    *   0 is typically reserved for the initial residual check.
    *   1+ are the actual Newton iterations.

**Metric Columns:**
*   Any number of columns representing convergence metrics (e.g., `Reservoir.Oil.CNV`, `Reservoir.Water.MB`).
*   **Naming Convention**: `Region.Phase.Type` (e.g., `Reservoir.Oil.CNV`).

### Optional: Well Failures

To report well convergence issues, provide the `well_failures` parameter: a list with one entry per iteration, where each entry contains `WellFailure` objects.

```python
from opm_convergence_analysis.core.well_data import WellFailure

well_failures = [
    [WellFailure("PROD-01", failure_type="ControlRate")],
    [],
    [WellFailure("INJ-02", failure_type="MassBalance", phase="Gas")],
    []
]
```

The `failure_type` is simulator-specific (e.g., OPM uses `"MassBalance"`, `"ControlRate"`, `"Pressure"`).

### 2. Steps DataFrame (`steps`)
A pandas DataFrame containing metadata for each time step. The index must correspond to the `step_index` in the iterations DataFrame.

**Standard Columns:**
*   `time` (float): Simulation time (e.g., days).
*   `date` (datetime): Simulation date.
*   `converged` (bool): Whether the step successfully converged.
*   `report_step` (int, optional): Report step index.
*   `time_step` (int, optional): Simulator time step index.

### 3. Metadata Dictionary (`metric_meta`)
A dictionary defining properties for each metric column in the `iterations` DataFrame.

**Structure:**
```python
{
    "Reservoir.Oil.CNV": {
        "display_name": "Reservoir Oil Convergence",
        "group": "Convergence",       # e.g. "Convergence", "Material Balance"
        "tolerance": 1e-3             # Threshold for failure analysis
    }
}
```

## Implementing a Custom Reader

To support a new simulator, create a class that inherits from `opm_convergence_analysis.core.BaseReader`.

### 1. Inherit from BaseReader

```python
from opm_convergence_analysis.core import BaseReader, SimulationData
from typing import Union
from pathlib import Path

class MySimulatorReader(BaseReader):
    def can_read(self, source: Union[str, Path]) -> bool:
        """Return True if this reader can parse the given source."""
        # Logic to check file extension or content
        return str(source).endswith(".myformat")

    def read(self, source: Union[str, Path], **kwargs) -> SimulationData:
        """Parse the source and return a SimulationData object."""
        # ... parsing implementation ...
        pass
```

### 2. Return SimulationData

The `read` method must return a valid `SimulationData` object.

```python
import pandas as pd
from opm_convergence_analysis.core import SimulationData
from opm_convergence_analysis.core.well_data import WellFailure

def read(self, source, **kwargs):
    # ... parsing logic ...

    # 1. Create Iterations DataFrame
    iterations = pd.DataFrame({
        'step_index': [0, 0, 1, 1],
        'iteration_index': [1, 2, 1, 2],
        'Reservoir.Oil.CNV': [0.5, 0.0005, 0.4, 0.0001],
        'Reservoir.Oil.MB': [0.01, 0.00001, 0.02, 0.00001]
    })

    # 2. Create Steps DataFrame
    steps = pd.DataFrame({
        'time': [1.0, 2.0],
        'date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')],
        'converged': [True, True]
    })

    # 3. Define Metadata
    meta = {
        'Reservoir.Oil.CNV': {
            'display_name': 'Reservoir.Oil.CNV',
            'group': 'Convergence',
            'tolerance': 1e-3
        },
        'Reservoir.Oil.MB': {
            'display_name': 'Reservoir.Oil.MB',
            'group': 'Material Balance',
            'tolerance': 1e-7
        }
    }
    
    # 4. (Optional) Parse well failures into generic format
    # Use simulator-specific failure type strings
    well_failures = [
        [WellFailure("WellA", failure_type="ControlRate")],  # Iter 1
        [],  # Iter 2
        [WellFailure("WellB", failure_type="MassBalance", phase="Gas")],  # Iter 3
        [],  # Iter 4
    ]

    return SimulationData(
        iterations=iterations,
        steps=steps,
        metric_meta=meta,
        simulator_name="MySimulator",
        case_name="Case1",
        well_failures=well_failures  # Optional
    )
```

## Adding Support for New Simulators

1.  Create a new folder in `src/opm_convergence_analysis/simulators/`.
2.  Implement your `Reader` class in `reader.py`.
3.  Expose it in `__init__.py`.
4.  (Optional) Register it in the auto-detection logic if you want `app.py` to automatically pick it up (currently requires manual selection or unified detection).
