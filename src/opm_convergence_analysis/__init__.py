"""
Reservoir Simulation Convergence Analysis Library

This package provides tools for analyzing convergence behavior in reservoir
simulation runs. It is designed to be simulator-agnostic while providing
built-in support for OPM Flow.

Main components:
- Simulators: Readers for various simulator output formats (e.g., OPM Flow)
- Analyzer: Calculate error metrics and convergence indicators
- Visualization: Plot convergence analysis results

Example usage:
    from opm_convergence_analysis.simulators import get_reader
    from opm_convergence_analysis.core import Analyzer
    from opm_convergence_analysis.visualization import create_plotter

    # 1. Load data (auto-detected format)
    reader = get_reader("simulation.INFOITER")
    data = reader.read("simulation.INFOITER")

    # 2. Analyze convergence
    analyzer = Analyzer()
    errors, labels, metrics = analyzer.analyze(data)

    # 3. Visualize results
    plotter = create_plotter()
    fig = plotter.create_dashboard(data, errors, labels, metrics)
    fig.show()
"""

from .simulators.opm.infoiter import InfoIterReader as DataReader
from .core.analyzer import Analyzer
from .core.models import SimulationData
from .visualization import ConvergencePlotter, create_plotter


__version__ = "0.1.0"
__author__ = "Jakob Torben"
__email__ = "jakob.torben@sintef.no"

__all__ = [
    "DataReader",
    "Analyzer",
    "ConvergencePlotter",
    "create_plotter",
]
