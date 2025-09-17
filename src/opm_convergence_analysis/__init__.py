"""
Convergence Analysis Library for OPM Flow Simulations

This package provides tools for analyzing convergence behavior in reservoir
simulation runs, particularly for OPM Flow simulations. It includes functions
for reading INFOITER files and analyzing convergence behavior.

Main components:
- DataReader: Parse INFOITER files from OPM Flow simulations
- Analyzer: Calculate error metrics and convergence indicators
- Visualization: Plot convergence analysis results

Example usage:
    import opm_convergence_analysis as oca

    # Load and analyze data
    data = oca.load_infoiter("simulation.INFOITER")
    results = oca.analyze_convergence(data)

    # Visualize results
    oca.plot_convergence_analysis(data, results)
"""

from .core.data_reader import DataReader
from .core.analyzer import Analyzer
from .visualization import ConvergencePlotter, create_plotter


# High-level API functions
def load_infoiter(filename):
    """Load INFOITER file and return parsed data structure."""
    reader = DataReader()
    return reader.read_infoiter(filename)


def analyze_convergence(data, **kwargs):
    """Analyze convergence behavior from INFOITER data."""
    analyzer = Analyzer()

    # If no tolerances provided, try to load from DBG file
    if "tol" not in kwargs and "dbg_file" in data:
        from .core.dbg_reader import DBGReader

        try:
            dbg_reader = DBGReader(data["dbg_file"])

            # Only use tolerances that actually exist in the DBG file
            tol = {}
            if "ToleranceMb" in dbg_reader.parameters:
                tol["mb"] = dbg_reader.parameters["ToleranceMb"]
            if "ToleranceCnv" in dbg_reader.parameters:
                tol["cnv"] = dbg_reader.parameters["ToleranceCnv"]

            if not tol:
                raise ValueError("No tolerance parameters found in DBG file")

            kwargs["tol"] = tol
        except Exception as e:
            raise ValueError(f"Could not load tolerances from DBG file: {e}")
    elif "tol" not in kwargs:
        raise ValueError(
            "No tolerances provided and no DBG file found. Please provide tolerances explicitly or ensure DBG file exists."
        )

    return analyzer.analyze(data, **kwargs)


def plot_convergence_analysis(data, errors, labels, metrics, **kwargs):
    """
    Create interactive convergence analysis dashboard.

    Args:
        data: Data structure from load_infoiter
        errors: Error array from analyze_convergence
        labels: Error metric labels from analyze_convergence
        metrics: Metrics structure from analyze_convergence
        **kwargs: Additional plotting options:
            - theme: Plot theme (default: "plotly_white")
            - steps: Specific steps to plot (default: None for all)
            - save_path: Path to save HTML file (default: None)
            - show: Whether to open in browser (default: False)

    Returns:
        Plotly figure with interactive dashboard
    """
    plotter = create_plotter(theme=kwargs.get("theme", "plotly_white"))
    fig = plotter.create_dashboard(
        data, errors, labels, metrics, steps=kwargs.get("steps", None)
    )

    # Handle output options
    save_path = kwargs.get("save_path")
    show = kwargs.get("show", False)

    if save_path:
        fig.write_html(save_path)
        print(f"Dashboard saved to: {save_path}")

    if show:
        fig.show()

    return fig


def analyze_and_plot(infoiter_file, output_file=None, **kwargs):
    """
    Convenience function to load, analyze, and plot convergence data in one call.

    Args:
        infoiter_file: Path to INFOITER file
        output_file: Path to save HTML dashboard (default: auto-generated name)
        **kwargs: Additional options passed to plot_convergence_analysis

    Returns:
        Tuple of (errors, labels, metrics, figure)
    """
    # Load data
    data = load_infoiter(infoiter_file)

    # Analyze convergence
    errors, labels, metrics = analyze_convergence(data)

    # Generate output filename if not provided
    if output_file is None:
        from pathlib import Path

        input_path = Path(infoiter_file)
        output_file = f"{input_path.stem}_convergence_analysis.html"

    # Create and save plot
    fig = plot_convergence_analysis(
        data, errors, labels, metrics, save_path=output_file, **kwargs
    )

    print(f"✅ Analysis complete! Dashboard saved to '{output_file}'")
    print("Open the HTML file in your browser to view the interactive dashboard.")

    return errors, labels, metrics, fig


__version__ = "0.1.0"
__author__ = "Jakob Torben"
__email__ = "jakob.torben@sintef.no"

__all__ = [
    "DataReader",
    "Analyzer",
    "ConvergencePlotter",
    "load_infoiter",
    "analyze_convergence",
    "plot_convergence_analysis",
    "analyze_and_plot",
    "create_plotter",
]
