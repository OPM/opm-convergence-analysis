"""
Main plotter class for convergence analysis visualization.

This module provides the unified ConvergencePlotter interface for all convergence
analysis visualizations, including dashboards, radar plots, and well analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import apply_theme_layout, create_color_sequence
from ..config import PRIMARY_COLORS, FONT_FAMILY
from .components import DistancePlotComponent, RadarPlotComponent
from .well_analysis import WellStatusPlotComponent, WellFailureSummaryComponent
from ..core.models import SimulationData


class ConvergencePlotter:
    """
    Main convergence plotter for creating interactive convergence analysis visualizations.

    This class provides comprehensive functionality for convergence analysis
    visualizations, including dashboards, radar plots, and well analysis.
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the ConvergencePlotter.

        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
        self.primary_colors = PRIMARY_COLORS
        self.font_family = FONT_FAMILY

        # Initialize plot components
        self.distance_component = DistancePlotComponent(self)
        self.radar_component = RadarPlotComponent(self)
        self.well_status_component = WellStatusPlotComponent(self)
        self.well_summary_component = WellFailureSummaryComponent(self)

        self._current_step = 0

    def create_dashboard(
        self,
        data: SimulationData,
        errors: np.ndarray,
        labels: List[str],
        metrics: Dict[str, Any],
        steps: Optional[List[int]] = None,
    ) -> go.Figure:
        """
        Create an interactive convergence analysis dashboard.

        Args:
            data: Data structure from DataReader
            errors: Error array from Analyzer
            labels: Error metric labels
            metrics: Metrics structure from Analyzer
            steps: List of steps to visualize (default: all steps)

        Returns:
            Interactive Plotly figure with convergence analysis dashboard
        """
        return self._create_convergence_analysis_dashboard(
            data, errors, labels, metrics, steps
        )

    def create_convergence_analysis_dashboard(
        self,
        data: Dict[str, Any],
        errors: np.ndarray,
        labels: List[str],
        metrics: Dict[str, Any],
        steps: Optional[List[int]] = None,
    ) -> go.Figure:
        """
        Create a convergence analysis dashboard with distance metrics and radar plot.

        Args:
            data: Data structure from DataReader
            errors: Error array from Analyzer
            labels: Error metric labels
            metrics: Metrics structure from Analyzer
            steps: List of steps to visualize (default: all steps)

        Returns:
            Interactive Plotly figure with convergence analysis
        """
        return self._create_convergence_analysis_dashboard(
            data, errors, labels, metrics, steps
        )

    def _create_convergence_analysis_dashboard(
        self,
        data: SimulationData,
        errors: np.ndarray,
        labels: List[str],
        metrics: Dict[str, Any],
        steps: Optional[List[int]] = None,
    ) -> go.Figure:
        """
        Create the convergence analysis dashboard with all components.
        """
        # Validate and prepare steps
        steps = self._validate_and_prepare_steps(steps, data, metrics)

        # Create subplots for convergence analysis
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Number of unconverged metrics and distance from convergence",
                "Distance from Convergence per metric",
                "Well Status & Failures",
                "Well Failure Details",
            ],
            specs=[
                [{"secondary_y": True}, {"type": "polar"}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
            column_widths=[0.4, 0.6],  # Give more width to polar plot
            row_heights=[0.55, 0.45],  # Give more height to top row
            horizontal_spacing=0.08,
            vertical_spacing=0.20,
        )

        # Add components for the first step
        current_step = steps[0]
        self._add_convergence_analysis_components(
            fig, data, errors, labels, metrics, current_step
        )

        fig.update_layout(
            template=self.theme,
            height=850,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50),
        )

        # Shift polar plot title up to avoid overlap
        for ann in fig.layout.annotations:
            if "Distance from Convergence per metric" in ann.text:
                ann.y += 0.03

        return fig

    def _validate_and_prepare_steps(
        self,
        steps: Optional[List[int]],
        data: SimulationData,
        metrics: Dict[str, Any],
    ) -> List[int]:
        """Validate and prepare steps for visualization."""
        if isinstance(data, SimulationData):
            n_steps = data.n_steps
        else:
            curve_pos = data.get("curve_pos", [])
            n_steps = len(curve_pos) - 1 if len(curve_pos) > 1 else 1

        if steps is None:
            # Default to first step
            return [0]

        # Validate steps are within bounds
        valid_steps = [step for step in steps if 0 <= step < n_steps]

        # Return first valid step or default to 0
        return valid_steps[:1] if valid_steps else [0]

    def _add_convergence_analysis_components(
        self,
        fig: go.Figure,
        data: SimulationData,
        errors: np.ndarray,
        labels: List[str],
        metrics: Dict[str, Any],
        step: int,
    ):
        """Add components for the convergence analysis dashboard."""

        if isinstance(data, SimulationData):
            # Get indices for this step
            # Assuming data.iterations has a 'step_index' column
            # We need raw integer indices corresponding to the 'errors' array rows
            row_ix = np.where(data.iterations["step_index"] == step)[0]
        else:
            curve_pos = data.get("curve_pos", [])
            if step >= len(curve_pos) - 1:
                step = 0  # Fallback to first step
            row_ix = np.arange(curve_pos[step], curve_pos[step + 1])

        # Add components to their respective subplots
        self.distance_component.add_to_figure(fig, 1, 1, metrics, row_ix, step)
        self.radar_component.add_to_figure(fig, 1, 2, errors, labels, row_ix, step)
        self.well_status_component.add_to_figure(fig, 2, 1, data, row_ix, step)
        self.well_summary_component.add_to_figure(fig, 2, 2, data, row_ix, step)


def create_radar_plot(
    errors: np.ndarray,
    labels: List[str],
    step_indices: Optional[List[int]] = None,
    theme: str = "plotly_white",
) -> go.Figure:
    """
    Create a standalone radar plot for error metrics.

    Args:
        errors: Error array (m x k)
        labels: Error metric labels
        step_indices: Specific iteration indices to plot (default: all)
        theme: Plotly theme

    Returns:
        Plotly figure with radar plot
    """
    if step_indices is None:
        step_indices = list(range(len(errors)))

    max_error = max(np.max(errors[step_indices, :]), 6)

    fig = go.Figure()

    # Create color sequence for iterations
    colors = create_color_sequence(len(step_indices), "viridis")

    for i, idx in enumerate(step_indices):
        fig.add_trace(
            go.Scatterpolar(
                r=errors[idx, :],
                theta=labels,
                fill="toself",
                name=f"Iteration {i+1}",
                line_color=colors[i],
                fillcolor=colors[i],
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, max_error], tickmode="linear", tick0=0, dtick=2
            ),
            angularaxis=dict(tickfont_size=10),
        ),
        title="Error Metrics Radar Plot",
        template=theme,
        showlegend=True,
    )

    return fig


# Convenience functions for easy access
def create_plotter(theme: str = "plotly_white") -> ConvergencePlotter:
    """
    Create a ConvergencePlotter instance.

    Args:
        theme: Plotly theme to use

    Returns:
        ConvergencePlotter instance
    """
    return ConvergencePlotter(theme=theme)
