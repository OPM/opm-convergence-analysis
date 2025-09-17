"""
Individual plot components for dashboard visualization.

This module contains specialized classes for creating individual plot components
that can be combined into dashboards. Each component handles a specific type of
visualization with consistent styling and behavior.
"""

import numpy as np
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px

from .base import format_hover_template


class PlotComponent:
    """Base class for individual plot components."""

    def __init__(self, plotter):
        """Initialize with reference to parent plotter for styling."""
        self.plotter = plotter

    def add_to_figure(self, fig: go.Figure, row: int, col: int, **kwargs):
        """Add this component to a subplot in the given figure."""
        raise NotImplementedError("Subclasses must implement add_to_figure")


class DistancePlotComponent(PlotComponent):
    """Component for distance and failure count plot with dual y-axis."""

    def add_to_figure(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        metrics: Dict[str, Any],
        row_ix: np.ndarray,
        step: int,
    ):
        """Add distance plot to the figure."""
        iterations = np.arange(1, len(row_ix) + 1)
        fail_data = metrics["fail"][row_ix]
        dist_data = metrics["dist"][row_ix]

        # Failure count bars
        fig.add_trace(
            go.Bar(
                x=iterations,
                y=fail_data,
                name="#unconverged",
                marker=dict(
                    color="#9999FF",
                    line=dict(width=1, color="rgba(255,255,255,0.8)"),
                    opacity=0.8,
                ),
                showlegend=False,
                hovertemplate=format_hover_template(
                    "Unconverged Wells", "Iteration", "Count"
                ),
            ),
            row=row,
            col=col,
            secondary_y=False,
        )

        # Distance line
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=dist_data,
                mode="lines+markers",
                name="Distance",
                line=dict(color="red", width=2, shape="linear"),
                marker=dict(size=6, color="red", symbol="circle"),
                showlegend=False,
                hovertemplate=format_hover_template(
                    "Convergence Distance",
                    "Iteration",
                    "Distance",
                    {"Format": "%{y:.2e}"},
                ),
            ),
            row=row,
            col=col,
            secondary_y=True,
        )

        self._add_subplot_legend(
            fig, ["#unconverged", "Distance"], ["#9999FF", "red"], row, col, "top right"
        )
        self._update_axes(fig, fail_data, dist_data, row, col)

    def _add_subplot_legend(
        self,
        fig: go.Figure,
        items: List[str],
        colors: List[str],
        row: int,
        col: int,
        position: str,
    ):
        """Add custom legend to subplot."""
        legend_text = ""
        for item, color in zip(items, colors):
            legend_text += (
                f'<span style="color:{color}; font-size:14px;">■</span> {item}<br>'
            )
        legend_text = legend_text.rstrip("<br>")

        positions = {
            "top right": (0.98, 0.98, "right", "top"),
        }
        x, y, xanchor, yanchor = positions.get(position, positions["top right"])

        axis_num = (row - 1) * 2 + col
        xref = f"x{axis_num} domain" if axis_num > 1 else "x domain"
        yref = f"y{axis_num} domain" if axis_num > 1 else "y domain"

        fig.add_annotation(
            x=x,
            y=y,
            xref=xref,
            yref=yref,
            text=legend_text,
            showarrow=False,
            font=dict(size=12, family=self.plotter.font_family),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            borderpad=4,
            align="left",
            xanchor=xanchor,
            yanchor=yanchor,
        )

    def _update_axes(
        self,
        fig: go.Figure,
        fail_data: np.ndarray,
        dist_data: np.ndarray,
        row: int,
        col: int,
    ):
        """Update axes for distance plot."""
        fig.update_xaxes(
            title_text="Iteration",
            row=row,
            col=col,
            title_font={"family": self.plotter.font_family, "size": 14},
        )

        fig.update_yaxes(
            title_text="# Unconverged",
            title_font=dict(
                color=self.plotter.primary_colors["info"],
                family=self.plotter.font_family,
                size=12,
            ),
            range=[0, max(7, max(fail_data) * 1.1) if len(fail_data) > 0 else 7],
            gridcolor="rgba(0,0,0,0.1)",
            row=row,
            col=col,
            secondary_y=False,
        )

        fig.update_yaxes(
            title_text="Distance",
            title_font=dict(
                color=self.plotter.primary_colors["danger"],
                family=self.plotter.font_family,
                size=12,
            ),
            type="log" if max(dist_data) > 1000 else "linear",
            range=[
                0,
                (
                    max(dist_data) * 1.1
                    if len(dist_data) > 0 and max(dist_data) > 0
                    else 1
                ),
            ],
            gridcolor="rgba(0,0,0,0.1)",
            row=row,
            col=col,
            secondary_y=True,
        )


class RadarPlotComponent(PlotComponent):
    """Component for error metrics radar plot."""

    def add_to_figure(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        errors: np.ndarray,
        labels: List[str],
        row_ix: np.ndarray,
        step: int,
    ):
        """Add radar plot to the figure."""
        n_iterations = len(row_ix)
        max_error = max(np.max(errors[row_ix, :]), 6)

        # Enhanced color scheme
        if n_iterations <= 8:
            colors = px.colors.qualitative.Set2[:n_iterations]
        else:
            colors = px.colors.sample_colorscale("viridis", n_iterations)

        colors = [
            f"rgba{px.colors.hex_to_rgb(c) + (0.8,)}" if "#" in c else c for c in colors
        ]

        for i, iter_idx in enumerate(row_ix):
            opacity = 0.7 if i == n_iterations - 1 else 0.4
            line_width = 3 if i == n_iterations - 1 else 2

            fig.add_trace(
                go.Scatterpolar(
                    r=errors[iter_idx, :],
                    theta=labels,
                    fill="toself",
                    name=f"Iter {i+1}" + (" (Final)" if i == n_iterations - 1 else ""),
                    line=dict(color=colors[i], width=line_width),
                    fillcolor=colors[i],
                    opacity=opacity,
                    showlegend=bool(i < 6 or i == n_iterations - 1),
                    hovertemplate=format_hover_template(
                        f"Iteration {i+1}", "%{theta}", "%{r:.2e}"
                    ),
                ),
                row=row,
                col=col,
            )

        self._update_polar_axes(fig, row, col)

    def _update_polar_axes(self, fig: go.Figure, row: int, col: int):
        """Update polar axes styling."""
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 6],
                tickmode="linear",
                tick0=0,
                dtick=1,
                gridcolor="rgba(0,0,0,0.15)",
                linecolor="rgba(0,0,0,0.3)",
                tickfont=dict(size=12, family=self.plotter.font_family),
            ),
            angularaxis=dict(
                tickfont=dict(
                    size=12, family=self.plotter.font_family, color="#495057"
                ),
                gridcolor="rgba(0,0,0,0.15)",
                linecolor="rgba(0,0,0,0.3)",
                rotation=90,
                direction="clockwise",
            ),
            bgcolor="rgba(248,249,250,0.3)",
            row=row,
            col=col,
        )
