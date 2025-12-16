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
    """Component for convergence distance radar/polar plot."""

    COLORSCALE = "Inferno_r"

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
        """Add radar plot showing convergence distance per metric."""
        n_iter = len(row_ix)
        labels_closed = list(labels) + [labels[0]]
        colors = px.colors.sample_colorscale(
            self.COLORSCALE,
            np.linspace(0, 1, n_iter),
        )
        discrete_colorscale = self._build_discrete_colorscale(colors, n_iter)

        for i, iter_idx in enumerate(row_ix):
            is_last = i == n_iter - 1
            progress = i / max(n_iter - 1, 1)
            r_values = list(errors[iter_idx, :]) + [errors[iter_idx, 0]]
            rgb = px.colors.unlabel_rgb(colors[i])

            # Visible markers at vertices; colorbar attached to the last trace
            marker_config = dict(
                size=4 + 3 * progress,
                color=[i + 1.5] * len(r_values),  # Numeric value for colorscale mapping
                colorscale=discrete_colorscale,
                cmin=1,
                cmax=n_iter + 1,
                showscale=is_last,
                colorbar=dict(
                    title=dict(text="Iter", side="top", font=dict(size=11)),
                    thickness=12,
                    len=0.35,
                    y=0.78,
                    x=0.92,
                    tickvals=self._colorbar_tickvals(n_iter),
                    ticktext=self._colorbar_ticktext(n_iter),
                    tickmode="array",
                    outlinewidth=1,
                    outlinecolor="#888",
                ) if is_last else None,
            )

            fig.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=labels_closed,
                    fill="toself",
                    name=f"Iter {i+1}",
                    mode="lines+markers",
                    line=dict(
                        color=f"rgba{rgb + (0.95,)}",
                        width=1.2 + 2.6 * progress,
                    ),
                    marker=marker_config,
                    fillcolor=f"rgba{rgb + (0.12 + 0.20 * progress,)}",
                    showlegend=False,
                    hovertemplate=f"<b>Iteration {i+1}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        self._update_polar_axes(fig, row, col, errors, row_ix)

    def _build_discrete_colorscale(self, colors: List[str], n_iter: int) -> List:
        """Build a discrete colorscale with sharp color boundaries."""
        discrete_scale = []
        for i, color in enumerate(colors):
            discrete_scale.append([i / n_iter, color])
            discrete_scale.append([(i + 1) / n_iter, color])
        return discrete_scale

    def _colorbar_tickvals(self, n_iter: int) -> List[float]:
        """Get tick values for colorbar (centered in each color band)."""
        if n_iter <= 8:
            return [i + 0.5 for i in range(1, n_iter + 1)]
        return [1.5, n_iter + 0.5]

    def _colorbar_ticktext(self, n_iter: int) -> List[str]:
        """Get tick labels for colorbar."""
        if n_iter <= 8:
            return [str(i) for i in range(1, n_iter + 1)]
        return ["1", str(n_iter)]

    def _update_polar_axes(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        errors: np.ndarray,
        row_ix: np.ndarray,
    ):
        """Configure polar axes styling and range."""
        range_max = min(max(int(np.ceil(np.max(errors[row_ix, :]))), 2), 6)

        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, range_max],
                tickvals=list(range(range_max + 1)),
                tickmode="array",
                gridcolor="rgba(100,100,100,0.2)",
                linecolor="rgba(0,0,0,0.3)",
                tickfont=dict(
                    size=11,
                    family=self.plotter.font_family,
                    color="#444",
                ),
            ),
            angularaxis=dict(
                tickfont=dict(
                    size=11,
                    family=self.plotter.font_family,
                    color="#333",
                ),
                gridcolor="rgba(100,100,100,0.15)",
                linecolor="rgba(0,0,0,0.2)",
                rotation=90,
                direction="clockwise",
            ),
            bgcolor="rgba(252,252,255,0.3)",
            row=row,
            col=col,
        )
