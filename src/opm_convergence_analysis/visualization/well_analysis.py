"""
Well analysis components for dashboard visualization.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from .components import PlotComponent
from ..core.models import SimulationData


class WellFailureAnalyzer:
    """Analyzes well failure patterns from generic WellFailure objects."""

    @staticmethod
    def analyze_well_failures_from_data(
        data: SimulationData, row_ix: np.ndarray
    ) -> Tuple[Dict[str, int], Dict[str, List[Dict[str, Any]]], int]:
        """
        Analyze well failure patterns from SimulationData.well_failures.

        Returns:
            Tuple of (failure_counts, well_failures_by_well, failed_iteration_count)
        """
        if data.well_failures is None or len(data.well_failures) == 0:
            return {}, {}, 0

        failure_counts = defaultdict(int)
        well_failures = defaultdict(list)
        failed_iterations = 0

        for i, idx in enumerate(row_ix):
            if idx < len(data.well_failures) and data.well_failures[idx]:
                failed_iterations += 1
                iteration_num = i + 1

                for failure in data.well_failures[idx]:
                    reason = failure.get_display_reason()
                    failure_counts[reason] += 1
                    well_failures[failure.well_name].append(
                        {
                            "iteration": iteration_num,
                            "reason": reason,
                        }
                    )

        return dict(failure_counts), dict(well_failures), failed_iterations


class WellStatusPlotComponent(PlotComponent):
    """Binary plot showing which iterations had well failures."""

    def add_to_figure(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        data: SimulationData,
        row_ix: np.ndarray,
        step: int,
    ):
        if data.well_failures is None or len(data.well_failures) == 0:
            self._add_message(fig, row, col, "No well status data available")
            return

        iterations = np.arange(1, len(row_ix) + 1)
        status_values = np.zeros(len(row_ix), dtype=int)
        hover_text = []

        for i, idx in enumerate(row_ix):
            if idx < len(data.well_failures) and data.well_failures[idx]:
                status_values[i] = 1
                well_info = [
                    f"{f.well_name}: {f.failure_type}" for f in data.well_failures[idx]
                ]
                hover_text.append("<br>".join(well_info))
            else:
                hover_text.append("All wells OK")

        colors = ["#27ae60" if v == 0 else "#e74c3c" for v in status_values]

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=status_values,
                mode="markers+lines",
                marker=dict(color=colors, size=8, symbol="circle"),
                line=dict(color="#7f8c8d", width=1),
                hovertemplate="<b>Iteration:</b> %{x}<br><b>Status:</b> %{customdata}<br><extra></extra>",
                customdata=hover_text,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_yaxes(
            tickvals=[0, 1],
            ticktext=["Wells OK", "Wells Failed"],
            title="Well Status",
            range=[-0.2, 1.2],
            row=row,
            col=col,
        )
        fig.update_xaxes(title="Iteration", row=row, col=col)

    def _add_message(self, fig: go.Figure, row: int, col: int, message: str):
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref=f"x{3}" if row == 2 and col == 1 else "x",
            yref=f"y{3}" if row == 2 and col == 1 else "y",
            showarrow=False,
            font=dict(size=16, color="#7f8c8d"),
            row=row,
            col=col,
        )


class WellFailureSummaryComponent(PlotComponent):
    """Text summary of well failures with statistics."""

    def add_to_figure(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        data: SimulationData,
        row_ix: np.ndarray,
        step: int,
    ):
        failure_counts, well_failures, failed_iterations = (
            WellFailureAnalyzer.analyze_well_failures_from_data(data, row_ix)
        )

        if not failure_counts:
            self._add_annotation(
                fig,
                row,
                col,
                f"✅ All wells converged<br>({len(row_ix)} iterations)",
                color="#27ae60",
            )
            self._hide_axes(fig, row, col)
            return

        summary = self._build_summary(
            well_failures, failure_counts, failed_iterations, len(row_ix)
        )
        self._add_annotation(fig, row, col, summary, color="#495057")
        self._hide_axes(fig, row, col)

    def _build_summary(
        self,
        well_failures: Dict,
        failure_counts: Dict,
        failed_iter: int,
        total_iter: int,
    ) -> str:
        lines = [
            "<b>Well Failure Summary</b>",
            "",
            f"Failed Wells: <b>{len(well_failures)}</b> | Failed Iterations: <b>{failed_iter}</b>",
            "─" * 60,
        ]

        # Show top 12 wells sorted by failure count
        sorted_wells = sorted(
            well_failures.items(), key=lambda x: len(x[1]), reverse=True
        )
        max_failures = max(len(f) for f in well_failures.values())

        for idx, (well_name, failures) in enumerate(sorted_wells[:12]):
            if idx > 0:
                lines.append("")

            # Aggregate failure types
            type_counts = defaultdict(int)
            for f in failures:
                type_counts[f["reason"]] += 1

            types = ", ".join(
                f"{t} ({c}x)" if c > 1 else t
                for t, c in sorted(
                    type_counts.items(), key=lambda x: x[1], reverse=True
                )
            )

            iters = [f["iteration"] for f in failures]
            iter_str = (
                f"iter {iters[0]}"
                if len(iters) == 1
                else f"iters {min(iters)}-{max(iters)} ({len(iters)} total)"
            )

            color = self._get_color_emoji(len(failures), max_failures)
            lines.append(f"{color} <b>{well_name}</b>: {types}")
            lines.append(f"      <i>{iter_str}</i>")

        if len(sorted_wells) > 12:
            lines.append("", f"<i>... and {len(sorted_wells) - 12} more wells</i>")

        # Summary stats for multiple wells
        if len(well_failures) > 1:
            lines.extend(["", "─" * 40, "", "<b>📈 Top Issues:</b>"])
            for i, (ftype, count) in enumerate(
                sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:3], 1
            ):
                pct = (count / failed_iter) * 100
                lines.append(f"  {i}. {ftype}: <b>{count}</b> ({pct:.0f}%)")

        return "<br>".join(lines)

    def _get_color_emoji(self, count: int, max_count: int) -> str:
        if count == 0:
            return "🟢"
        severity = count / max(max_count, 1)
        return "🔴" if severity >= 0.75 else "🟠" if severity >= 0.5 else "🟡"

    def _add_annotation(
        self, fig: go.Figure, row: int, col: int, text: str, color: str
    ):
        fig.add_annotation(
            text=text,
            x=0.5,
            y=0.95,
            xref=f"x{4}" if row == 2 and col == 2 else "x",
            yref=f"y{4}" if row == 2 and col == 2 else "y",
            showarrow=False,
            font=dict(size=12, color=color, family="monospace"),
            bgcolor="rgba(248,249,250,0.98)",
            bordercolor="#dee2e6",
            borderwidth=1,
            borderpad=12,
            align="left" if color == "#495057" else "center",
            xanchor="center",
            yanchor="top",
            row=row,
            col=col,
        )

    def _hide_axes(self, fig: go.Figure, row: int, col: int):
        fig.update_xaxes(
            showticklabels=False, showgrid=False, zeroline=False, row=row, col=col
        )
        fig.update_yaxes(
            showticklabels=False, showgrid=False, zeroline=False, row=row, col=col
        )
