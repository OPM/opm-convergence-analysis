"""
Well analysis components for dashboard visualization.

This module contains specialized classes for analyzing and visualizing well-related
convergence issues, including well status tracking and failure analysis.
"""

import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go

from .components import PlotComponent


class WellFailureAnalyzer:
    """Analyzes well failure patterns and provides structured summaries."""

    @staticmethod
    def parse_failure_reason(failure_string: str) -> str:
        """
        Parse well failure reason from failure string.

        Handles all failure types from the WellFailure enum:
        - Invalid, MassBalance, Pressure, ControlBHP, ControlTHP,
        - ControlRate, Unsolvable, WrongFlowDirection

        Args:
            failure_string: Raw failure string like "ATO002 ControlRate" or "ATO005 MassBalance Phase=2"

        Returns:
            Human-readable failure reason string
        """
        failure_upper = failure_string.upper()

        # Mass Balance failures (with phase information)
        if "MASSBALANCE" in failure_upper:
            if "PHASE=0" in failure_upper:
                return "Mass Balance (Water)"
            elif "PHASE=1" in failure_upper:
                return "Mass Balance (Oil)"
            elif "PHASE=2" in failure_upper:
                return "Mass Balance (Gas)"
            else:
                return "Mass Balance"

        # Control failures
        elif "CONTROLRATE" in failure_upper:
            return "Control Rate"
        elif "CONTROLBHP" in failure_upper:
            return "Control BHP"
        elif "CONTROLTHP" in failure_upper:
            return "Control THP"

        # Pressure failures
        elif "PRESSURE" in failure_upper:
            return "Pressure"

        # Solver failures
        elif "UNSOLVABLE" in failure_upper:
            return "Unsolvable"

        # Flow direction failures
        elif "WRONGFLOWDIRECTION" in failure_upper:
            return "Wrong Flow Direction"

        # Invalid or unknown failures
        elif "INVALID" in failure_upper:
            return "Invalid"

        # Fallback for unrecognized failure types
        else:
            # Try to extract a meaningful name from the string
            parts = failure_string.strip().split()
            if len(parts) > 1:
                # Return the failure type part (after well name)
                failure_type = " ".join(parts[1:])
                return f"Other ({failure_type})"
            else:
                return "Unknown"

    @staticmethod
    def analyze_well_failures(
        well_status_strings: List[str], failed_wells_bool: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, int], Dict[str, List[Dict[str, Any]]], int]:
        """
        Analyze well failure patterns from status strings.

        Args:
            well_status_strings: List of well status strings for each iteration
            failed_wells_bool: Optional boolean array indicating failed iterations

        Returns:
            Tuple of (failure_counts, well_failures, failed_iterations)
        """
        failure_counts = {}
        well_failures = {}
        failed_iterations = 0

        for i, status in enumerate(well_status_strings):
            # Convert to string and check for failures
            status_str = str(status) if status is not None else ""
            is_failed_by_bool = (
                failed_wells_bool[i] if failed_wells_bool is not None else False
            )

            # Use both string check and boolean check
            has_fail_in_string = "FAIL" in status_str
            if has_fail_in_string or is_failed_by_bool:
                failed_iterations += 1
                iteration_num = i + 1

                if has_fail_in_string:
                    # Extract failure reasons and well names from strings like "FAIL { ATO002 ControlRate }"
                    failures = re.findall(r"\{([^}]+)\}", status_str)

                    if failures:
                        for failure in failures:
                            # Extract well name (first part before space)
                            parts = failure.strip().split()
                            well_name = parts[0] if parts else "Unknown"

                            # Parse failure reason
                            reason = WellFailureAnalyzer.parse_failure_reason(failure)

                            # Count failure types
                            failure_counts[reason] = failure_counts.get(reason, 0) + 1

                            # Track well failures
                            if well_name not in well_failures:
                                well_failures[well_name] = []
                            well_failures[well_name].append(
                                {
                                    "iteration": iteration_num,
                                    "reason": reason,
                                    "full_reason": failure.strip(),
                                }
                            )
                    else:
                        # FAIL found but no braces - try to parse what we can
                        reason = WellFailureAnalyzer.parse_failure_reason(status_str)
                        failure_counts[reason] = failure_counts.get(reason, 0) + 1
                        if "Unknown" not in well_failures:
                            well_failures["Unknown"] = []
                        well_failures["Unknown"].append(
                            {
                                "iteration": iteration_num,
                                "reason": reason,
                                "full_reason": status_str,
                            }
                        )
                else:
                    # Failed by boolean but no FAIL in string - try to parse what we can
                    reason = WellFailureAnalyzer.parse_failure_reason(status_str)
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
                    if "Unknown" not in well_failures:
                        well_failures["Unknown"] = []
                    well_failures["Unknown"].append(
                        {
                            "iteration": iteration_num,
                            "reason": reason,
                            "full_reason": f"Failed (status: {status_str})",
                        }
                    )

        return failure_counts, well_failures, failed_iterations


class WellStatusPlotComponent(PlotComponent):
    """Component for well status plot showing failed wells over iterations."""

    def add_to_figure(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        data: Dict[str, Any],
        row_ix: np.ndarray,
        step: int,
    ):
        """Add well status plot to the figure."""
        # Check if we have well status data
        if "raw" not in data or "FailedWells" not in data["raw"]:
            self._add_no_data_message(fig, row, col, "No well status data available")
            return

        failed_wells = data["raw"]["FailedWells"][row_ix]
        iterations = np.arange(1, len(row_ix) + 1)

        # Get detailed well status strings if available
        detailed_status = self._get_detailed_status(data, row_ix, failed_wells)

        # Create binary well status plot
        colors = ["#27ae60", "#e74c3c"]  # Green for success, red for failure
        status_values = failed_wells.astype(int)

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=status_values,
                mode="markers+lines",
                marker=dict(
                    color=[colors[val] for val in status_values],
                    size=8,
                    symbol="circle",
                ),
                line=dict(color="#7f8c8d", width=1),
                name="Well Status",
                hovertemplate="<b>Iteration:</b> %{x}<br>"
                "<b>Status:</b> %{customdata}<br>"
                "<extra></extra>",
                customdata=detailed_status,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Update y-axis to show categories
        fig.update_yaxes(
            tickvals=[0, 1],
            ticktext=["Wells OK", "Wells Failed"],
            title="Well Status",
            range=[-0.2, 1.2],
            row=row,
            col=col,
        )

        fig.update_xaxes(title="Iteration", row=row, col=col)

    def _get_detailed_status(
        self, data: Dict[str, Any], row_ix: np.ndarray, failed_wells: np.ndarray
    ) -> List[str]:
        """Get detailed well status strings for hover information."""
        detailed_status = []

        if "WellStatus" in data["raw"]:
            well_status_strings = data["raw"]["WellStatus"][row_ix]
            for i, status in enumerate(well_status_strings):
                if isinstance(status, str) and "FAIL" in status:
                    # Extract well names and reasons
                    failures = re.findall(r"\{([^}]+)\}", status)
                    well_info = []
                    for failure in failures:
                        parts = failure.strip().split()
                        well_name = parts[0] if parts else "Unknown"
                        if "ControlRate" in failure:
                            reason = "ControlRate"
                        elif "MassBalance" in failure:
                            reason = "MassBalance"
                        else:
                            reason = "Other"
                        well_info.append(f"{well_name}: {reason}")
                    detailed_status.append("<br>".join(well_info))
                else:
                    detailed_status.append("All wells OK")
        else:
            detailed_status = ["Failed" if val else "OK" for val in failed_wells]

        return detailed_status

    def _add_no_data_message(self, fig: go.Figure, row: int, col: int, message: str):
        """Add message when no data is available."""
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
    """Component for well failure summary showing failure reasons, counts, and specific wells."""

    def add_to_figure(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        data: Dict[str, Any],
        row_ix: np.ndarray,
        step: int,
    ):
        """Add well failure summary to the figure."""
        # Check if we have well status data
        if "raw" not in data or "WellStatus" not in data["raw"]:
            self._add_no_data_message(
                fig, row, col, "No detailed well status data available"
            )
            return

        # Get well status strings for this step
        well_status_strings = data["raw"]["WellStatus"][row_ix]
        failed_wells_bool = (
            data["raw"]["FailedWells"][row_ix] if "FailedWells" in data["raw"] else None
        )

        # Analyze failure patterns
        failure_counts, well_failures, failed_iterations = (
            WellFailureAnalyzer.analyze_well_failures(
                well_status_strings, failed_wells_bool
            )
        )

        total_iterations = len(well_status_strings)

        if not failure_counts:
            self._add_success_message(fig, row, col, total_iterations)
            return

        # Create structured failure summary
        summary_text = self._create_failure_summary(
            well_failures, failure_counts, failed_iterations, total_iterations
        )

        fig.add_annotation(
            text=summary_text,
            x=0.5,
            y=0.95,
            xref=f"x{4}" if row == 2 and col == 2 else "x",
            yref=f"y{4}" if row == 2 and col == 2 else "y",
            showarrow=False,
            font=dict(size=12, color="#495057", family="monospace"),
            bgcolor="rgba(248,249,250,0.98)",
            bordercolor="#dee2e6",
            borderwidth=1,
            borderpad=12,
            align="left",
            xanchor="center",
            yanchor="top",
            row=row,
            col=col,
        )

        # Hide axes since we're showing a text summary
        self._hide_axes(fig, row, col)

    def _create_failure_summary(
        self,
        well_failures: Dict[str, List[Dict[str, Any]]],
        failure_counts: Dict[str, int],
        failed_iterations: int,
        total_iterations: int,
    ) -> str:
        """Create structured failure summary text."""
        num_failed_wells = len(well_failures)

        table_lines = []
        table_lines.append("<b>Well Failure Summary</b>")
        table_lines.append("")  # Add blank line for spacing
        table_lines.append(
            f"Failed Wells: <b>{num_failed_wells}</b> | Failed Iterations: <b>{failed_iterations}</b>"
        )
        table_lines.append("─" * 60)

        if num_failed_wells > 0:
            self._add_well_details(table_lines, well_failures)
            self._add_summary_statistics(
                table_lines, well_failures, failure_counts, failed_iterations
            )

        return "<br>".join(table_lines)

    def _add_well_details(
        self, table_lines: List[str], well_failures: Dict[str, List[Dict[str, Any]]]
    ):
        """Add detailed well failure information."""
        max_wells_to_show = 12

        # Sort wells by total number of failures (most problematic first)
        sorted_wells = sorted(
            well_failures.items(), key=lambda x: len(x[1]), reverse=True
        )

        well_entries = []
        for idx, (well_name, failures) in enumerate(sorted_wells):
            if idx >= max_wells_to_show:
                remaining_wells = len(sorted_wells) - max_wells_to_show
                well_entries.append(f"<i>... and {remaining_wells} more wells</i>")
                break

            # Count failures by type for this well
            failure_type_counts = {}
            for failure in failures:
                failure_type = failure["reason"]
                failure_type_counts[failure_type] = (
                    failure_type_counts.get(failure_type, 0) + 1
                )

            # Get all iterations for this well
            iterations = [f["iteration"] for f in failures]

            # Format failure types with frequencies
            type_parts = []
            for failure_type, count in sorted(
                failure_type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                if count == 1:
                    type_parts.append(failure_type)
                else:
                    type_parts.append(f"{failure_type} ({count}x)")

            types_text = ", ".join(type_parts)

            # Format iterations compactly
            if len(iterations) == 1:
                iter_text = f"iter {iterations[0]}"
            elif len(iterations) <= 3:
                iter_text = f"iters {', '.join(map(str, sorted(iterations)))}"
            else:
                iter_text = f"iters {min(iterations)}-{max(iterations)} ({len(iterations)} total)"

            # Get color coding for severity
            max_failures = max(len(failures) for failures in well_failures.values())
            color_emoji = self._get_well_color(len(failures), max_failures)

            well_entries.append(f"{color_emoji} <b>{well_name}</b>: {types_text}")
            well_entries.append(f"    └─ {iter_text}")

        # Add well entries with proper spacing
        for i, entry in enumerate(well_entries):
            if entry.startswith("    └─"):
                table_lines.append(f"      <i>{entry[6:]}</i>")
            else:
                if i > 0 and not entry.startswith("<i>"):
                    table_lines.append("")
                table_lines.append(entry)

    def _add_summary_statistics(
        self,
        table_lines: List[str],
        well_failures: Dict[str, List[Dict[str, Any]]],
        failure_counts: Dict[str, int],
        failed_iterations: int,
    ):
        """Add summary statistics to the failure report."""
        if len(well_failures) > 1:
            table_lines.append("")
            table_lines.append("─" * 40)
            table_lines.append("")  # Add extra spacing

            # Show top failure types
            top_types = sorted(
                failure_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            table_lines.append("<b>📈 Top Issues:</b>")
            for i, (ftype, count) in enumerate(top_types, 1):
                percentage = (count / failed_iterations) * 100
                table_lines.append(
                    f"  {i}. {ftype}: <b>{count}</b> ({percentage:.0f}%)"
                )

            # Add most problematic well
            most_problematic = max(well_failures.items(), key=lambda x: len(x[1]))
            max_failures = max(len(failures) for failures in well_failures.values())
            most_problematic_color = self._get_well_color(
                len(most_problematic[1]), max_failures
            )
            table_lines.append("")
            table_lines.append("")  # Add extra spacing
            table_lines.append(
                f"<b>🔴 Most Problematic:</b> {most_problematic_color} <b>{most_problematic[0]}</b> ({len(most_problematic[1])} failures)"
            )

    def _get_well_color(self, failure_count: int, max_failures: int) -> str:
        """Get color emoji based on failure severity."""
        if failure_count == 0:
            return "🟢"

        if max_failures == 1:
            return "🟡"

        severity = failure_count / max_failures

        if severity >= 0.75:
            return "🔴"
        elif severity >= 0.5:
            return "🟠"
        else:
            return "🟡"

    def _add_success_message(
        self, fig: go.Figure, row: int, col: int, total_iterations: int
    ):
        """Add success message when no failures occurred."""
        fig.add_annotation(
            text=f"✅ All wells converged<br>({total_iterations} iterations)",
            x=0.5,
            y=0.95,
            xref=f"x{4}" if row == 2 and col == 2 else "x",
            yref=f"y{4}" if row == 2 and col == 2 else "y",
            showarrow=False,
            font=dict(size=16, color="#27ae60"),
            bgcolor="rgba(248,249,250,0.98)",
            bordercolor="#dee2e6",
            borderwidth=1,
            borderpad=12,
            align="center",
            xanchor="center",
            yanchor="top",
            row=row,
            col=col,
        )
        self._hide_axes(fig, row, col)

    def _add_no_data_message(self, fig: go.Figure, row: int, col: int, message: str):
        """Add message when no data is available."""
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.95,
            xref=f"x{4}" if row == 2 and col == 2 else "x",
            yref=f"y{4}" if row == 2 and col == 2 else "y",
            showarrow=False,
            font=dict(size=16, color="#7f8c8d"),
            align="center",
            xanchor="center",
            yanchor="top",
            row=row,
            col=col,
        )
        self._hide_axes(fig, row, col)

    def _hide_axes(self, fig: go.Figure, row: int, col: int):
        """Hide axes for text-only displays."""
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=row,
            col=col,
        )
