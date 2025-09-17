"""
Utility functions for dashboard callbacks.

Contains helper functions to simplify and optimize callback logic.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional

from ..ui.styles import status_badge_style


def compute_iteration_data(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute iteration statistics from curve position data.

    Args:
        data: Raw data dictionary containing curve_pos

    Returns:
        Tuple of (iterations_per_step, cumulative_sums, total_iterations)
    """
    curve_pos = data.get("curve_pos", [])
    if len(curve_pos) <= 1:
        return np.array([]), np.array([]), 0

    curve_pos_array = np.array(curve_pos)
    iterations_per_step = np.diff(curve_pos_array)
    cumulative_sums = np.cumsum(iterations_per_step)
    total_iterations = int(np.sum(iterations_per_step))

    return iterations_per_step, cumulative_sums, total_iterations


def create_slider_marks(num_steps: int, max_marks: int = 10) -> Dict[int, str]:
    """

    Args:
        num_steps: Total number of steps
        max_marks: Maximum number of marks to show

    Returns:
        Dictionary mapping step indices to mark labels
    """
    if num_steps <= 0:
        return {}

    marks = {}

    if num_steps <= max_marks:
        # Show every step if we have few steps
        for i in range(num_steps):
            marks[i] = f"{i+1}"
    else:
        # Smart marking strategy for many steps
        step_interval = max(1, num_steps // max_marks)

        # Always mark first step
        marks[0] = "1"

        # Mark intermediate steps
        for i in range(step_interval, num_steps - step_interval, step_interval):
            marks[i] = f"{i+1}"

        # Always mark last step
        marks[num_steps - 1] = f"{num_steps}"

        # Add quartile marks for better navigation
        if num_steps > 20:
            quarter = num_steps // 4
            half = num_steps // 2
            three_quarter = 3 * num_steps // 4

            if quarter not in marks:
                marks[quarter] = f"{quarter+1}"
            if half not in marks:
                marks[half] = f"{half+1}"
            if three_quarter not in marks:
                marks[three_quarter] = f"{three_quarter+1}"

    return marks


def get_step_details(data: Dict[str, Any], step_idx: int) -> str:
    """
    Get formatted step details (report step and time step).

    Args:
        data: Raw data dictionary
        step_idx: Step index (0-based)

    Returns:
        Formatted step details string
    """
    if not data or "raw" not in data or "curve_pos" not in data:
        return ""

    curve_pos = data["curve_pos"]
    if step_idx >= len(curve_pos) - 1:
        return ""

    step_start_idx = curve_pos[step_idx]
    raw_data = data["raw"]

    if "ReportStep" in raw_data and "TimeStep" in raw_data:
        # Check bounds to avoid IndexError
        if step_start_idx >= len(raw_data["ReportStep"]) or step_start_idx >= len(
            raw_data["TimeStep"]
        ):
            return ""

        report_step = raw_data["ReportStep"][step_start_idx]
        time_step = raw_data["TimeStep"][step_start_idx]
        return f"Report Step: {report_step}, Time Step: {time_step}"

    return ""


def get_convergence_status(
    analysis_results: Optional[Tuple], step_idx: int
) -> Tuple[str, Dict[str, Any]]:
    """
    Get convergence status and styling for a step.

    Args:
        analysis_results: Analysis results tuple (errors, labels, metrics)
        step_idx: Step index (0-based)

    Returns:
        Tuple of (status_text, status_style)
    """
    if not analysis_results:
        return "", status_badge_style()

    _, _, metrics = analysis_results
    if "conv" not in metrics or step_idx >= len(metrics["conv"]):
        return "", status_badge_style()

    is_converged = metrics["conv"][step_idx]
    if is_converged:
        return "✅ CONVERGED", status_badge_style("success")
    else:
        return "❌ NOT CONVERGED", status_badge_style("error")


def create_intensity_plot(
    iterations_per_step: np.ndarray,
    cumulative_sums: np.ndarray,
    current_step: int,
    total_iterations: int,
) -> go.Figure:
    """
    Create the iteration intensity visualization plot.

    Args:
        iterations_per_step: Iterations per step array
        cumulative_sums: Cumulative iteration sums
        current_step: Current step index (0-based)
        total_iterations: Total number of iterations

    Returns:
        Plotly figure for iteration intensity
    """
    fig = go.Figure()

    if len(iterations_per_step) == 0:
        fig.add_annotation(
            text="No iteration data available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="#6c757d"),
        )
        _apply_empty_layout(fig)
        return fig

    steps = np.arange(len(iterations_per_step))
    iterations = iterations_per_step.astype(int)

    # Create more pronounced intensity visualization
    max_iterations = np.max(iterations) if len(iterations) > 0 else 1
    min_iterations = np.min(iterations) if len(iterations) > 0 else 0

    # Normalize iterations for better visual contrast
    if max_iterations > min_iterations:
        normalized_iterations = (iterations - min_iterations) / (
            max_iterations - min_iterations
        )
    else:
        normalized_iterations = np.ones(len(iterations))

    # Create more distinct colors for computational intensity
    colors = []
    edge_colors = []

    for norm_iter in normalized_iterations:
        # Color coding for computational intensity
        if norm_iter > 0.8:  # Very high intensity
            colors.append("rgba(220, 53, 69, 0.8)")  # Strong red
            edge_colors.append("rgba(220, 53, 69, 1.0)")
        elif norm_iter > 0.6:  # High intensity
            colors.append("rgba(253, 126, 20, 0.8)")  # Strong orange
            edge_colors.append("rgba(253, 126, 20, 1.0)")
        elif norm_iter > 0.4:  # Medium intensity
            colors.append("rgba(255, 193, 7, 0.8)")  # Yellow
            edge_colors.append("rgba(255, 193, 7, 1.0)")
        else:  # Low intensity
            colors.append("rgba(40, 167, 69, 0.8)")  # Green
            edge_colors.append("rgba(40, 167, 69, 1.0)")

    # Add bars showing actual iteration counts
    fig.add_trace(
        go.Bar(
            x=steps,
            y=iterations,  # Use actual iteration counts
            marker=dict(color=colors, line=dict(color=edge_colors, width=1)),
            name="Nonlinear Iterations",
            hovertemplate=(
                "<b>Step %{customdata[0]}</b><br>"
                "Nonlinear Iterations: %{y}<br>"
                "Relative Intensity: %{customdata[1]:.1%}<extra></extra>"
            ),
            customdata=np.column_stack([(steps + 1), normalized_iterations]).tolist(),
            showlegend=False,
            width=0.8,
        )
    )

    # Add cumulative line
    progress_percentages = (
        (cumulative_sums / total_iterations * 100)
        if total_iterations > 0
        else np.zeros_like(cumulative_sums)
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=cumulative_sums,
            mode="lines+markers",
            line=dict(color="#007bff", width=3, shape="spline"),
            marker=dict(size=5, color="#007bff", line=dict(color="white", width=1)),
            name="Cumulative Iterations",
            yaxis="y2",
            hovertemplate=(
                "<b>Step %{customdata[0]}</b><br>"
                "Cumulative Iterations: %{y}<br>"
                "Total Progress: %{customdata[1]:.1f}%<extra></extra>"
            ),
            customdata=np.column_stack([(steps + 1), progress_percentages]).tolist(),
            showlegend=False,
        )
    )

    # Highlight current step
    if 0 <= current_step < len(steps):
        # Add vertical line
        fig.add_vline(
            x=current_step,
            line=dict(color="#2c3e50", width=3, dash="solid"),
            opacity=0.8,
        )

        # Add annotation with step info
        current_iterations = int(iterations[current_step])
        max_y = np.max(iterations) if len(iterations) > 0 else 1

        fig.add_annotation(
            x=current_step,
            y=max_y * 1.1,  # Position above the highest bar
            text=f"<b>Step {current_step + 1}</b><br>{current_iterations} iter",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#2c3e50",
            ax=0,
            ay=-30,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#2c3e50",
            borderwidth=1,
            font=dict(size=10, color="#2c3e50"),
        )

    _apply_intensity_layout(fig, total_iterations, len(steps))
    return fig


def _apply_empty_layout(fig: go.Figure) -> None:
    """Apply layout for empty plot."""
    fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )


def _apply_intensity_layout(
    fig: go.Figure, total_iterations: int, num_steps: int
) -> None:
    """Apply layout for intensity plot with proper alignment."""
    # Get the maximum iteration count
    max_iterations = 0
    for trace in fig.data:
        if hasattr(trace, "y") and trace.y is not None:
            if trace.yaxis != "y2":  # Only consider bars, not cumulative line
                max_iterations = max(
                    max_iterations, max(trace.y) if len(trace.y) > 0 else 0
                )

    fig.update_layout(
        height=180,  # Slightly taller for better visibility
        margin=dict(l=60, r=60, t=40, b=30),  # More space for labels
        plot_bgcolor="rgba(248,249,250,0.3)",  # Subtle background
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,  # No legend needed
        xaxis=dict(
            title="Step Number",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            gridwidth=1,
            title_font=dict(size=11, color="#2c3e50"),
            tickfont=dict(size=9, color="#495057"),
            tickmode="linear",
            tick0=0,
            dtick=max(1, num_steps // 15) if num_steps > 0 else 1,  # Smart tick spacing
            range=(
                [-0.5, num_steps - 0.5] if num_steps > 0 else [0, 10]
            ),  # Proper range for alignment
            fixedrange=False,  # Allow zooming
        ),
        yaxis=dict(
            title="Nonlinear Iterations",
            side="left",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.1)",
            title_font=dict(size=11, color="#2c3e50"),
            tickfont=dict(size=9, color="#495057"),
            range=(
                [0, max_iterations * 1.2] if max_iterations > 0 else [0, 10]
            ),  # Extra space for annotations
            showticklabels=True,
            tickformat="d",  # Integer format
        ),
        yaxis2=dict(
            title="Cumulative Iterations",
            side="right",
            overlaying="y",
            showgrid=False,
            title_font=dict(size=11, color="#007bff"),
            tickfont=dict(size=9, color="#007bff"),
            range=[0, total_iterations * 1.1] if total_iterations > 0 else [0, 100],
            showticklabels=True,
            tickformat="d",  # Integer format
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.1)", font_size=11
        ),
    )


def format_iteration_info(
    iterations_per_step: np.ndarray,
    cumulative_sums: np.ndarray,
    current_step: int,
    total_iterations: int,
) -> str:
    """
    Format iteration information text

    Args:
        iterations_per_step: Iterations per step array
        cumulative_sums: Cumulative iteration sums
        current_step: Current step index (0-based)
        total_iterations: Total number of iterations

    Returns:
        Formatted iteration information string
    """
    if len(iterations_per_step) == 0 or current_step >= len(iterations_per_step):
        return "No iteration data available"

    current_iterations = int(iterations_per_step[current_step])
    current_cumulative = int(cumulative_sums[current_step])

    # Calculate progress percentage
    progress_pct = (
        (current_cumulative / total_iterations * 100) if total_iterations > 0 else 0
    )

    # Calculate intensity relative to average
    avg_iterations = np.mean(iterations_per_step) if len(iterations_per_step) > 0 else 0
    intensity_ratio = current_iterations / avg_iterations if avg_iterations > 0 else 0

    # Determine intensity description
    if intensity_ratio > 2.0:
        intensity_desc = "Very High"
    elif intensity_ratio > 1.5:
        intensity_desc = "High"
    elif intensity_ratio > 0.75:
        intensity_desc = "Medium"
    else:
        intensity_desc = "Low"

    return f"Step {current_step + 1}: {current_iterations} nonlinear iterations | Cumulative: {current_cumulative}/{total_iterations}"


def determine_new_step(
    ctx_triggered: List[Dict[str, Any]],
    current_step: int,
    max_step: int,
    click_data: Optional[Dict],
    slider_value: Optional[int],
    prev_clicks: int,
    next_clicks: int,
) -> int:
    """
    Determine the new step based on user interaction.

    Args:
        ctx_triggered: Callback context triggered list
        current_step: Current step index
        max_step: Maximum step index
        click_data: Plot click data
        slider_value: Slider value
        prev_clicks: Previous button clicks
        next_clicks: Next button clicks

    Returns:
        New step index (0-based)
    """
    if not ctx_triggered:
        return 0

    trigger_id = ctx_triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "prev-step-btn" and prev_clicks:
        return max(0, current_step - 1)
    elif trigger_id == "next-step-btn" and next_clicks:
        return min(max_step, current_step + 1)
    elif trigger_id == "iteration-intensity-plot" and click_data:
        clicked_x = click_data["points"][0]["x"]
        return min(max(0, int(clicked_x)), max_step)
    elif trigger_id == "iteration-slider" and slider_value is not None:
        return min(max(0, int(slider_value)), max_step)

    return current_step
