"""
Callback functions for the convergence analysis dashboard.

Contains all Dash callback functions organized by functionality.
"""

from typing import Dict, Any, List, Tuple
import dash
from dash import Input, Output, State, callback_context, html
import plotly.graph_objects as go
import numpy as np

from .data_handler import DataHandler
from .callback_utils import (
    compute_iteration_data,
    create_slider_marks,
    get_step_details,
    get_convergence_status,
    create_intensity_plot,
    format_iteration_info,
    determine_new_step,
    get_step_date_info,
)
from ..visualization import create_plotter
from ..visualization.base import apply_theme_layout
import base64
import io
import tempfile
import os
from pathlib import Path


def register_callbacks(app: dash.Dash, data_handler: DataHandler):
    """
    Register all callbacks for the dashboard.

    Args:
        app: Dash application instance
        data_handler: DataHandler instance
    """
    register_upload_callbacks(app, data_handler)
    register_step_navigation_callbacks(app, data_handler)
    register_main_dashboard_callbacks(app, data_handler)
    register_keyboard_navigation(app)


def register_upload_callbacks(app: dash.Dash, data_handler: DataHandler):
    """Register callbacks for file upload functionality."""

    @app.callback(
        [
            Output("header-status", "children"),
            Output("header-status", "style"),
            Output("upload-trigger", "children"),
        ],
        [Input("upload-data", "contents")],
        [State("upload-data", "filename")],
    )
    def handle_file_upload(contents, filenames):
        """Handle file upload and load case data."""
        print(
            f"Upload callback triggered with contents: {contents is not None}, filenames: {filenames}"
        )
        if contents is None:
            # No file uploaded, return default status
            return (
                [html.Span("Ready", style={"fontWeight": "bold", "color": "#27ae60"})],
                {"marginLeft": "20px", "display": "inline-block"},
                "",  # upload-trigger
            )

        try:
            # Handle both single file and multiple files
            if not isinstance(contents, list):
                contents = [contents]
                filenames = [filenames]

            # Create a temporary directory for this upload session
            temp_dir = tempfile.mkdtemp()

            # Save all uploaded files with their original names
            uploaded_files = []
            for content, filename in zip(contents, filenames):
                if content is not None:
                    # Decode the uploaded file
                    content_type, content_string = content.split(",")
                    decoded = base64.b64decode(content_string)

                    # Save the uploaded file with its original name
                    uploaded_file_path = Path(temp_dir) / filename
                    with open(uploaded_file_path, "wb") as f:
                        f.write(decoded)
                    uploaded_files.append(uploaded_file_path)

            # Try to load the case from the uploaded files
            # The system will automatically look for companion files in the same directory
            success = False
            for uploaded_file in uploaded_files:
                success = data_handler.load_case_from_path(str(uploaded_file))
                if success:
                    break  # Stop on first successful load

            # Clean up the temporary directory
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except OSError:
                pass  # Directory might already be deleted

            if success:
                # Get case summary for status display
                summary = data_handler.get_case_summary()
                status_text = f"Case Loaded ({summary['n_steps']} steps)"
                status_color = "#27ae60"  # Green
                trigger_value = "uploaded"  # Trigger plot updates
            else:
                status_text = "Upload Failed - Invalid Files"
                status_color = "#e74c3c"  # Red
                trigger_value = "failed"

            return (
                [
                    html.Span(
                        status_text, style={"fontWeight": "bold", "color": status_color}
                    )
                ],
                {"marginLeft": "20px", "display": "inline-block"},
                trigger_value,  # upload-trigger
            )

        except Exception as e:
            print(f"Error handling file upload: {e}")
            return (
                [
                    html.Span(
                        "Upload Error", style={"fontWeight": "bold", "color": "#e74c3c"}
                    )
                ],
                {"marginLeft": "20px", "display": "inline-block"},
                "error",  # upload-trigger
            )


def register_step_navigation_callbacks(app: dash.Dash, data_handler: DataHandler):
    """Register callbacks for step navigation."""

    @app.callback(
        [
            Output("prev-step-btn", "disabled"),
            Output("next-step-btn", "disabled"),
            Output("current-step-display", "children"),
            Output("step-details-display", "children"),
            Output("convergence-status-display", "children"),
            Output("convergence-status-display", "style"),
            Output("current-step-store", "data"),
            Output("iteration-intensity-plot", "figure"),
            Output("iteration-slider", "min"),
            Output("iteration-slider", "max"),
            Output("iteration-slider", "value"),
            Output("iteration-slider", "marks"),
            Output("iteration-slider", "tooltip"),
            Output("iteration-info", "children"),
        ],
        [
            Input("prev-step-btn", "n_clicks"),
            Input("next-step-btn", "n_clicks"),
            Input("iteration-intensity-plot", "clickData"),
            Input("iteration-slider", "value"),
            Input("upload-trigger", "children"),
            Input("startup-interval", "n_intervals"),
        ],
        [State("current-step-store", "data")],
    )
    def update_step_navigation(
        prev_clicks,
        next_clicks,
        click_data,
        slider_value,
        upload_trigger,
        startup_interval,
        current_step_store,
    ):
        """Handle step navigation through buttons, plot clicks, slider, and upload trigger with enhanced display."""
        ctx = callback_context

        if not data_handler.data:
            return _handle_no_data_case()

        # Get current step from step store
        current_step = current_step_store if current_step_store is not None else 0

        # Compute iteration data once
        iterations_per_step, cumulative_sums, total_iterations = compute_iteration_data(
            data_handler.data
        )

        # Calculate step bounds
        n_steps = len(data_handler.data.get("curve_pos", [])) - 1
        max_step = max(0, n_steps - 1)

        # Check what triggered this callback
        upload_was_triggered = any(
            trigger["prop_id"] == "upload-trigger.children" for trigger in ctx.triggered
        )
        startup_interval_triggered = any(
            trigger["prop_id"] == "startup-interval.n_intervals"
            for trigger in ctx.triggered
        )

        if upload_was_triggered and upload_trigger in ["uploaded", "failed", "error"]:
            # Reset to first step when new data is uploaded
            new_step = 0
        elif startup_interval_triggered and data_handler.data:
            # Handle initial load case - startup interval triggered and data exists
            new_step = 0
        elif not ctx.triggered:
            # Handle case with no triggers but data exists
            new_step = 0
        else:
            # Determine new step based on user interaction
            new_step = determine_new_step(
                ctx.triggered,
                current_step,
                max_step,
                click_data,
                slider_value,
                prev_clicks,
                next_clicks,
            )

        # Create slider configuration
        slider_max = (
            len(iterations_per_step) - 1 if len(iterations_per_step) > 0 else 100
        )
        slider_marks = create_slider_marks(len(iterations_per_step))

        # Get step information
        step_details = get_step_details(data_handler.data, new_step)
        convergence_status, convergence_style = get_convergence_status(
            data_handler.analysis_results, new_step
        )

        # Create iteration intensity plot
        intensity_fig = create_intensity_plot(
            iterations_per_step, cumulative_sums, new_step, total_iterations
        )

        # Format iteration info
        iteration_info = format_iteration_info(
            iterations_per_step, cumulative_sums, new_step, total_iterations
        )

        # Disable tooltip to avoid navigation issues
        tooltip_template = None

        return (
            new_step <= 0,  # prev_disabled
            new_step >= max_step,  # next_disabled
            step_details,  # current_step_display (PRIMARY: Report Step, Time Step, Date - large & bold)
            f"Substep {new_step + 1} of {max_step + 1}",  # step_details_display (SECONDARY: navigation info)
            convergence_status,  # convergence_status_display
            convergence_style,  # convergence_status_style
            new_step,  # current_step_store
            intensity_fig,  # iteration_intensity_plot
            0,  # slider min
            int(slider_max),  # slider max
            new_step,  # slider value
            slider_marks,  # slider marks
            {
                "always_visible": False,
            },  # slider tooltip (disabled)
            iteration_info,  # iteration info
        )


def _handle_no_data_case() -> tuple:
    """Handle the case when no data is loaded."""
    from ..ui.styles import status_badge_style

    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text="No data loaded",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="#6c757d"),
    )
    empty_fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return (
        True,  # prev_disabled
        True,  # next_disabled
        "No data loaded",  # current_step_display
        "",  # step_details_display
        "",  # convergence_status_display
        status_badge_style(),  # convergence_status_style
        0,  # current_step_store
        empty_fig,  # iteration_intensity_plot
        0,  # slider min
        100,  # slider max
        0,  # slider value
        {},  # slider marks
        {
            "always_visible": False,
        },  # slider tooltip (disabled)
        "",  # iteration info
    )


def register_main_dashboard_callbacks(app: dash.Dash, data_handler: DataHandler):
    """Register callbacks for main dashboard plots."""

    @app.callback(
        Output("main-plot", "figure"),
        [
            Input("current-step-store", "data"),
            Input("upload-trigger", "children"),
        ],
    )
    def update_main_plots(step_data, upload_trigger):
        """Update main dashboard plots based on current step and upload trigger."""

        plotter = create_plotter(theme="plotly_white")

        if not data_handler.data or not data_handler.analysis_results:
            return _create_empty_figure(plotter)

        try:
            data = data_handler.data
            errors, labels, metrics = data_handler.analysis_results

            # Get current step from step_data (0-based index)
            current_step = step_data if step_data is not None else 0

            # Create main dashboard with current step
            main_fig = plotter.create_dashboard(
                data, errors, labels, metrics, steps=[current_step]
            )

            return main_fig

        except Exception as e:
            print(f"Error updating plots: {e}")
            import traceback

            traceback.print_exc()
            return _create_error_figure(plotter, str(e))


def _create_empty_figure(plotter) -> go.Figure:
    """Create empty figure when no data is available."""
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text="📁 Upload INFOITER + DBG files to begin convergence analysis",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="#95a5a6"),
    )
    empty_fig = apply_theme_layout(empty_fig, plotter.theme, "", 400)

    return empty_fig


def _create_error_figure(plotter, error_message: str) -> go.Figure:
    """Create error figure when there's an issue with data processing."""
    error_fig = go.Figure()
    error_fig.add_annotation(
        text=f"⚠️ Error: {error_message}",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#e74c3c"),
    )
    error_fig = apply_theme_layout(error_fig, plotter.theme, "Error", 400)

    return error_fig


def register_keyboard_navigation(app: dash.Dash):
    """Register keyboard navigation callbacks."""

    # Client-side callback for keyboard navigation
    app.clientside_callback(
        """
        function(current_step) {
            // Remove any existing keyboard listeners to avoid duplicates
            if (window.keyboardListenerAdded) {
                return '';
            }
            
            // Add keyboard event listener
            document.addEventListener('keydown', function(event) {
                if (event.key === 'ArrowLeft') {
                    event.preventDefault();
                    // Trigger previous button click
                    const prevBtn = document.getElementById('prev-step-btn');
                    if (prevBtn && !prevBtn.disabled) {
                        prevBtn.click();
                    }
                } else if (event.key === 'ArrowRight') {
                    event.preventDefault();
                    // Trigger next button click
                    const nextBtn = document.getElementById('next-step-btn');
                    if (nextBtn && !nextBtn.disabled) {
                        nextBtn.click();
                    }
                }
            });
            
            // Mark that keyboard listener has been added
            window.keyboardListenerAdded = true;
            return '';
        }
        """,
        Output("keyboard-handler", "children"),
        [Input("current-step-store", "data")],
    )
