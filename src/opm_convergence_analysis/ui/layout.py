"""
Main layout creation for the convergence analysis dashboard.

Contains functions to create the complete dashboard layout.
"""

from dash import dcc, html

from .components import (
    create_header,
    create_step_navigation,
)
from ..config import PLOT_CONFIG_EXPORT


def create_main_layout() -> html.Div:
    """
    Create the main dashboard layout.

    Returns:
        Complete Dash layout component
    """
    return html.Div(
        [
            # Header
            create_header(),
            # Invisible div for keyboard event handling
            html.Div(
                id="keyboard-handler",
                tabIndex=0,
                style={
                    "position": "fixed",
                    "top": "-9999px",
                    "left": "-9999px",
                    "width": "1px",
                    "height": "1px",
                    "opacity": "0",
                },
            ),
            # Store current step state
            dcc.Store(id="current-step-store", data=0),
            # Hidden div to trigger plot updates on file upload
            html.Div(id="upload-trigger", style={"display": "none"}),
            # Main Content Area
            html.Div(
                id="main-content-area",
                children=[create_convergence_analysis_content()],
            ),
            # Keyboard event listener
            dcc.Store(id="keyboard-listener"),
        ],
        style={"padding": "20px", "backgroundColor": "#f8f9fa", "minHeight": "100vh"},
    )


def create_convergence_analysis_content() -> html.Div:
    """
    Create convergence analysis mode content.

    Returns:
        Dash HTML component for convergence analysis
    """
    return html.Div(
        [
            # Step Navigation
            create_step_navigation(),
            # Main Plot
            dcc.Graph(
                id="main-plot",
                style={"height": "800px"},
                config=PLOT_CONFIG_EXPORT("convergence_analysis", 800, 1400),
            ),
        ],
        style={
            "border": "1px solid #dee2e6",
            "padding": "25px",
            "margin": "15px 0",
            "backgroundColor": "#ffffff",
            "borderRadius": "8px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )
