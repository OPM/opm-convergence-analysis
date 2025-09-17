"""
Reusable UI components for the convergence analysis dashboard.

Contains functions to create standard UI components used throughout the application.
"""

from dash import dcc, html
from typing import Dict, Any, List, Optional

from .styles import (
    COLORS,
    card_style,
    button_style,
    status_badge_style,
    flex_container,
    text_style,
)


def create_header() -> html.Div:
    """
    Create the main header component.

    Returns:
        Dash HTML component for the header
    """
    return html.Div(
        [
            html.H1(
                "Convergence Analysis Dashboard",
                style=text_style(size="xlarge", weight="bold", color="text_primary"),
            ),
            html.Div(
                [
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            [
                                html.Button(
                                    "📁 Upload Case Files",
                                    style=button_style(color="secondary", size="small"),
                                ),
                                html.Div(
                                    "Upload INFOITER + DBG files",
                                    style={
                                        **text_style(
                                            size="xsmall", color="text_secondary"
                                        ),
                                        "marginTop": "4px",
                                        "fontStyle": "italic",
                                    },
                                ),
                            ]
                        ),
                        style={"display": "inline-block"},
                        multiple=True,
                        accept=".INFOITER,.DBG,.DATA",
                    ),
                    html.Div(
                        id="header-status",
                        children=[
                            html.Span(
                                "Ready",
                                style=text_style(weight="bold", color="success"),
                            )
                        ],
                        style={"marginLeft": "20px", "display": "inline-block"},
                    ),
                ],
                style=flex_container(justify="flex-start"),
            ),
        ],
        style={
            **flex_container(),
            "backgroundColor": COLORS["background"],
            "padding": "25px 30px",
            "borderBottom": f"2px solid {COLORS['border']}",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )


def create_step_navigation() -> html.Div:
    """
    Create enhanced step navigation controls with report step, time step, and convergence status.

    Returns:
        Dash HTML component for step navigation
    """
    return html.Div(
        [
            # Navigation buttons and main info row
            html.Div(
                [
                    html.Button(
                        "◀ PREVIOUS",
                        id="prev-step-btn",
                        n_clicks=0,
                        style=button_style(color="primary", size="medium"),
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="current-step-display",
                                children="Step 0 of 0",
                                style=text_style(
                                    size="large", weight="bold", color="text_primary"
                                ),
                            ),
                            html.Div(
                                id="step-details-display",
                                children="",
                                style={
                                    **text_style(size="medium", color="text_secondary"),
                                    "marginTop": "8px",
                                },
                            ),
                            html.Div(
                                id="convergence-status-display",
                                children="",
                                style={
                                    **status_badge_style(),
                                    "marginTop": "12px",
                                },
                            ),
                        ],
                        style={
                            **flex_container(direction="column", align="center"),
                            "flex": "1",
                            "margin": "0 30px",
                        },
                    ),
                    html.Button(
                        "NEXT ▶",
                        id="next-step-btn",
                        n_clicks=0,
                        style=button_style(color="primary", size="medium"),
                    ),
                ],
                style={
                    **flex_container(),
                    "marginBottom": "25px",
                    "minHeight": "80px",
                },
            ),
            # Iteration intensity visualization with slider
            _create_iteration_section(),
        ],
        style=card_style(),
    )


def _create_iteration_section() -> html.Div:
    """Create the iteration intensity and slider section with legend."""
    return html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "Nonlinear Iteration Analysis:",
                        style={
                            **text_style(
                                size="medium", weight="bold", color="text_primary"
                            ),
                            "display": "block",
                            "marginBottom": "5px",
                        },
                    ),
                ],
            ),
            dcc.Graph(
                id="iteration-intensity-plot",
                style={"height": "180px", "marginBottom": "10px"},
                config={
                    "displayModeBar": False,
                    "staticPlot": False,
                    "doubleClick": False,
                    "showTips": False,
                    "displaylogo": False,
                    "responsive": True,
                    "autosizable": True,
                    "scrollZoom": False,
                },
            ),
            html.Div(
                [
                    dcc.Slider(
                        id="iteration-slider",
                        min=0,
                        max=100,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode="drag",
                        className="iteration-slider",
                    ),
                ],
                style={"margin": "10px 0"},
            ),
            html.Div(
                id="iteration-info",
                children="",
                style={
                    **text_style(size="small", color="text_secondary"),
                    "textAlign": "center",
                    "marginTop": "15px",
                    "padding": "8px",
                    "backgroundColor": "rgba(248,249,250,0.7)",
                    "borderRadius": "4px",
                    "fontWeight": "500",
                },
            ),
        ],
        style={"width": "100%"},
    )
