"""
Base plotting utilities and shared functionality.

Contains common plotting functions and base classes used by all visualization modules.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..config import (
    PRIMARY_COLORS,
    FONT_FAMILY,
)


def apply_theme_layout(
    fig, theme: str = "plotly_white", title: str = None, height: int = 600
):
    """
    Apply consistent theming and layout to a figure.

    Args:
        fig: Plotly figure to modify
        theme: Plotly theme to use
        title: Optional title for the figure
        height: Figure height in pixels

    Returns:
        Modified figure with applied theme
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available, skipping theme application")
        return fig
    layout_updates = {
        "template": theme,
        "height": height,
        "font": {"family": FONT_FAMILY, "size": 12},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "margin": dict(l=70, r=70, t=120, b=70),
    }

    if title:
        layout_updates["title"] = {
            "text": title,
            "x": 0.5,
            "font": {"size": 18, "family": FONT_FAMILY, "color": "#2c3e50"},
        }

    fig.update_layout(**layout_updates)
    return fig


def create_color_sequence(n_colors: int, colorscale: str = "viridis") -> List[str]:
    """
    Create a sequence of colors for plotting.

    Args:
        n_colors: Number of colors to generate
        colorscale: Plotly colorscale to use

    Returns:
        List of color strings
    """
    if not PLOTLY_AVAILABLE:
        return ["#1f77b4"] * n_colors

    return px.colors.sample_colorscale(colorscale, n_colors)


def format_hover_template(
    data_name: str,
    x_label: str = "X",
    y_label: str = "Y",
    extra_fields: Dict[str, str] = None,
) -> str:
    """
    Create a formatted hover template for consistent styling.

    Args:
        data_name: Name of the data series
        x_label: Label for x-axis data
        y_label: Label for y-axis data
        extra_fields: Additional fields to include in hover

    Returns:
        Formatted hover template string
    """
    template = f"<b>{data_name}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}"

    if extra_fields:
        for key, value in extra_fields.items():
            template += f"<br>{key}: {value}"

    template += "<extra></extra>"
    return template
