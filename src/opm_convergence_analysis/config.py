"""
Configuration module for convergence analysis dashboard.

Contains all constants, styling, and configuration settings used throughout the application.
"""

from typing import Dict, List, Any

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Color schemes
PRIMARY_COLORS = {
    "info": "#3498db",
    "danger": "#e74c3c",
}

# Font configuration
FONT_FAMILY = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'


# ============================================================================
# EXTERNAL STYLESHEETS AND RESOURCES
# ============================================================================

EXTERNAL_STYLESHEETS = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
]

META_TAGS = [
    {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    {
        "name": "description",
        "content": "Interactive OPM Flow convergence analysis dashboard",
    },
]

# ============================================================================
# PLOT CONFIGURATION PRESETS
# ============================================================================

PLOT_CONFIG_EXPORT = lambda filename, height=750, width=1400: {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": filename,
        "height": height,
        "width": width,
        "scale": 2,
    },
}
