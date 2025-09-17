"""
Centralized styling utilities for the dashboard.

Contains all CSS styles and styling utilities used throughout the application.
"""

from typing import Dict, Any

# Color palette
COLORS = {
    "primary": "#007bff",
    "primary_hover": "#0056b3",
    "secondary": "#6c757d",
    "success": "#28a745",
    "danger": "#dc3545",
    "warning": "#ffc107",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
    "text_primary": "#2c3e50",
    "text_secondary": "#6c757d",
    "text_muted": "#95a5a6",
    "border": "#dee2e6",
    "background": "#ffffff",
    "background_light": "#f8f9fa",
}


# Common style utilities
def card_style(padding: str = "25px") -> Dict[str, Any]:
    """Standard card styling."""
    return {
        "padding": padding,
        "border": f"1px solid {COLORS['border']}",
        "margin": "15px 0",
        "backgroundColor": COLORS["background"],
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
    }


def button_style(
    color: str = "primary", size: str = "medium", disabled: bool = False
) -> Dict[str, Any]:
    """Standard button styling."""
    base_style = {
        "border": "none",
        "borderRadius": "6px",
        "cursor": "pointer" if not disabled else "not-allowed",
        "fontWeight": "600",
        "textAlign": "center",
        "transition": "all 0.2s ease",
        "opacity": "0.6" if disabled else "1.0",
    }

    # Size variants
    if size == "small":
        base_style.update(
            {
                "padding": "8px 12px",
                "fontSize": "12px",
                "minWidth": "80px",
                "height": "32px",
            }
        )
    elif size == "large":
        base_style.update(
            {
                "padding": "16px 24px",
                "fontSize": "16px",
                "minWidth": "140px",
                "height": "52px",
            }
        )
    else:  # medium
        base_style.update(
            {
                "padding": "12px 20px",
                "fontSize": "14px",
                "minWidth": "120px",
                "height": "44px",
            }
        )

    # Color variants
    if color == "primary":
        base_style.update(
            {
                "backgroundColor": COLORS["primary"],
                "color": "white",
                "boxShadow": f"0 2px 4px rgba(0,123,255,0.3)",
            }
        )
    elif color == "secondary":
        base_style.update(
            {
                "backgroundColor": COLORS["secondary"],
                "color": "white",
            }
        )

    return base_style


def status_badge_style(status: str = "default") -> Dict[str, Any]:
    """Status badge styling."""
    base_style = {
        "fontSize": "14px",
        "fontWeight": "600",
        "padding": "8px 16px",
        "borderRadius": "6px",
        "display": "inline-block",
        "border": "2px solid",
        "lineHeight": "1.2",
    }

    if status == "success":
        base_style.update(
            {
                "color": "#155724",
                "backgroundColor": "#d4edda",
                "borderColor": "#c3e6cb",
            }
        )
    elif status == "error":
        base_style.update(
            {
                "color": "#721c24",
                "backgroundColor": "#f8d7da",
                "borderColor": "#f5c6cb",
            }
        )
    else:  # default
        base_style.update(
            {
                "backgroundColor": COLORS["light"],
                "borderColor": COLORS["border"],
                "color": COLORS["text_secondary"],
            }
        )

    return base_style


# Custom CSS for slider component
SLIDER_CSS = """
/* Main slider container */
.iteration-slider {
    margin: 15px 0 20px 0 !important;
    padding: 0 50px !important; /* Match plot margins */
}

.iteration-slider .rc-slider {
    margin: 0 !important;
    height: 10px !important;
    position: relative !important;
}

/* Slider rail (background track) */
.iteration-slider .rc-slider-rail {
    background: linear-gradient(90deg, #e9ecef 0%, #dee2e6 100%) !important;
    height: 10px !important;
    border-radius: 5px !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1) !important;
}

/* Slider track (active portion) */
.iteration-slider .rc-slider-track {
    background: linear-gradient(90deg, #007bff 0%, #0056b3 100%) !important;
    height: 10px !important;
    border-radius: 5px !important;
    box-shadow: 0 1px 3px rgba(0,123,255,0.3) !important;
}

/* Slider handle (draggable element) */
.iteration-slider .rc-slider-handle {
    width: 24px !important;
    height: 24px !important;
    margin-top: -7px !important;
    border: 3px solid #007bff !important;
    background: radial-gradient(circle, #ffffff 0%, #f8f9fa 100%) !important;
    box-shadow: 0 3px 8px rgba(0,123,255,0.4) !important;
    border-radius: 50% !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: grab !important;
}

.iteration-slider .rc-slider-handle:hover {
    border-color: #0056b3 !important;
    box-shadow: 0 5px 15px rgba(0,123,255,0.5) !important;
    transform: scale(1.15) !important;
    cursor: grabbing !important;
}

.iteration-slider .rc-slider-handle:focus {
    border-color: #0056b3 !important;
    box-shadow: 0 0 0 5px rgba(0,123,255,0.25), 0 5px 15px rgba(0,123,255,0.4) !important;
    outline: none !important;
    transform: scale(1.1) !important;
}

.iteration-slider .rc-slider-handle:active {
    box-shadow: 0 2px 6px rgba(0,123,255,0.6) !important;
    transform: scale(1.05) !important;
    cursor: grabbing !important;
}

/* Slider marks (step indicators) */
.iteration-slider .rc-slider-mark {
    top: 15px !important;
}

.iteration-slider .rc-slider-mark-text {
    font-size: 10px !important;
    color: #6c757d !important;
    font-weight: 600 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    text-align: center !important;
    transform: translateX(-50%) !important;
}

.iteration-slider .rc-slider-mark-text-active {
    color: #007bff !important;
    font-weight: 700 !important;
    font-size: 11px !important;
}

/* Slider dots */
.iteration-slider .rc-slider-dot {
    width: 6px !important;
    height: 6px !important;
    margin-top: 2px !important;
    border: 1px solid #6c757d !important;
    background-color: #ffffff !important;
    border-radius: 50% !important;
    transition: all 0.2s ease !important;
}

.iteration-slider .rc-slider-dot-active {
    border-color: #007bff !important;
    background-color: #007bff !important;
    transform: scale(1.3) !important;
}

/* Enhanced tooltip styling */
.rc-slider-tooltip .rc-tooltip-inner {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
    color: white !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

.rc-slider-tooltip .rc-tooltip-arrow {
    border-top-color: #2c3e50 !important;
}

/* Intensity visualization legend */
.intensity-legend {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 15px !important;
    margin-top: 10px !important;
    font-size: 11px !important;
    color: #6c757d !important;
}

.intensity-legend-item {
    display: flex !important;
    align-items: center !important;
    gap: 5px !important;
}

.intensity-legend-color {
    width: 12px !important;
    height: 12px !important;
    border-radius: 2px !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
}
"""


def get_custom_css() -> str:
    """Get all custom CSS as a string."""
    return SLIDER_CSS


# Layout utilities
def flex_container(
    direction: str = "row",
    justify: str = "space-between",
    align: str = "center",
    gap: str = "15px",
) -> Dict[str, Any]:
    """Flexible container styling."""
    return {
        "display": "flex",
        "flexDirection": direction,
        "justifyContent": justify,
        "alignItems": align,
        "gap": gap,
    }


def text_style(
    size: str = "medium", weight: str = "normal", color: str = "text_primary"
) -> Dict[str, Any]:
    """Text styling utility."""
    base_style = {
        "color": COLORS.get(color, color),
        "lineHeight": "1.4",
        "margin": "0",
    }

    # Size variants
    if size == "small":
        base_style["fontSize"] = "12px"
    elif size == "large":
        base_style["fontSize"] = "18px"
    elif size == "xlarge":
        base_style["fontSize"] = "24px"
    else:  # medium
        base_style["fontSize"] = "14px"

    # Weight variants
    if weight == "bold":
        base_style["fontWeight"] = "600"
    elif weight == "light":
        base_style["fontWeight"] = "400"
    else:  # normal
        base_style["fontWeight"] = "500"

    return base_style
