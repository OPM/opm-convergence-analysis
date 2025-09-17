"""
Main Dash application for convergence monitoring dashboard.

Contains the application factory and configuration.
"""

import dash
from typing import Optional

from ..config import EXTERNAL_STYLESHEETS, META_TAGS
from ..ui.layout import create_main_layout
from ..ui.styles import get_custom_css
from ..visualization import create_plotter
from .data_handler import DataHandler
from .callbacks import register_callbacks


def create_app(
    theme: str = "plotly_white", debug: bool = False
) -> tuple[dash.Dash, DataHandler]:
    """
    Create and configure the Dash application.

    Args:
        theme: Default theme for the application
        debug: Whether to run in debug mode

    Returns:
        Tuple of (app, data_handler)
    """
    # Initialize Dash app with enhanced styling
    app = dash.Dash(
        __name__,
        title="Convergence Monitoring Dashboard",
        external_stylesheets=EXTERNAL_STYLESHEETS,
        meta_tags=META_TAGS,
    )

    # Create data handler
    data_handler = DataHandler()

    # Add custom CSS for enhanced styling
    app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {get_custom_css()}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """

    # Set layout
    app.layout = create_main_layout()

    # Register callbacks
    register_callbacks(app, data_handler)

    # Initialize plotter (will be recreated with user-selected theme)
    plotter = create_plotter(theme=theme)

    return app, data_handler


def run_app(
    app: dash.Dash, host: str = "127.0.0.1", port: int = 8050, debug: bool = False
):
    """
    Run the Dash application.

    Args:
        app: Dash application instance
        host: Host to bind to
        port: Port to run on
        debug: Debug mode
    """
    print(f"Open your browser to: http://{host}:{port}")
    print("Use the controls to adjust analysis parameters in real-time!")

    app.run(debug=debug, host=host, port=port)
