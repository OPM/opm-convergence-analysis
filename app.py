#!/usr/bin/env python3
"""
Interactive Convergence Analysis Dashboard Application

A clean, refactored Dash-based web application providing real-time interactive
visualization for OPM Flow convergence analysis.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the package to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.opm_convergence_analysis.dashboard import create_app, DataHandler
from src.opm_convergence_analysis.dashboard.app import run_app
import src.opm_convergence_analysis as oca


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Convergence Analysis Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                                        # Start dashboard (use web upload)
  python app.py --case /path/to/case.INFOITER         # Load specific INFOITER file
  python app.py --case /path/to/case.DBG              # Load DBG file (+ INFOITER if found)
  python app.py --case /path/to/case_directory/       # Load from directory
  python app.py --case /path/to/case.DATA             # Load using DATA file path
  python app.py --port 8080 --debug                   # Custom port and debug mode
        """,
    )

    parser.add_argument(
        "--case",
        "-c",
        type=str,
        help="Path to case file (.INFOITER, .DBG, .DATA) or directory",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the dashboard to (default: 127.0.0.1)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--theme",
        choices=["plotly_white", "plotly_dark", "simple_white", "presentation"],
        default="plotly_white",
        help="Default theme (default: plotly_white)",
    )

    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_arguments()

    print("Starting Convergence Analysis Dashboard...")

    # Create the application
    app, data_handler = create_app(theme=args.theme, debug=args.debug)

    # Load case data if specified
    if args.case:
        print(f"Loading case from: {args.case}")
        if data_handler.load_case_from_path(args.case):
            print("Case loaded successfully")

            if data_handler.loaded_case_params:
                print("Parameters loaded from DBG file:")
                for key, value in data_handler.loaded_case_params.items():
                    print(f"   {key}: {value}")
        else:
            print("Failed to load case. Please check the path and try again.")
            print("You can upload case files using the web interface.")
    else:
        print("No case specified - use the upload interface to load case files")
        print("Or restart with: python app.py --case /path/to/your/case")

    # Run the application
    run_app(app, host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
