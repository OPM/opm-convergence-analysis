"""
Dashboard module for convergence monitoring.

Contains the main application, callbacks, and data handling functionality.
"""

from .app import create_app
from .callbacks import register_callbacks
from .data_handler import DataHandler

__all__ = ["create_app", "register_callbacks", "DataHandler"]
