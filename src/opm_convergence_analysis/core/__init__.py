"""
Core functionality for convergence analysis.

This module contains the main classes that handle the core operations:
- DataReader: Reading and parsing INFOITER files
- Analyzer: Analyzing convergence behavior
"""

from .data_reader import DataReader
from .analyzer import Analyzer

__all__ = ["DataReader", "Analyzer"]
