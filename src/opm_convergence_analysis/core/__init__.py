"""
Core functionality for convergence analysis.

This module contains the main classes that handle the core operations:
- Analyzer: Analyzing convergence behavior
- SimulationData: Generic data model
- BaseReader: Abstract base class for readers
"""

from .analyzer import Analyzer
from .models import SimulationData
from .base_reader import BaseReader

__all__ = ["Analyzer", "SimulationData", "BaseReader"]
