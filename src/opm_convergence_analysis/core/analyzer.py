"""
Analyzer class for calculating error metrics from simulation data.

This module provides the Analyzer class which analyzes convergence behavior
using the generic SimulationData model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from .models import SimulationData


class Analyzer:
    """
    Analyzer for convergence behavior.

    This class calculates error metrics and convergence indicators
    from generic SimulationData.
    """

    def __init__(self):
        """Initialize the Analyzer."""
        pass

    def analyze(
        self, model: SimulationData, **kwargs
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Analyze convergence using the generic SimulationData model.

        Args:
            model: SimulationData object
            **kwargs: Optional overrides (e.g., 'tol' dict)

        Returns:
            Tuple of (errors, labels, metrics)
            - errors: numpy array (n_iterations x n_metrics)
            - labels: list of metric names
            - metrics: dict containing analysis results and the source model
        """
        # 1. Identify metrics to analyze
        # We look for columns in iterations dataframe that have metadata
        df = model.iterations
        meta = model.metric_meta

        # Handle tolerance overrides
        # kwargs['tol'] might be {'Convergence': 1e-3, 'Material Balance': 1e-6}
        user_tols = kwargs.get("tol", {})

        # Filter metrics that we can analyze (have numeric data)
        metric_columns = []
        labels = []
        tolerances = []

        # Sort columns to ensure deterministic order (e.g., by group then name)
        # Or preserve order from metadata insertion
        candidate_cols = [c for c in df.columns if c in meta]

        # Define group priority
        group_order = {"Convergence": 0, "Material Balance": 1}

        def sort_key(col_name):
            group = meta[col_name].get("group", "Other")
            # Sort by group priority, but preserve original order within groups (stable sort)
            return group_order.get(group, 99)

        candidate_cols.sort(key=sort_key)

        for col in candidate_cols:
            col_meta = meta[col]
            group = col_meta.get("group")

            # Determine tolerance
            tol = 1.0  # Default

            # 1. User override by group
            if group and group in user_tols:
                tol = user_tols[group]
            # 2. Metadata tolerance
            elif "tolerance" in col_meta:
                tol = col_meta["tolerance"]

            metric_columns.append(col)
            labels.append(col_meta.get("display_name", col))
            tolerances.append(tol)

        if not metric_columns:
            # Return empty structures
            return (
                np.array([]),
                [],
                {
                    "fail": np.zeros(len(df), dtype=bool),
                    "dist": np.zeros(len(df)),
                    "conv": (
                        model.steps["converged"].values
                        if not model.steps.empty
                        else np.array([])
                    ),
                    "model": model,
                },
            )

        # 2. Calculate Errors
        # Extract values as numpy array
        values = df[metric_columns].values
        tol_array = np.array(tolerances)

        # Error = log10(value / tolerance)
        # Handle division by zero and log of zero/negative
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = values / tol_array[np.newaxis, :]
            log_ratios = np.log10(ratios)
            errors = np.maximum(log_ratios, 0.0)

        # Fill NaNs (e.g. from 0/0) with 0
        errors = np.nan_to_num(errors, nan=0.0)

        # 3. Calculate Derived Metrics
        # Failure: Count of metrics > tolerance (error > 0)
        # Returns count of failed metrics
        fail = np.sum(errors > 0, axis=1)

        # Distance: Sum of errors (L1 norm of log-errors)
        dist = np.sum(errors, axis=1)

        # Convergence Status per step
        # We can just grab it from the steps dataframe
        # But the metrics['conv'] expected by dashboard is an array aligned with steps
        conv = model.steps["converged"].values

        metrics = {
            "fail": fail,
            "dist": dist,
            "conv": conv,
            "model": model,
        }

        return errors, labels, metrics
