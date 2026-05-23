"""
Kronos - Time Series Forecasting Library
========================================

A fork of shiyu-coder/Kronos providing tools for stock market
prediction and time series analysis.

Modules:
    model     - Core Kronos forecasting model
    utils     - Utility functions for data preparation
    holidays  - Holiday calendar support for major markets
"""

__version__ = "0.1.0"
__author__ = "Kronos Contributors"
__license__ = "MIT"

from kronos.model import KronosModel
from kronos.utils import (
    prepare_time_series,
    normalize_series,
    denormalize_series,
)

__all__ = [
    "KronosModel",
    "prepare_time_series",
    "normalize_series",
    "denormalize_series",
]
