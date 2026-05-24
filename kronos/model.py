"""Core Kronos model module.

Provides the main Kronos time-series prediction model, which implements
a cyclical/seasonal decomposition approach for financial market forecasting.
"""

import numpy as np
from typing import Optional, Union


class Kronos:
    """Kronos time-series forecasting model.

    Fits a multi-component harmonic model to historical price data and
    extrapolates future values.  The model decomposes the signal into:
      - A linear trend component
      - A configurable number of sinusoidal (Fourier) harmonics

    Parameters
    ----------
    n_harmonics : int, optional
        Number of sinusoidal components to fit.  Higher values capture
        more complex seasonal patterns but risk over-fitting.  Default: 5.
    period : int, optional
        Dominant cycle length in trading days.  Default: 252 (one trading year).
    """

    def __init__(self, n_harmonics: int = 5, period: int = 252):
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1")
        if period < 2:
            raise ValueError("period must be >= 2")

        self.n_harmonics = n_harmonics
        self.period = period

        # Fitted parameters (set after calling fit)
        self._coefficients: Optional[np.ndarray] = None
        self._n_obs: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_design_matrix(self, t: np.ndarray) -> np.ndarray:
        """Build the Fourier design matrix for time indices *t*.

        Columns: [1, t, sin(2π·1·t/T), cos(2π·1·t/T), ...,
                         sin(2π·k·t/T), cos(2π·k·t/T)]
        """
        cols = [np.ones(len(t)), t]
        for k in range(1, self.n_harmonics + 1):
            angle = 2 * np.pi * k * t / self.period
            cols.append(np.sin(angle))
            cols.append(np.cos(angle))
        return np.column_stack(cols)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, y: Union[list, np.ndarray]) -> "Kronos":
        """Fit the model to observed values *y*.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Historical time-series values (e.g. closing prices), ordered
            chronologically.

        Returns
        -------
        self : Kronos
            The fitted model instance (allows method chaining).
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) < 4:
            raise ValueError("y must be a 1-D array with at least 4 observations")

        self._n_obs = len(y)
        t = np.arange(self._n_obs, dtype=float)
        X = self._build_design_matrix(t)

        # Ordinary least-squares via numpy
        self._coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Generate *steps* future predictions beyond the training window.

        Parameters
        ----------
        steps : int
            Number of future time steps to forecast.

        Returns
        -------
        predictions : np.ndarray of shape (steps,)
            Forecasted values.
        """
        if self._coefficients is None:
            raise RuntimeError("Model has not been fitted yet.  Call fit() first.")
        if steps < 1:
            raise ValueError("steps must be >= 1")

        t_future = np.arange(self._n_obs, self._n_obs + steps, dtype=float)
        X_future = self._build_design_matrix(t_future)
        return X_future @ self._coefficients

    def fit_predict(self, y: Union[list, np.ndarray], steps: int) -> np.ndarray:
        """Convenience method: fit the model then return *steps* predictions."""
        return self.fit(y).predict(steps)

    def __repr__(self) -> str:  # pragma: no cover
        fitted = "fitted" if self._coefficients is not None else "unfitted"
        return (
            f"Kronos(n_harmonics={self.n_harmonics}, "
            f"period={self.period}, status={fitted})"
        )
