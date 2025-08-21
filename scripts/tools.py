"""
tools.py — metric helpers

Provided metrics
----------------
- mape(actual, pred): Mean Absolute Percentage Error (in %), ε-stabilized
- print_output(y_true, y_pred, verbose=True):
    Computes MAE, MSE, Spearman's ρ for 1D arrays; prints if verbose.
    Returns (mae, mse, src).
- print_output_seq(Y_true, Y_pred, verbose=True):
    Per-day metrics for 2D arrays shaped [N, T].
    Prints per-day lists and averages; returns:
      (mae_list, mse_list, src_list, AMAE, ASE, ASRC)

Notes
-----
- All functions are tolerant to NaN/Inf in inputs (ignored via masking).
- Spearman can return NaN when inputs are constant; we coerce that to 0.0.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mape(actual: Iterable[float], pred: Iterable[float], eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error in percent.
    Ignores NaN/Inf and guards against division by zero with `eps`.

    Parameters
    ----------
    actual, pred : iterables of floats
    eps : float
        Small constant to stabilize division.

    Returns
    -------
    float
        MAPE in percent.
    """
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(pred,   dtype=np.float64)

    mask = ~(np.isnan(a) | np.isnan(p) | np.isinf(a) | np.isinf(p))
    if mask.sum() == 0:
        return 0.0

    a = a[mask]
    p = p[mask]
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs((a - p) / denom)) * 100.0)


def _safe_spearman(a: np.ndarray, p: np.ndarray) -> float:
    """
    Spearman's rho that returns 0.0 instead of NaN when arrays are constant
    or completely masked.
    """
    if a.size == 0 or p.size == 0:
        return 0.0
    rho = stats.spearmanr(a, p)[0]
    if np.isnan(rho):
        return 0.0
    return float(rho)


def print_output(labels: Iterable[float], preds: Iterable[float], verbose: bool = True) -> Tuple[float, float, float]:
    """
    Compute MAE, MSE, Spearman's rho for 1D vectors.

    Parameters
    ----------
    labels, preds : iterables of floats
    verbose : bool
        If True, print the three metrics.

    Returns
    -------
    (mae, mse, src) : tuple of floats
    """
    y = np.asarray(labels, dtype=np.float64)
    yhat = np.asarray(preds,  dtype=np.float64)

    # Mask invalid values
    mask = ~(np.isnan(y) | np.isnan(yhat) | np.isinf(y) | np.isinf(yhat))
    if mask.sum() == 0:
        mae = mse = 0.0
        src = 0.0
    else:
        y = y[mask]
        yhat = yhat[mask]
        mae = float(mean_absolute_error(y, yhat))
        mse = float(mean_squared_error(y, yhat))
        src = _safe_spearman(y, yhat)

    if verbose:
        print(mae)
        print(mse)
        print(src)
    return mae, mse, src


def print_output_seq(labels, preds, verbose: bool = True):
    """
    Per-day metrics for 2D sequences.

    Parameters
    ----------
    labels : array-like, shape [N, T]
    preds  : array-like, shape [N, T]
    verbose : bool
        If True, print per-day lists and averages.

    Returns
    -------
    (mae_list, mse_list, src_list, AMAE, ASE, ASRC)
      mae_list, mse_list, src_list : list[float] each of length T
      AMAE : float  (mean of MAE over T days)
      ASE  : float  (mean of MSE over T days)
      ASRC : float  (mean of Spearman over T days)
    """
    P = np.asarray(preds,  dtype=np.float64)
    L = np.asarray(labels, dtype=np.float64)

    if L.ndim != 2 or P.ndim != 2:
        raise ValueError(f"labels and preds must be 2D, got shapes {L.shape} and {P.shape}")

    if L.shape != P.shape:
        raise ValueError(f"labels and preds must have the same shape, got {L.shape} vs {P.shape}")

    T = L.shape[1]
    mae_list: List[float] = []
    mse_list: List[float] = []
    src_list: List[float] = []

    for d in range(T):
        y = L[:, d]
        yhat = P[:, d]

        # Mask invalid values
        mask = ~(np.isnan(y) | np.isnan(yhat) | np.isinf(y) | np.isinf(yhat))
        if mask.sum() == 0:
            mae = mse = 0.0
            src = 0.0
        else:
            y = y[mask]
            yhat = yhat[mask]
            mae = float(mean_absolute_error(y, yhat))
            mse = float(mean_squared_error(y, yhat))
            src = _safe_spearman(y, yhat)

        mae_list.append(mae)
        mse_list.append(mse)
        src_list.append(src)

    AMAE = float(np.mean(mae_list))
    ASE  = float(np.mean(mse_list))
    ASRC = float(np.mean(src_list))

    if verbose:
        print(mae_list, AMAE)
        print(mse_list, ASE)
        print(src_list, ASRC)

    return mae_list, mse_list, src_list, AMAE, ASE, ASRC
