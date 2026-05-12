"""Identification and validation of the calibration matrix **C**.

The forward model is:

.. math::

    \\boldsymbol{\\varepsilon} = \\mathbf{C}\\,\\mathbf{f}

where

* :math:`\\boldsymbol{\\varepsilon} \\in \\mathbb{R}^m` — measured strains
  (``m`` gauges / bridges),
* :math:`\\mathbf{f} \\in \\mathbb{R}^n` — applied load components
  (forces + moments),
* :math:`\\mathbf{C} \\in \\mathbb{R}^{m \\times n}` — calibration matrix
  (sensitivity, units strain/load).

For *K* load cases the model extends to
:math:`\\mathbf{E} = \\mathbf{C}\\,\\mathbf{F}_{\\mathrm{mat}}` where
:math:`\\mathbf{E} \\in \\mathbb{R}^{m \\times K}` and
:math:`\\mathbf{F}_{\\mathrm{mat}} \\in \\mathbb{R}^{n \\times K}`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def identify(F_mat: NDArray, E_mat: NDArray) -> NDArray:
    """Identify the calibration matrix **C** from known load cases.

    Solves :math:`\\mathbf{E} = \\mathbf{C}\\,\\mathbf{F}_{\\mathrm{mat}}`
    in the least-squares sense, i.e. minimises
    :math:`\\|\\mathbf{C}\\,\\mathbf{F}_{\\mathrm{mat}} - \\mathbf{E}\\|_F`.

    The system is transposed internally so that :func:`numpy.linalg.lstsq`
    solves for each row of **C** independently via the normal equations of
    :math:`\\mathbf{F}_{\\mathrm{mat}}^\\top \\mathbf{C}^\\top =
    \\mathbf{E}^\\top`.

    Parameters
    ----------
    F_mat : NDArray, shape (n, K)
        Applied loads for each of the *K* calibration load cases.
        Each column is one load vector :math:`\\mathbf{f}^{(k)}`.
    E_mat : NDArray, shape (m, K)
        Measured strains for each load case.  Each column is
        :math:`\\boldsymbol{\\varepsilon}^{(k)}`.

    Returns
    -------
    NDArray, shape (m, n)
        Identified calibration matrix **C**.

    Raises
    ------
    ValueError
        If the number of load cases *K* does not match between *F_mat* and
        *E_mat*, or if *K < n* (under-determined system).

    Notes
    -----
    The identification is purely linear (no regularisation).  If the load
    cases are poorly distributed or nearly collinear, the condition number
    of the returned **C** will be large.  Use :func:`loadgauge.linalg.condition_report`
    to diagnose this.
    """
    n, K = F_mat.shape
    m, K2 = E_mat.shape
    if K != K2:
        raise ValueError(f"K mismatch: F_mat has {K} cases, E_mat has {K2}")
    if K < n:
        raise ValueError(
            f"Under-determined: need at least {n} load cases to identify "
            f"an (m × {n}) matrix, got {K}"
        )
    # Solve F_mat.T @ C.T = E_mat.T  for C.T, then transpose
    C_T, *_ = np.linalg.lstsq(F_mat.T, E_mat.T, rcond=None)
    return C_T.T


def validate(C: NDArray, F_mat: NDArray, E_mat: NDArray) -> dict:
    """Evaluate the calibration matrix against known load cases.

    Computes per-channel and global error metrics between the *predicted*
    strain :math:`\\hat{\\mathbf{E}} = \\mathbf{C}\\,\\mathbf{F}_{\\mathrm{mat}}`
    and the *measured* strain *E_mat*.

    Parameters
    ----------
    C : NDArray, shape (m, n)
        Calibration matrix to evaluate.
    F_mat : NDArray, shape (n, K)
        Applied loads (same convention as :func:`identify`).
    E_mat : NDArray, shape (m, K)
        Measured strains.

    Returns
    -------
    dict
        E_pred : NDArray, shape (m, K)
            Predicted strains :math:`\\mathbf{C}\\,\\mathbf{F}_{\\mathrm{mat}}`.
        residuals : NDArray, shape (m, K)
            :math:`\\mathbf{E}_{\\mathrm{pred}} - \\mathbf{E}_{\\mathrm{mat}}`.
        rmse_per_channel : NDArray, shape (m,)
            Root-mean-square error for each strain channel.
        r2_per_channel : NDArray, shape (m,)
            Coefficient of determination :math:`R^2` for each channel.
            A value of 1 means perfect prediction.
        rmse_global : float
            RMSE computed over all channels and load cases.
        r2_global : float
            :math:`R^2` computed on the flattened residuals.
    """
    E_pred = C @ F_mat
    residuals = E_pred - E_mat

    # per-channel RMSE and R²
    rmse_per_channel = np.sqrt(np.mean(residuals**2, axis=1))
    ss_res = np.sum(residuals**2, axis=1)
    ss_tot = np.sum((E_mat - E_mat.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2_per_channel = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 1.0)

    # global
    rmse_global = float(np.sqrt(np.mean(residuals**2)))
    ss_res_g = float(np.sum(residuals**2))
    ss_tot_g = float(np.sum((E_mat - E_mat.mean()) ** 2))
    r2_global = 1.0 - ss_res_g / ss_tot_g if ss_tot_g > 0.0 else 1.0

    return {
        "E_pred": E_pred,
        "residuals": residuals,
        "rmse_per_channel": rmse_per_channel,
        "r2_per_channel": r2_per_channel,
        "rmse_global": rmse_global,
        "r2_global": r2_global,
    }
