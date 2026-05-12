"""Apply a calibration (or system) matrix to strain measurements to recover loads.

Three strategies are provided, all sharing the same input/output convention:

* :func:`static` — direct pseudo-inverse, suitable when **C** is
  well-conditioned (quasi-static WFT-like transducer).
* :func:`tikhonov` — Tikhonov-regularised inversion of a general system matrix
  **H** (quasi-static or dynamic); :math:`\\lambda` auto-selected via GCV when
  not supplied.
* :func:`tsvd` — truncated-SVD inversion of **H** (explicit rank truncation).

Convention
----------
The forward model is :math:`\\boldsymbol{\\varepsilon} = \\mathbf{H}\\,\\mathbf{f}`.
All functions invert this relation:

* ``eps`` shape ``(m,)`` → snapshot reconstruction, returns ``f`` shape ``(n,)``.
* ``eps`` shape ``(m, T)`` → time-series reconstruction, returns ``f`` shape ``(n, T)``.
  (Only :func:`static` and :func:`tikhonov` support the 2-D input.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from loadgauge import linalg


def static(C: NDArray, eps: NDArray) -> NDArray:
    """Reconstruct loads from strains using the pseudo-inverse of **C**.

    Computes :math:`\\mathbf{f} = \\mathbf{C}^+\\,\\boldsymbol{\\varepsilon}`,
    where :math:`\\mathbf{C}^+` is the Moore-Penrose pseudo-inverse.

    Parameters
    ----------
    C : NDArray, shape (m, n)
        Calibration matrix (identified offline via :func:`loadgauge.calibration.identify`).
    eps : NDArray, shape (m,) or (m, T)
        Measured strains.  A 2-D input is treated as *T* independent snapshots
        stacked column-wise; each column is one time instant.

    Returns
    -------
    NDArray, shape (n,) or (n, T)
        Reconstructed load components.  Shape matches *eps* (scalar → scalar,
        time series → time series).

    Notes
    -----
    The pseudo-inverse is computed once and reused for all *T* snapshots, so
    this function is efficient for long time series.
    """
    C_plus = linalg.pseudoinverse(C)
    return C_plus @ eps  # broadcast handles both (m,) and (m, T) naturally


def tikhonov(
    H: NDArray,
    eps: NDArray,
    lam: float | None = None,
    L: NDArray | None = None,
    lam_grid: NDArray | None = None,
) -> NDArray:
    """Reconstruct loads via Tikhonov-regularised inversion.

    Solves :math:`\\min_{\\mathbf{f}}\\|\\mathbf{H}\\mathbf{f} -
    \\boldsymbol{\\varepsilon}\\|_2^2 + \\lambda^2\\|\\mathbf{L}\\mathbf{f}\\|_2^2`.

    When *lam* is ``None``, the regularisation parameter is chosen
    automatically by minimising the GCV functional over *lam_grid*.

    Parameters
    ----------
    H : NDArray, shape (m, n)
        System / sensitivity matrix.  For the quasi-static case this is the
        same as the calibration matrix **C**.
    eps : NDArray, shape (m,) or (m, T)
        Measured strains.  For a 2-D input the *same* :math:`\\lambda` is
        selected from the first column and then applied to every column.
    lam : float or None
        Regularisation parameter.  If ``None``, selected via GCV.
    L : NDArray, shape (p, n) or None
        Regularisation operator.  Defaults to the identity (standard
        Tikhonov, zero-order).
    lam_grid : NDArray or None
        Candidate :math:`\\lambda` values for the automatic search.  Defaults
        to ``np.logspace(-8, 2, 300)``.

    Returns
    -------
    NDArray, shape (n,) or (n, T)
        Reconstructed loads.

    Notes
    -----
    For the 2-D case the selected :math:`\\lambda` is determined from the
    first column of *eps* (column 0).  If your time series has a
    non-stationary noise level, call this function separately for each
    segment.
    """
    if lam_grid is None:
        lam_grid = np.logspace(-8, 2, 300)

    eps_2d = eps.ndim == 2

    # auto-select lambda using the first column (or the vector itself)
    if lam is None:
        eps_ref = eps[:, 0] if eps_2d else eps
        lam = linalg.gcv(H, eps_ref, lam_grid)

    if eps_2d:
        T = eps.shape[1]
        n = H.shape[1]
        F = np.empty((n, T))
        for t in range(T):
            F[:, t] = linalg.tikhonov_solve(H, eps[:, t], lam, L)
        return F

    return linalg.tikhonov_solve(H, eps, lam, L)


def tsvd(H: NDArray, eps: NDArray, k: int) -> NDArray:
    """Reconstruct loads via truncated SVD.

    Solves the system by retaining only the *k* largest singular values of
    **H**, discarding the contribution of small singular values that would
    amplify noise.

    Parameters
    ----------
    H : NDArray, shape (m, n)
        System / sensitivity matrix.
    eps : NDArray, shape (m,)
        Measured strains (single snapshot only).
    k : int
        Number of singular values to retain.

    Returns
    -------
    NDArray, shape (n,)
        Reconstructed loads.
    """
    return linalg.tsvd_solve(H, eps, k)
