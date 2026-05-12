"""Linear algebra primitives for load reconstruction.

All functions operate on plain :class:`numpy.ndarray` objects.
They are stateless and free of domain concepts (no "strain", no "load");
those semantics live in the higher-level modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Pseudo-inverse
# ---------------------------------------------------------------------------


def pseudoinverse(A: NDArray, rcond: float | None = None) -> NDArray:
    """Moore-Penrose pseudo-inverse of *A*.

    Thin wrapper around :func:`numpy.linalg.pinv` that makes the *rcond*
    cut-off explicit in the call-site rather than hidden in a default.

    Parameters
    ----------
    A : NDArray, shape (m, n)
        Matrix to invert.
    rcond : float or None
        Singular values below ``rcond * sigma_max`` are treated as zero.
        ``None`` delegates to the NumPy default (machine epsilon scaled by
        ``max(m, n)``).

    Returns
    -------
    NDArray, shape (n, m)
        Pseudo-inverse of *A*.
    """
    return np.linalg.pinv(A, rcond=rcond)


# ---------------------------------------------------------------------------
# Tikhonov regularisation
# ---------------------------------------------------------------------------


def tikhonov_solve(
    A: NDArray,
    b: NDArray,
    lam: float,
    L: NDArray | None = None,
) -> NDArray:
    """Solve the Tikhonov-regularised least-squares problem.

    Minimises

    .. math::

        \\min_{\\mathbf{x}} \\|\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\|_2^2
        + \\lambda^2 \\|\\mathbf{L}\\mathbf{x}\\|_2^2

    by forming the augmented system
    :math:`[\\mathbf{A};\\,\\lambda\\mathbf{L}]\\mathbf{x} = [\\mathbf{b};\\,\\mathbf{0}]`
    and calling :func:`numpy.linalg.lstsq`.

    Parameters
    ----------
    A : NDArray, shape (m, n)
    b : NDArray, shape (m,)
    lam : float
        Regularisation parameter :math:`\\lambda \\geq 0`.
    L : NDArray, shape (p, n) or None
        Regularisation operator. ``None`` defaults to the identity (standard
        Tikhonov).

    Returns
    -------
    NDArray, shape (n,)
        Regularised solution :math:`\\mathbf{x}_\\lambda`.
    """
    m, n = A.shape
    if L is None:
        L = np.eye(n)
    A_aug = np.vstack([A, lam * L])
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])
    x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    return x


# ---------------------------------------------------------------------------
# Truncated SVD
# ---------------------------------------------------------------------------


def tsvd_solve(A: NDArray, b: NDArray, k: int) -> NDArray:
    """Solve via truncated SVD, retaining the *k* largest singular values.

    Parameters
    ----------
    A : NDArray, shape (m, n)
    b : NDArray, shape (m,)
    k : int
        Number of singular values to retain.  Must satisfy
        ``1 <= k <= min(m, n)``.

    Returns
    -------
    NDArray, shape (n,)
        TSVD solution :math:`\\mathbf{x}_k`.

    Raises
    ------
    ValueError
        If *k* is out of range.
    """
    min_mn = min(A.shape)
    if not (1 <= k <= min_mn):
        raise ValueError(f"k must be in [1, {min_mn}], got {k}")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    x = (Vt[:k].T * (1.0 / s[:k])) @ (U[:, :k].T @ b)
    return x


# ---------------------------------------------------------------------------
# L-curve  (SVD-accelerated)
# ---------------------------------------------------------------------------


def lcurve(
    A: NDArray,
    b: NDArray,
    lam_grid: NDArray,
) -> tuple[NDArray, NDArray, float]:
    """Compute the L-curve for Tikhonov regularisation and locate its corner.

    The L-curve is a parametric log-log plot of the solution norm
    :math:`\\eta(\\lambda) = \\|\\mathbf{x}_\\lambda\\|_2` vs. the residual
    norm :math:`\\rho(\\lambda) = \\|\\mathbf{A}\\mathbf{x}_\\lambda -
    \\mathbf{b}\\|_2`.  The *corner* (maximum curvature) provides a data-driven
    estimate of the optimal :math:`\\lambda`.

    The norms are computed efficiently from the thin SVD of *A* without
    re-solving the system for every :math:`\\lambda`.

    Parameters
    ----------
    A : NDArray, shape (m, n)
    b : NDArray, shape (m,)
    lam_grid : NDArray, shape (p,)
        Candidate regularisation parameters, e.g. built with
        ``np.logspace(-6, 2, 200)``.

    Returns
    -------
    residual_norms : NDArray, shape (p,)
        :math:`\\rho(\\lambda_i)` for each grid point.
    solution_norms : NDArray, shape (p,)
        :math:`\\eta(\\lambda_i)` for each grid point.
    lam_corner : float
        :math:`\\lambda` value at the estimated corner.
    """
    U, s, _Vt = np.linalg.svd(A, full_matrices=False)
    Utb = U.T @ b  # projected right-hand side, shape (k,)
    b_perp_sq = float(np.linalg.norm(b) ** 2 - np.linalg.norm(Utb) ** 2)

    rho = np.empty(len(lam_grid))
    eta = np.empty(len(lam_grid))

    for i, lam in enumerate(lam_grid):
        f = s**2 / (s**2 + lam**2)  # filter factors
        eta[i] = np.sqrt(np.sum((f * Utb / s) ** 2))
        rho[i] = np.sqrt(np.sum(((1.0 - f) * Utb) ** 2) + b_perp_sq)

    # corner: maximum curvature of the log-log L-curve
    log_rho = np.log(rho)
    log_eta = np.log(eta)
    drho = np.gradient(log_rho)
    deta = np.gradient(log_eta)
    d2rho = np.gradient(drho)
    d2eta = np.gradient(deta)
    num = drho * d2eta - deta * d2rho
    den = (drho**2 + deta**2) ** 1.5
    curvature = np.where(den > 0.0, np.abs(num) / den, 0.0)

    idx_corner = int(np.argmax(curvature))
    return rho, eta, float(lam_grid[idx_corner])


# ---------------------------------------------------------------------------
# Generalised Cross-Validation  (SVD-accelerated)
# ---------------------------------------------------------------------------


def gcv(A: NDArray, b: NDArray, lam_grid: NDArray) -> float:
    """Select the regularisation parameter via Generalised Cross-Validation.

    Minimises the GCV functional

    .. math::

        G(\\lambda) =
        \\frac{\\|\\mathbf{A}\\mathbf{x}_\\lambda - \\mathbf{b}\\|_2^2}
        {\\left(\\operatorname{tr}(\\mathbf{I} - \\mathbf{H}_\\lambda) / m
        \\right)^2}

    where :math:`\\mathbf{H}_\\lambda = \\mathbf{A}(\\mathbf{A}^\\top
    \\mathbf{A} + \\lambda^2 \\mathbf{I})^{-1}\\mathbf{A}^\\top` is the
    influence (hat) matrix.  Computed efficiently from the thin SVD of *A*.

    Parameters
    ----------
    A : NDArray, shape (m, n)
    b : NDArray, shape (m,)
    lam_grid : NDArray, shape (p,)
        Candidate regularisation parameters.

    Returns
    -------
    float
        Optimal :math:`\\lambda` (minimiser of the GCV functional).
    """
    m = A.shape[0]
    U, s, _Vt = np.linalg.svd(A, full_matrices=False)
    Utb = U.T @ b
    b_perp_sq = float(np.linalg.norm(b) ** 2 - np.linalg.norm(Utb) ** 2)

    gcv_vals = np.empty(len(lam_grid))
    for i, lam in enumerate(lam_grid):
        d = s**2 / (s**2 + lam**2)  # filter factors (= diagonal of H projected)
        residual_sq = float(np.sum(((1.0 - d) * Utb) ** 2) + b_perp_sq)
        trace_imh = float(m - np.sum(d))  # tr(I - H_lam)
        gcv_vals[i] = residual_sq / (trace_imh / m) ** 2

    return float(lam_grid[int(np.argmin(gcv_vals))])


# ---------------------------------------------------------------------------
# Conditioning diagnostics
# ---------------------------------------------------------------------------


def condition_report(A: NDArray) -> dict:
    """Return conditioning diagnostics for matrix *A*.

    Parameters
    ----------
    A : NDArray, shape (m, n)

    Returns
    -------
    dict
        cond : float
            Condition number :math:`\\sigma_1 / \\sigma_{\\min}`.
            ``inf`` if the smallest singular value is zero.
        singular_values : NDArray
            All singular values in descending order.
        numerical_rank : int
            Number of singular values above the default tolerance
            ``sigma_1 * max(m, n) * eps``.
        rank_deficient : bool
            ``True`` if ``numerical_rank < min(m, n)``.
    """
    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0.0 else float("inf")
    tol = s[0] * max(A.shape) * np.finfo(float).eps
    num_rank = int(np.sum(s > tol))
    return {
        "cond": cond,
        "singular_values": s,
        "numerical_rank": num_rank,
        "rank_deficient": num_rank < min(A.shape),
    }
