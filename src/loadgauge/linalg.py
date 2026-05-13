"""Pure linear-algebra primitives for LoadGauge.

Stateless helpers that operate on plain :class:`numpy.ndarray` objects.
Domain semantics (strain, loads, STM) live in :mod:`loadgauge.core`; this
module only handles matrices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pseudoinverse(A: NDArray, rcond: float | None = None) -> NDArray:
    """Compute the Moore-Penrose pseudo-inverse of *A*.

    Parameters
    ----------
    A : NDArray, shape (m, n)
        Matrix to invert.
    rcond : float or None, optional
        Singular values below ``rcond * sigma_max`` are treated as zero.
        ``None`` delegates to NumPy's default (``max(m, n) * eps``).

    Returns
    -------
    NDArray, shape (n, m)
        Pseudo-inverse of *A*.
    """
    return np.linalg.pinv(A, rcond=rcond)


def condition_report(A: NDArray) -> dict:
    """Return conditioning diagnostics for matrix *A*.

    Parameters
    ----------
    A : NDArray, shape (m, n)
        Matrix to analyse.

    Returns
    -------
    dict
        cond : float
            Condition number :math:`\\sigma_1 / \\sigma_{\\min}`.
            ``inf`` when the smallest singular value is zero.
        singular_values : NDArray, shape (min(m, n),)
            Singular values in descending order.
        numerical_rank : int
            Number of singular values above the tolerance
            ``sigma_1 * max(m, n) * eps``.
        rank_deficient : bool
            ``True`` when ``numerical_rank < min(m, n)``.
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
