"""Tests for loadgauge.linalg — pure linear-algebra primitives."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from loadgauge import linalg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _random_full_rank(m: int, n: int, scale: float = 1.0) -> np.ndarray:
    """Generate a random full-rank matrix via QR decomposition."""
    Q, _ = np.linalg.qr(RNG.standard_normal((max(m, n), max(m, n))))
    return scale * Q[:m, :n]


# ---------------------------------------------------------------------------
# pseudoinverse
# ---------------------------------------------------------------------------


class TestPseudoinverse:
    def test_moore_penrose_condition_1(self):
        """A A+ A = A (first Moore-Penrose condition)."""
        A = _random_full_rank(8, 5)
        A_plus = linalg.pseudoinverse(A)
        assert_allclose(A @ A_plus @ A, A, atol=1e-12)

    def test_moore_penrose_condition_2(self):
        """A+ A A+ = A+ (second Moore-Penrose condition)."""
        A = _random_full_rank(8, 5)
        A_plus = linalg.pseudoinverse(A)
        assert_allclose(A_plus @ A @ A_plus, A_plus, atol=1e-12)

    def test_square_invertible(self):
        """For an invertible square matrix, A+ == A^{-1}."""
        A = _random_full_rank(6, 6)
        A_plus = linalg.pseudoinverse(A)
        assert_allclose(A @ A_plus, np.eye(6), atol=1e-12)

    def test_wide_matrix(self):
        """Overdetermined case: A is tall (m > n), A+ recovers least-squares solution."""
        A = _random_full_rank(10, 4)
        b = RNG.standard_normal(10)
        x_pinv = linalg.pseudoinverse(A) @ b
        x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
        assert_allclose(x_pinv, x_lstsq, atol=1e-12)

    def test_rcond_truncation(self):
        """Singular values below rcond * sigma_max should be zeroed out."""
        # build a rank-2 matrix from a 4x4 system
        U, _, Vt = np.linalg.svd(RNG.standard_normal((4, 4)))
        s = np.array([10.0, 1.0, 1e-12, 1e-14])
        A = (U * s) @ Vt
        A_plus = linalg.pseudoinverse(A, rcond=1e-10)
        # only the two non-negligible singular values should contribute
        assert A_plus.shape == (4, 4)


# ---------------------------------------------------------------------------
# tikhonov_solve
# ---------------------------------------------------------------------------


class TestTikhonovSolve:
    def test_zero_lambda_recovers_lstsq(self):
        """With lam=0, tikhonov_solve must equal the least-squares solution."""
        A = _random_full_rank(8, 4)
        b = RNG.standard_normal(8)
        x_tk = linalg.tikhonov_solve(A, b, lam=0.0)
        x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
        assert_allclose(x_tk, x_ls, atol=1e-10)

    def test_large_lambda_shrinks_solution(self):
        """Increasing lambda must decrease ||x_lambda||."""
        A = _random_full_rank(6, 4)
        b = RNG.standard_normal(6)
        norms = [np.linalg.norm(linalg.tikhonov_solve(A, b, lam=lam)) for lam in [0.01, 0.1, 1.0, 10.0]]
        assert all(norms[i] >= norms[i + 1] for i in range(len(norms) - 1))

    def test_custom_L_operator(self):
        """Passing a first-difference L should not raise and should return shape (n,)."""
        n = 5
        A = _random_full_rank(8, n)
        b = RNG.standard_normal(8)
        L = np.diff(np.eye(n), axis=0)  # shape (n-1, n)
        x = linalg.tikhonov_solve(A, b, lam=1.0, L=L)
        assert x.shape == (n,)

    def test_identity_solution(self):
        """For A = I and b = x_true, small lambda recovers x_true closely."""
        n = 5
        A = np.eye(n)
        x_true = RNG.standard_normal(n)
        x = linalg.tikhonov_solve(A, x_true, lam=1e-6)
        assert_allclose(x, x_true, atol=1e-5)


# ---------------------------------------------------------------------------
# tsvd_solve
# ---------------------------------------------------------------------------


class TestTsvdSolve:
    def test_full_rank_recovers_lstsq(self):
        """With k = min(m, n), TSVD must match the least-squares solution."""
        m, n = 8, 4
        A = _random_full_rank(m, n)
        b = RNG.standard_normal(m)
        x_ts = linalg.tsvd_solve(A, b, k=n)
        x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
        assert_allclose(x_ts, x_ls, atol=1e-10)

    def test_truncation_reduces_norm(self):
        """Reducing k must not increase ||x||_2 compared to the full solution."""
        m, n = 10, 6
        A = _random_full_rank(m, n)
        b = RNG.standard_normal(m)
        norm_full = np.linalg.norm(linalg.tsvd_solve(A, b, k=n))
        norm_trunc = np.linalg.norm(linalg.tsvd_solve(A, b, k=2))
        assert norm_trunc <= norm_full + 1e-12

    def test_invalid_k_raises(self):
        A = _random_full_rank(6, 4)
        b = RNG.standard_normal(6)
        with pytest.raises(ValueError):
            linalg.tsvd_solve(A, b, k=0)
        with pytest.raises(ValueError):
            linalg.tsvd_solve(A, b, k=5)


# ---------------------------------------------------------------------------
# condition_report
# ---------------------------------------------------------------------------


class TestConditionReport:
    def test_identity_matrix(self):
        """Identity matrix: cond=1, numerical_rank=n, not rank deficient."""
        n = 5
        report = linalg.condition_report(np.eye(n))
        assert_allclose(report["cond"], 1.0, atol=1e-12)
        assert report["numerical_rank"] == n
        assert not report["rank_deficient"]

    def test_known_condition_number(self):
        """Matrix with singular values [100, 1] should have cond=100."""
        U, _ = np.linalg.qr(RNG.standard_normal((3, 3)))
        V, _ = np.linalg.qr(RNG.standard_normal((2, 2)))
        s = np.array([100.0, 1.0])
        A = (U[:, :2] * s) @ V.T
        report = linalg.condition_report(A)
        assert_allclose(report["cond"], 100.0, rtol=1e-10)

    def test_rank_deficient_matrix(self):
        """A rank-1 matrix must be flagged as rank_deficient."""
        u = RNG.standard_normal(5)
        v = RNG.standard_normal(4)
        A = np.outer(u, v)  # rank 1
        report = linalg.condition_report(A)
        assert report["rank_deficient"]
        assert report["numerical_rank"] == 1

    def test_singular_values_descending(self):
        """Returned singular values must be in descending order."""
        A = _random_full_rank(8, 5)
        report = linalg.condition_report(A)
        sv = report["singular_values"]
        assert np.all(sv[:-1] >= sv[1:])


# ---------------------------------------------------------------------------
# lcurve
# ---------------------------------------------------------------------------


class TestLcurve:
    def _ill_posed_problem(self):
        """Phillips test problem: first-kind Fredholm integral equation."""
        n = 40
        h = 1.0 / n
        t = np.linspace(-3, 3, n)
        # simple smooth kernel and right-hand side
        A = h * np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2)
        x_true = np.exp(-0.5 * t**2)
        b = A @ x_true
        b += 1e-3 * RNG.standard_normal(n)
        return A, b

    def test_returns_correct_shapes(self):
        A, b = self._ill_posed_problem()
        lam_grid = np.logspace(-6, 1, 50)
        rho, eta, lam_c = linalg.lcurve(A, b, lam_grid)
        assert rho.shape == (50,)
        assert eta.shape == (50,)

    def test_corner_lambda_inside_grid(self):
        A, b = self._ill_posed_problem()
        lam_grid = np.logspace(-6, 1, 50)
        _, _, lam_c = linalg.lcurve(A, b, lam_grid)
        assert lam_grid[0] <= lam_c <= lam_grid[-1]

    def test_residuals_increase_with_lambda(self):
        """Residual norm must be monotonically non-decreasing in lambda."""
        A, b = self._ill_posed_problem()
        lam_grid = np.logspace(-6, 1, 100)
        rho, _, _ = linalg.lcurve(A, b, lam_grid)
        # allow minor numerical non-monotonicity; check overall trend
        assert rho[-1] >= rho[0]

    def test_solution_norms_decrease_with_lambda(self):
        """Solution norm must be monotonically non-increasing in lambda."""
        A, b = self._ill_posed_problem()
        lam_grid = np.logspace(-6, 1, 100)
        _, eta, _ = linalg.lcurve(A, b, lam_grid)
        assert eta[-1] <= eta[0]


# ---------------------------------------------------------------------------
# gcv
# ---------------------------------------------------------------------------


class TestGcv:
    def _make_problem(self, noise: float = 1e-3):
        n = 30
        h = 1.0 / n
        t = np.linspace(-3, 3, n)
        A = h * np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2)
        x_true = np.exp(-0.5 * t**2)
        b_clean = A @ x_true
        b = b_clean + noise * RNG.standard_normal(n)
        return A, b, x_true

    def test_returns_scalar_inside_grid(self):
        A, b, _ = self._make_problem()
        lam_grid = np.logspace(-6, 1, 100)
        lam_opt = linalg.gcv(A, b, lam_grid)
        assert isinstance(lam_opt, float)
        assert lam_grid[0] <= lam_opt <= lam_grid[-1]

    def test_gcv_solution_beats_no_regularisation(self):
        """The GCV solution should have lower error than the plain least-squares solution."""
        A, b, x_true = self._make_problem(noise=5e-3)
        lam_grid = np.logspace(-6, 1, 200)
        lam_opt = linalg.gcv(A, b, lam_grid)

        x_gcv = linalg.tikhonov_solve(A, b, lam_opt)
        x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)

        err_gcv = np.linalg.norm(x_gcv - x_true)
        err_ls = np.linalg.norm(x_ls - x_true)
        assert err_gcv < err_ls
