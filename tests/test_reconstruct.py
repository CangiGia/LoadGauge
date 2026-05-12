"""Tests for loadgauge.reconstruct — static, tikhonov, and tsvd reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from loadgauge import reconstruct

RNG = np.random.default_rng(7)


def _make_C(m: int = 8, n: int = 4, cond: float = 10.0) -> np.ndarray:
    """Random well-conditioned (m × n) calibration matrix."""
    Q_m, _ = np.linalg.qr(RNG.standard_normal((m, m)))
    Q_n, _ = np.linalg.qr(RNG.standard_normal((n, n)))
    s = np.linspace(cond, 1.0, min(m, n))
    return (Q_m[:, : len(s)] * s) @ Q_n[: len(s), :]


def _ill_posed(n: int = 40, noise: float = 1e-3):
    """Simple Gaussian convolution (ill-posed) system and noisy RHS."""
    h = 1.0 / n
    t = np.linspace(-3, 3, n)
    H = h * np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2)
    x_true = np.exp(-0.5 * t**2)
    b = H @ x_true + noise * RNG.standard_normal(n)
    return H, b, x_true


# ---------------------------------------------------------------------------
# reconstruct.static
# ---------------------------------------------------------------------------


class TestStatic:
    def test_snapshot_recovery(self):
        """For a noiseless snapshot, static must recover f exactly."""
        C = _make_C()
        f_true = RNG.standard_normal(4)
        eps = C @ f_true
        f_hat = reconstruct.static(C, eps)
        assert_allclose(f_hat, f_true, atol=1e-10)

    def test_time_series_recovery(self):
        """For a noiseless time series, each column of f must be recovered."""
        C = _make_C(m=8, n=4)
        T = 50
        F_true = RNG.standard_normal((4, T))
        E = C @ F_true  # shape (8, T)
        F_hat = reconstruct.static(C, E)
        assert F_hat.shape == (4, T)
        assert_allclose(F_hat, F_true, atol=1e-10)

    def test_output_shape_1d(self):
        C = _make_C(m=6, n=3)
        eps = RNG.standard_normal(6)
        f = reconstruct.static(C, eps)
        assert f.shape == (3,)

    def test_output_shape_2d(self):
        C = _make_C(m=6, n=3)
        E = RNG.standard_normal((6, 20))
        F = reconstruct.static(C, E)
        assert F.shape == (3, 20)

    def test_noisy_snapshot_relative_error(self):
        """With 1 % relative noise, the relative error in f must stay below 20 %."""
        C = _make_C(m=8, n=4, cond=10.0)
        f_true = RNG.standard_normal(4)
        eps_clean = C @ f_true
        noise = 0.01 * np.linalg.norm(eps_clean) * RNG.standard_normal(8)
        f_hat = reconstruct.static(C, eps_clean + noise)
        rel_err = np.linalg.norm(f_hat - f_true) / np.linalg.norm(f_true)
        assert rel_err < 0.20


# ---------------------------------------------------------------------------
# reconstruct.tikhonov
# ---------------------------------------------------------------------------


class TestTikhonov:
    def test_snapshot_shape(self):
        H, b, _ = _ill_posed()
        f = reconstruct.tikhonov(H, b)
        assert f.shape == (len(b),)

    def test_time_series_shape(self):
        n = 30
        H, _, _ = _ill_posed(n=n)
        T = 10
        E = RNG.standard_normal((n, T))
        F = reconstruct.tikhonov(H, E)
        assert F.shape == (n, T)

    def test_explicit_lambda_accepted(self):
        H, b, _ = _ill_posed()
        f = reconstruct.tikhonov(H, b, lam=1e-3)
        assert f.shape == b.shape

    def test_auto_lambda_beats_lstsq(self):
        """GCV-selected Tikhonov must have lower error than naive least-squares."""
        H, b, x_true = _ill_posed(noise=5e-3)
        f_tik = reconstruct.tikhonov(H, b)
        f_ls, *_ = np.linalg.lstsq(H, b, rcond=None)
        err_tik = np.linalg.norm(f_tik - x_true)
        err_ls = np.linalg.norm(f_ls - x_true)
        assert err_tik < err_ls

    def test_custom_lam_grid(self):
        H, b, _ = _ill_posed()
        lam_grid = np.logspace(-4, 0, 50)
        f = reconstruct.tikhonov(H, b, lam_grid=lam_grid)
        assert f.shape == b.shape

    def test_custom_L_operator(self):
        H, b, _ = _ill_posed(n=20)
        n = H.shape[1]
        L = np.diff(np.eye(n), axis=0)
        f = reconstruct.tikhonov(H, b, lam=1e-2, L=L)
        assert f.shape == (n,)


# ---------------------------------------------------------------------------
# reconstruct.tsvd
# ---------------------------------------------------------------------------


class TestTsvd:
    def test_snapshot_shape(self):
        H, b, _ = _ill_posed()
        f = reconstruct.tsvd(H, b, k=5)
        assert f.shape == b.shape

    def test_full_rank_matches_lstsq(self):
        """k = min(m,n) must match the lstsq solution for a full-rank system."""
        m, n = 8, 4
        H = _make_C(m=m, n=n)
        b = RNG.standard_normal(m)
        f_ts = reconstruct.tsvd(H, b, k=n)
        f_ls, *_ = np.linalg.lstsq(H, b, rcond=None)
        assert_allclose(f_ts, f_ls, atol=1e-10)

    def test_low_k_smoother_solution(self):
        """For the ill-posed problem, small k should give lower residual than lstsq."""
        H, b, x_true = _ill_posed(noise=1e-2)
        f_ts = reconstruct.tsvd(H, b, k=5)
        f_ls, *_ = np.linalg.lstsq(H, b, rcond=None)
        err_ts = np.linalg.norm(f_ts - x_true)
        err_ls = np.linalg.norm(f_ls - x_true)
        assert err_ts < err_ls

    def test_invalid_k_propagated(self):
        H, b, _ = _ill_posed(n=10)
        with pytest.raises(ValueError):
            reconstruct.tsvd(H, b, k=0)
