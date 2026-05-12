"""Tests for loadgauge.calibration — identification and validation of C."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from loadgauge import calibration

RNG = np.random.default_rng(0)


def _make_calibration_data(
    m: int = 8,
    n: int = 4,
    K: int = 30,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (C_true, F_mat, E_mat) for a synthetic calibration experiment."""
    # well-conditioned C_true via random orthonormal factors
    Q_m, _ = np.linalg.qr(RNG.standard_normal((m, m)))
    Q_n, _ = np.linalg.qr(RNG.standard_normal((n, n)))
    s = np.linspace(5.0, 1.0, min(m, n))  # singular values in [1, 5]
    C_true = (Q_m[:, : len(s)] * s) @ Q_n[: len(s), :]

    F_mat = RNG.standard_normal((n, K))
    E_mat = C_true @ F_mat
    if noise_std > 0.0:
        E_mat = E_mat + RNG.normal(0.0, noise_std, E_mat.shape)

    return C_true, F_mat, E_mat


# ---------------------------------------------------------------------------
# identify
# ---------------------------------------------------------------------------


class TestIdentify:
    def test_noiseless_exact_recovery(self):
        """With zero noise and K >> n the identified C must match C_true exactly."""
        C_true, F_mat, E_mat = _make_calibration_data(m=8, n=4, K=40, noise_std=0.0)
        C_hat = calibration.identify(F_mat, E_mat)
        err_rel = np.linalg.norm(C_hat - C_true, "fro") / np.linalg.norm(C_true, "fro")
        assert err_rel < 1e-10

    def test_low_noise_relative_frobenius_below_threshold(self):
        """With moderate noise the relative Frobenius error must stay below 1 %."""
        C_true, F_mat, E_mat = _make_calibration_data(
            m=8, n=4, K=100, noise_std=1e-3
        )
        C_hat = calibration.identify(F_mat, E_mat)
        err_rel = np.linalg.norm(C_hat - C_true, "fro") / np.linalg.norm(C_true, "fro")
        assert err_rel < 0.01

    def test_output_shape(self):
        C_true, F_mat, E_mat = _make_calibration_data(m=6, n=3, K=20)
        C_hat = calibration.identify(F_mat, E_mat)
        assert C_hat.shape == (6, 3)

    def test_k_mismatch_raises(self):
        F_mat = RNG.standard_normal((4, 10))
        E_mat = RNG.standard_normal((8, 12))  # wrong K
        with pytest.raises(ValueError, match="mismatch"):
            calibration.identify(F_mat, E_mat)

    def test_underdetermined_raises(self):
        """K < n must raise ValueError."""
        F_mat = RNG.standard_normal((5, 3))  # n=5, K=3 < n
        E_mat = RNG.standard_normal((8, 3))
        with pytest.raises(ValueError, match="Under-determined"):
            calibration.identify(F_mat, E_mat)

    def test_minimal_system_square(self):
        """Exactly K = n load cases (square F_mat.T): should still work."""
        C_true, F_mat, E_mat = _make_calibration_data(m=6, n=4, K=4, noise_std=0.0)
        C_hat = calibration.identify(F_mat, E_mat)
        assert C_hat.shape == (6, 4)

    def test_more_gauges_than_loads(self):
        """m >> n is the typical over-sensed case."""
        C_true, F_mat, E_mat = _make_calibration_data(m=20, n=3, K=50, noise_std=0.0)
        C_hat = calibration.identify(F_mat, E_mat)
        err_rel = np.linalg.norm(C_hat - C_true, "fro") / np.linalg.norm(C_true, "fro")
        assert err_rel < 1e-10


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_perfect_C_gives_zero_rmse(self):
        C_true, F_mat, E_mat = _make_calibration_data(m=8, n=4, K=20, noise_std=0.0)
        report = calibration.validate(C_true, F_mat, E_mat)
        assert_allclose(report["rmse_global"], 0.0, atol=1e-12)

    def test_perfect_C_gives_unit_r2(self):
        C_true, F_mat, E_mat = _make_calibration_data(m=8, n=4, K=20, noise_std=0.0)
        report = calibration.validate(C_true, F_mat, E_mat)
        assert_allclose(report["r2_per_channel"], np.ones(8), atol=1e-12)
        assert_allclose(report["r2_global"], 1.0, atol=1e-12)

    def test_output_keys(self):
        C_true, F_mat, E_mat = _make_calibration_data()
        report = calibration.validate(C_true, F_mat, E_mat)
        for key in ("E_pred", "residuals", "rmse_per_channel", "r2_per_channel",
                    "rmse_global", "r2_global"):
            assert key in report

    def test_output_shapes(self):
        m, n, K = 8, 4, 20
        C_true, F_mat, E_mat = _make_calibration_data(m=m, n=n, K=K)
        report = calibration.validate(C_true, F_mat, E_mat)
        assert report["E_pred"].shape == (m, K)
        assert report["residuals"].shape == (m, K)
        assert report["rmse_per_channel"].shape == (m,)
        assert report["r2_per_channel"].shape == (m,)

    def test_noisy_identification_r2_near_one(self):
        """Identified C under mild noise should yield R² > 0.99 on training data."""
        C_true, F_mat, E_mat = _make_calibration_data(m=8, n=4, K=100, noise_std=1e-3)
        C_hat = calibration.identify(F_mat, E_mat)
        report = calibration.validate(C_hat, F_mat, E_mat)
        assert report["r2_global"] > 0.99

    def test_residuals_sign(self):
        """residuals = E_pred - E_mat, so E_pred = E_mat + residuals."""
        C_true, F_mat, E_mat = _make_calibration_data()
        report = calibration.validate(C_true, F_mat, E_mat)
        assert_allclose(report["E_pred"] - report["residuals"], E_mat, atol=1e-12)
