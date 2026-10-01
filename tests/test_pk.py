"""Spectrum loader and the grid-native P -> P_G transform (gate G1)."""

import math

import numpy as np
import pytest

from logunusual import pk
from logunusual.grid import Box, k_grid
from tests.conftest import PK_TSV


def test_tsv_loader_and_spline_nodes(spectrum):
    k, P, h = pk.load_pk_tsv(PK_TSV)
    assert k.size == 10001 and len(h) == 64
    assert k[0] == pytest.approx(1e-4) and k[-1] == pytest.approx(1.0)
    # nodes reproduced exactly (log-log cubic spline interpolates its nodes)
    assert np.allclose(spectrum(k), P, rtol=1e-12, atol=0)
    assert spectrum(0.0) == 0.0
    assert spectrum.file_hash == h


def test_spline_extrapolates_as_power_laws(spectrum):
    k, P = spectrum.k, spectrum.P
    lo = spectrum(k[0] / 10.0)
    hi = spectrum(k[-1] * 10.0)
    s_lo = math.log(P[1] / P[0]) / math.log(k[1] / k[0])
    s_hi = math.log(P[-1] / P[-2]) / math.log(k[-1] / k[-2])
    assert lo == pytest.approx(P[0] * 0.1**s_lo, rel=1e-9)
    assert hi == pytest.approx(P[-1] * 10.0**s_hi, rel=1e-9)
    assert 0.5 < s_lo < 1.5  # CDM: P ~ k^n_s at low k
    assert -3.5 < s_hi < -1.0
    assert np.allclose(spectrum.scaled(4.0)(k), 4.0 * P)


@pytest.mark.parametrize("n, L", [(16, 160.0), (32, 320.0), (64, 640.0), (64, 1000.0)])
@pytest.mark.parametrize("bias", [1.0, 1.76])
def test_pk_on_grid_table_matches_direct_evaluation(spectrum, n, L, bias):
    # `pk_on_grid` gathers a per-radius table; the reference evaluates the spectrum on
    # `k_grid`'s |k|. The index must be exact, and P may move only by rounding:
    # slope_max * |dk/k| (the two |k| constructions, measured here; <= 2 eps) plus
    # 4 ulp of max |ln P| (P = exp(spline(ln k)): each side's spline + exp rounds by
    # ~2 ulp of ln P). Measured 9 eps on every v28 table against a bound of ~37.
    box = Box(n, L)
    sp = lambda k: bias * bias * spectrum(k)  # noqa: E731
    _, _, kmag = k_grid(box)
    q = pk.radius_index(box)
    assert q.dtype == np.int32
    assert np.array_equal(q, np.rint((kmag / box.k_f) ** 2).astype(np.int32))

    m = kmag > 0
    dk = np.abs(box.k_f * np.sqrt(q[m].astype(np.float64)) / kmag[m] - 1).max()
    kk = np.geomspace(kmag[m].min(), kmag.max(), 4001)
    slope = np.abs(np.gradient(np.log(sp(kk)), np.log(kk))).max()
    direct = sp(kmag)
    tol = slope * dk + 4 * np.spacing(np.abs(np.log(direct[m])).max())

    P = pk.pk_on_grid(sp, box)
    assert P[0, 0, 0] == 0.0
    assert np.abs(P[m] / direct[m] - 1).max() <= tol


def _gaussian_pair(A, R):
    """xi(r) = A exp(-r^2 / 2R^2)  <->  P(k) = A (2pi)^{3/2} R^3 exp(-k^2 R^2 / 2)."""
    P = (
        lambda k: A * (2 * math.pi) ** 1.5 * R**3 * np.exp(-0.5 * (k * R) ** 2)
    )  # noqa: E731
    xi = lambda r: A * np.exp(-0.5 * (r / R) ** 2)  # noqa: E731
    return P, xi


def _grid_r(box):
    x = np.fft.fftfreq(box.n_mesh) * box.box_size  # signed separations
    return np.sqrt(
        x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2
    )


def test_grid_xi_matches_closed_form_gaussian_pair():
    # L/R = 16 (periodic images ~ exp(-128)), k_nyq R = 4 pi (band-limit ~ exp(-79)):
    # the Riemann sum is spectrally accurate, so the only difference from the closed
    # form is the DC term P(0)/V that `pk_on_grid` removes.
    box = Box(64, 640.0)
    A, R = 0.7, 40.0
    P, xi = _gaussian_pair(A, R)
    pgrid = pk.pk_on_grid(P, box)
    assert pgrid[0, 0, 0] == 0.0
    xi_grid = pk.grid_xi(pgrid, box)
    expected = xi(_grid_r(box)) - P(0.0) / box.volume
    assert np.allclose(xi_grid, expected, rtol=0, atol=1e-10 * A)
    # round trip is exact
    back = pk.grid_pk_from_xi(xi_grid, box)
    assert np.allclose(back, pgrid, rtol=1e-12, atol=1e-12 * pgrid.max())


def test_grid_pkG_second_order_expansion_is_exact_for_gaussian_pair():
    # xi_G = log1p(eps xi) = eps xi - eps^2 xi^2 / 2 + O(eps^3); xi^2 is itself Gaussian
    # with width R/sqrt(2), so its transform is closed-form. Removing the DC mode shifts
    # xi by c0 = P(0)/V, whose square adds `+eps c0 P(k)` at second order. The residual
    # after removing the two leading orders must scale as eps^3. At high k the
    # second-order term
    # (width R/sqrt 2 in k-space) outlives the first (width R): the UNCLIPPED P_G goes
    # negative there, which is the Xavier et al. non-attainability, and `grid_pkG` must
    # report exactly those modes as clipped.
    box = Box(64, 640.0)
    A, R = 1.0, 40.0
    P, xi = _gaussian_pair(A, R)
    P2, _ = _gaussian_pair(A * A, R / math.sqrt(2))  # transform of xi^2
    _, _, kmag = k_grid(box)
    residuals = []
    for eps in (1e-2, 1e-3):
        pgrid = pk.pk_on_grid(lambda k: eps * P(k), box)
        unclipped = pk.grid_pk_from_xi(np.log1p(pk.grid_xi(pgrid, box)), box)
        unclipped[0, 0, 0] = 0.0
        c0 = P(0.0) / box.volume
        pred = pgrid * (1.0 + eps * c0) - 0.5 * eps**2 * P2(kmag)  # pgrid carries eps
        pred[0, 0, 0] = 0.0
        residuals.append(np.abs(unclipped - pred).max() / pgrid.max())
        pkG, diag = pk.grid_pkG(pgrid, box)
        assert diag["n_clipped"] == np.count_nonzero(unclipped < 0) > 0
        assert np.allclose(
            pkG, np.maximum(unclipped, 0.0), rtol=1e-12, atol=1e-12 * eps
        )
        assert diag["sigma2"] == pytest.approx(
            eps * (A - P(0.0) / box.volume), rel=1e-9
        )
    # third order: shrinks by 100x for a 10x smaller eps (measured 1.0e-6 -> 1.0e-9)
    assert residuals[0] < 1e-5
    assert residuals[1] < residuals[0] / 50


def test_grid_pkG_rejects_unattainable_and_reports_clipping():
    box = Box(16, 16.0)
    kx = k_grid(box)[0]

    # a single mode along x with amplitude a: xi = a cos(k0 x)
    def one_mode(a):
        pgrid = np.zeros(box.rfft_shape)
        pgrid[1, 0, 0] = pgrid[-1, 0, 0] = 0.5 * a * box.volume
        return pgrid

    with pytest.raises(ValueError, match="no lognormal"):
        pk.grid_pkG(one_mode(1.2), box)
    # log1p(a cos t) has Fourier coefficients 2 (-1)^{n+1} rho^n / n: alternating signs,
    # so every even harmonic of P_G is negative and gets clipped (and reported).
    pkG, diag = pk.grid_pkG(one_mode(0.9), box)
    assert diag["n_clipped"] > 0 and diag["clipped_power_fraction"] > 0
    assert pkG.min() == 0.0 and pkG[0, 0, 0] == 0.0
    assert pkG[1, 0, 0] > 0 and pkG[2, 0, 0] == 0.0 and pkG[3, 0, 0] > 0
    assert kx[1] == pytest.approx(box.k_f)


def test_grid_pkG_jax_and_numpy_agree(spectrum):
    import jax.numpy as jnp

    box = Box(32, 320.0)
    pgrid = pk.pk_on_grid(spectrum, box)
    a, da = pk.grid_pkG(pgrid, box, np)
    b, db = pk.grid_pkG(pgrid, box, jnp)
    assert np.allclose(a, np.asarray(b), rtol=1e-12, atol=1e-12 * a.max())
    assert da["n_clipped"] == db["n_clipped"]
