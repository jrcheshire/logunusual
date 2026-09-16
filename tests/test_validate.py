"""Estimator contracts: exact against brute force, CIC identities, shot noise."""

import numpy as np
import pytest

from logunusual import grid, validate
from logunusual.grid import Box


def _brute_force_multipoles(delta, box, ells=(0, 2, 4)):
    """Full-grid fftn, plain loops over Legendre weights, shells as in validate."""
    n = box.n_mesh
    dk = np.fft.fftn(delta)
    P = np.abs(dk) ** 2 * box.volume / n**6
    f = np.fft.fftfreq(n) * 2 * np.pi / box.dx
    kx, ky, kz = np.meshgrid(f, f, f, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    with np.errstate(invalid="ignore"):
        mu = np.where(kmag > 0, kz / kmag, 0.0)
    edges = validate.shell_edges(box)
    out = {}
    for ell in ells:
        L = validate._LEGENDRE[ell](mu)
        vals = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (kmag >= lo) & (kmag < hi)
            vals.append((2 * ell + 1) * (P[sel] * L[sel]).sum() / sel.sum())
        out[f"P{ell}"] = np.array(vals)
    return out


def test_multipoles_match_brute_force_on_full_grid():
    box = Box(16, 100.0)
    rng = np.random.default_rng(0)
    delta = rng.standard_normal(box.shape)
    delta *= (
        1.0 + 0.5 * np.cos(2 * np.pi * np.arange(16) / 16)[None, None, :]
    )  # anisotropy
    res = validate.power_multipoles(np.fft.rfftn(delta), box)
    ref = _brute_force_multipoles(delta, box)
    for ell in (0, 2, 4):
        assert np.allclose(res[f"P{ell}"], ref[f"P{ell}"], rtol=1e-11, atol=1e-11)
    assert res["nmodes"].sum() == pytest.approx(
        np.count_nonzero(_shell_mask(box)), abs=0
    )


def _shell_mask(box):
    n = box.n_mesh
    f = np.fft.fftfreq(n) * 2 * np.pi / box.dx
    kx, ky, kz = np.meshgrid(f, f, f, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    e = validate.shell_edges(box)
    return (kmag >= e[0]) & (kmag < e[-1])


def test_cross_power_and_shot_subtraction_are_linear():
    box = Box(16, 100.0)
    rng = np.random.default_rng(1)
    a = np.fft.rfftn(rng.standard_normal(box.shape))
    auto = validate.power_multipoles(a, box)
    cross = validate.power_multipoles(a, box, delta_k_b=3.0 * a)
    assert np.allclose(cross["P0"], 3.0 * auto["P0"], rtol=1e-12)
    shot = validate.power_multipoles(a, box, shot_k=2.5)
    assert np.allclose(shot["P0"], auto["P0"] - 2.5, rtol=1e-12)


def test_cic_paint_conserves_mass_and_places_corners_exactly():
    box = Box(8, 8.0)
    xyz = np.array([[2.0, 3.0, 4.0], [2.5, 3.5, 4.5], [7.9, 7.9, 7.9]])
    rho = validate.cic_paint(xyz, box)
    assert rho.sum() == pytest.approx(3.0)
    assert rho[2, 3, 4] == pytest.approx(
        1.0 + 0.125
    )  # corner point + 1/8 of the centred one
    # a full lattice of corner points is exactly uniform
    g = np.arange(8) * box.dx
    lat = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    assert np.allclose(validate.cic_paint(lat, box), 1.0)
    dk = validate.delta_k_from_positions(lat, box)
    assert np.allclose(dk, 0.0, atol=1e-12)


def test_uniform_poisson_catalog_has_jing_shot_noise():
    # Pure Poisson process: after CIC deconvolution the power is the Jing (2005) shot
    # spectrum. Per-mode variance of |delta_k|^2 is P^2 over the independent half-grid
    # modes (`gaussian_se`). Gate: |z| < 4.5 in every shell, mean z^2 ~ 1 (measured
    # ~1.1).
    box = Box(32, 100.0)
    rng = np.random.default_rng(2)
    n = 400_000
    xyz = rng.random((n, 3)) * box.box_size
    nbar = n / box.volume
    dk = validate.delta_k_from_positions(xyz, box)
    res = validate.power_multipoles(dk, box)
    shot_k = validate.shot_noise_k(box, nbar)
    pred = validate.shell_average(shot_k, box)
    z = (res["P0"] - pred) / validate.gaussian_se(shot_k, box)
    assert np.all(np.abs(z) < 4.5), z
    assert 0.4 < np.mean(z**2) < 1.8, np.mean(z**2)  # measured ~1.0
    # and the un-deconvolved shot is below 1/nbar (aliasing factor <= 1)
    assert np.all(
        validate.shot_noise_k(box, nbar, deconvolved=False) <= 1.0 / nbar + 1e-12
    )


def test_kaiser_formulae():
    assert validate.kaiser_boost(0, 0.0) == 1.0
    assert validate.kaiser_ratio(0.0) == 0.0
    beta = 0.5
    assert validate.kaiser_boost(0, beta) == pytest.approx(
        1 + 2 * beta / 3 + beta**2 / 5
    )
    assert validate.kaiser_boost(2, beta) == pytest.approx(
        4 * beta / 3 + 4 * beta**2 / 7
    )
    assert validate.kaiser_boost(4, beta) == pytest.approx(8 * beta**2 / 35)
    # mu-integral identity: sum_ell P_ell L_ell(mu) reconstructs (1 + beta mu^2)^2
    mu = np.linspace(-1, 1, 7)
    recon = sum(
        validate.kaiser_boost(ell, beta) * validate._LEGENDRE[ell](mu)
        for ell in (0, 2, 4)
    )
    assert np.allclose(recon, (1 + beta * mu**2) ** 2)


def test_rebin_is_weight_average():
    k = np.array([0.1, 0.2, 0.3, 0.4])
    v = np.array([1.0, 3.0, 5.0, 7.0])
    w = np.array([1.0, 3.0, 1.0, 1.0])
    kk, vv, ww = validate.rebin(k, v, w, np.array([0.05, 0.25, 0.45]))
    assert np.allclose(vv, [(1 + 9) / 4, 6.0]) and np.allclose(ww, [4.0, 2.0])
    assert np.allclose(kk, [(0.1 + 0.6) / 4, 0.35])


def test_effective_window_limits():
    box, box_est = Box(16, 160.0), Box(32, 160.0)
    # p = 0 (points ON the generator lattice): sum_n sinc^2(f + n) = 1 exactly, so the
    # un-deconvolved mesh power equals the grid power at every mode. The truncated sum
    # misses ~2/(pi^2 n_alias) per axis, squared and cubed: 6/(pi^2 n_alias) = 1.5e-4.
    W0 = validate.effective_window(box, box_est, 0, n_alias=4000)
    assert np.allclose(W0, 1.0, atol=3e-4)
    assert np.all(W0 <= 1.0)
    # p = 1: at k = 0 no aliasing, response 1; response < 1 elsewhere (the first image
    # has the opposite sign), and equal meshes give a larger departure than a 4x mesh
    R2 = validate.estimator_response(box, box_est, 1)
    R4 = validate.estimator_response(box, Box(64, 160.0), 1)
    assert R2[0, 0, 0] == pytest.approx(1.0) and R4[0, 0, 0] == pytest.approx(1.0)
    assert R2[4, 0, 0] < 1.0 and abs(R4[8, 0, 0] - 1) < abs(R2[4, 0, 0] - 1)  # same k
    with pytest.raises(ValueError):
        validate.effective_window(box, Box(24, 160.0), 1)


def test_effective_window_carries_the_half_cell_image_sign():
    # Cell centres at (i + 1/2) dx: image m of the grid coefficient carries (-1)^m.
    # On the generator's own mesh (ratio 1) image n has m = n; on a 2x mesh m = 2n.
    box = Box(8, 8.0)
    for r, p in ((1, 1), (2, 1), (1, 2), (3, 1)):
        est = Box(8 * r, 8.0)
        W = validate.effective_window(box, est, p, n_alias=64)
        f, fz = grid.frequencies(est)
        ns = np.arange(-64, 65)
        for i in range(est.n_mesh):
            x = f[i] + ns
            expect = (
                np.sum(
                    (-1.0) ** (np.abs(r * ns) % 2)
                    * np.sinc(x) ** 2
                    * np.sinc(r * x) ** p
                )
                ** 2
            )
            assert W[i, 0, 0] == pytest.approx(expect, rel=1e-12)
    # ratio 1, f = 1/4, p = 1: the n = -1 image subtracts (the hand sum)
    est = Box(8, 8.0)
    W1 = validate.effective_window(box, est, 1)
    i = 2  # f = 2/8 = 0.25
    terms = [(-1.0) ** (n % 2) * np.sinc(0.25 + n) ** 3 for n in range(-64, 65)]
    assert W1[i, 0, 0] == pytest.approx(sum(terms) ** 2, rel=1e-12)
    assert np.sinc(-0.75) ** 3 > 0 and terms[63] < 0  # the n = -1 term is negative
