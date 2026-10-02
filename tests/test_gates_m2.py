"""M2 statistical gates (slow): shell density and radial profile at production
amplitude (G10), and the estimator response on a 1x mesh (G11). Seed counts recorded
with the SE they produced."""

import numpy as np
import pytest

from logunusual import gates, shell, suite
from logunusual.grid import Box

pytestmark = pytest.mark.slow

B5, F5 = suite.BIN_SUITE_V28[4].b, suite.BIN_SUITE_V28[4].f  # bin 5
BOX = Box(128, 1000.0)  # dx 7.8 Mpc/h, production-like cell
# bin 5's shell thickness scaled into the box (71 Mpc/h = 9 cells), placed so that a
# 90 Mpc/h buffer fits under L/2: the window-max bound sqrt(3)/2 dx + f max|Psi| (the
# guard until M5) measured 40-68 Mpc/h across 16 seeds at this grid (2026-09-20), so the
# 30 Mpc/h the scaled production buffer gave was NOT sufficient and could not be
# raised in place (L/2 - rmax was 42.5).
SHELL = shell.Shell(300.0, 371.1, 90.0)
NBAR = 3e-3


def _report(name, e):
    print(f"\n{name}: n_real={e.n_real}")
    for k, m, se, z in zip(e.k, e.mean, e.se, e.z):
        print(f"  x={k:.3f}  ratio={m:.4f} +- {se:.4f}  z={z:+.2f}")


@pytest.fixture(scope="module")
def band_mask(tmp_path_factory):
    import h5py
    import healpy as hp

    nside = 32
    m = np.zeros(hp.nside2npix(nside), np.int8)
    th, _ = hp.pix2ang(nside, np.arange(m.size), nest=True)
    m[np.abs(np.cos(th)) < 0.7] = 1
    p = tmp_path_factory.mktemp("mask") / "band.h5"
    with h5py.File(p, "w") as f:
        f.attrs["PIXTYPE"] = "HEALPIX"
        f.attrs["ORDERING"] = "NESTED"
        f.create_dataset("MASK", data=m)
    return shell.AngularMask.from_h5(p)


def test_g10_shell_density_and_profile(spectrum, band_mask):
    # 16 seeds: the shell-scale sample variance of a b = 1.76 lognormal field gives
    # a per-realization scatter of a few percent in the total, so the SE is ~1%:
    # a consistency gate (|z| < 4), the SE printed.
    res = gates.gate_shell_density(
        spectrum, B5, F5, BOX, SHELL, band_mask, NBAR, range(7000, 7016)
    )
    _report("G10 N_kept / (nbar fsky V_shell)", res["total"])
    _report("G10 n(r) / nbar per sub-shell", res["profile"])
    print("G10 draws", res["draws"])
    assert res["draws"]["passed"], res["draws"]
    assert res["total"].passed, res["total"].z
    assert res["profile"].passed, res["profile"].z


def test_g11_one_x_mesh_response(spectrum):
    # The M1 catalog gate with the estimator on the generator's own mesh: the
    # deconvolved monopole over b^2 P_in x estimator_response, k < k_nyq/2, on the
    # 32-seed bands of G5 with 64 seeds: the bands are sized for a Gaussian scatter to
    # sit at the SE floor, and the lognormal ratio's scatter is super-Gaussian (with
    # bands re-derived per seed count, 32/48/64 seeds all left a band at 0.68-0.74%
    # against the 0.67% floor while every |z| was < 2.2). The alias-image sign fix of
    # 2026-09-04 was found by this gate (fixed-field arm |z| up to 43 before it).
    res = gates.gate_catalog(
        spectrum, B5, F5, BOX, BOX, NBAR, range(8000, 8064), band_seeds=32
    )
    _report("G11 1x mesh: monopole vs b^2 P_in x response", res["monopole"])
    print(res["monopole"].extra)
    _report(
        "G11 1x mesh: monopole vs fixed-field prediction", res["monopole_fixed_field"]
    )
    assert res["monopole"].passed, res["monopole"].z
    assert res["monopole_fixed_field"].passed, res["monopole_fixed_field"].z
