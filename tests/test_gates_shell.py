"""Shell-product statistical checks (slow): shell density and radial profile at
production amplitude, and the estimator response on the generator's own (1x) mesh. Seed
counts are recorded with the SE they produced; bars in `docs/validation.md`."""

import numpy as np
import pytest

from logunusual import gates, shell, suite
from logunusual.grid import Box

pytestmark = pytest.mark.slow

B5, F5 = suite.BIN_SUITE_V28[4].b, suite.BIN_SUITE_V28[4].f  # bin 5
BOX = Box(128, 1000.0)  # dx 7.8 Mpc/h, production-like cell
# bin 5's shell thickness scaled into the box (71 Mpc/h = 9 cells), placed so a 90
# Mpc/h buffer fits under L/2; sqrt(3)/2 dx + f max|Psi| over the window measures
# 40-68 Mpc/h across 16 seeds here, so the scaled production buffer (30) is too small.
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


def test_shell_density_and_profile(spectrum, band_mask):
    # 16 seeds: shell-scale sample variance gives a few percent scatter per realization
    # in the total, so SE ~1%: a consistency check (|z| < 4), the SE printed.
    res = gates.gate_shell_density(
        spectrum, B5, F5, BOX, SHELL, band_mask, NBAR, range(7000, 7016)
    )
    _report("shell N_kept / (nbar fsky V_shell)", res["total"])
    _report("shell n(r) / nbar per sub-shell", res["profile"])
    print("shell draws", res["draws"])
    assert res["draws"]["passed"], res["draws"]
    assert res["total"].passed, res["total"].z
    assert res["profile"].passed, res["profile"].z


def test_one_x_mesh_response(spectrum):
    # The box catalog check with the estimator on the generator's own mesh, k <
    # k_nyq/2. 64 seeds through bands sized for 32: the lognormal ratio's scatter is
    # super-Gaussian (bands re-derived per seed count leave a band at 0.68-0.74%
    # against the 0.67% floor at 32-64 seeds, every |z| < 2.2). Without the half-cell
    # alias-image sign in the response the fixed-field arm reaches |z| = 43.
    res = gates.gate_catalog(
        spectrum, B5, F5, BOX, BOX, NBAR, range(8000, 8064), band_seeds=32
    )
    _report("1x mesh: monopole vs b^2 P_in x response", res["monopole"])
    print(res["monopole"].extra)
    _report("1x mesh: monopole vs fixed-field prediction", res["monopole_fixed_field"])
    assert res["monopole"].passed, res["monopole"].z
    assert res["monopole_fixed_field"].passed, res["monopole_fixed_field"].z
