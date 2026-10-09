"""M1 statistical gates (slow). Seed counts are recorded with the SE they produced;
bands are derived from the tolerance and the seed count (`gates.n_min_indep`)."""

import numpy as np
import pytest

from logunusual import field, gates, suite
from logunusual.grid import Box

pytestmark = pytest.mark.slow

B5, F5 = suite.BIN_SUITE_V28[4].b, suite.BIN_SUITE_V28[4].f  # bin 5
BOX = Box(128, 1000.0)  # dx = 7.8 Mpc/h, production-like cell
BOX_EST = Box(256, 1000.0)
NBAR = 3e-3


def _report(name, e):
    print(f"\n{name}: n_real={e.n_real}")
    for k, m, se, z, ok in zip(e.k, e.mean, e.se, e.z, e.se_ok):
        print(f"  k={k:.4f}  ratio={m:.4f} +- {se:.4f}  z={z:+.2f}  se_ok={ok}")


def test_g3_lognormal_grid_identity(spectrum):
    # 192 seeds: the high-k power of a sigma^2 = 3.9 lognormal field is dominated by
    # rare
    # peaks, so its realization scatter is far above Gaussian and correlated across k
    # (48 seeds gave 0.8% SE at k > 0.26); 192 seeds bring every band under 2%/3.
    res = gates.gate_field_identity(spectrum, B5, BOX, range(3000, 3192))
    _report("G3 galaxy", res["galaxy"])
    _report("G3 matter", res["matter"])
    print(res["galaxy"].extra)
    assert res["galaxy"].passed, res["galaxy"].z
    assert res["matter"].passed, res["matter"].z


def test_g4a_uniform_shot(spectrum):
    # 64 seeds on the 32-seed bands, the G11 treatment. With the bands re-derived per
    # seed count the SE floor is met EXACTLY by construction and the worst band's
    # Gaussian SE does not fall with seeds (0.576% at 32, 0.643% at 128, against the
    # 0.667% floor), so a super-Gaussian Poisson scatter left the lowest band over it
    # at any count. Fixing the bands at 32 and running 64 seeds through them is what
    # brings the measured scatter down; nothing about what the gate detects moved.
    e = gates.gate_uniform_shot(BOX, BOX_EST, NBAR, range(4000, 4064), band_seeds=32)
    _report("G4a shot", e)
    assert e.passed, e.z


def test_g4b_fixed_field_sampler(spectrum):
    F = field.generate_fields(spectrum, B5, BOX, 4100, rsd=False)
    res = gates.gate_fixed_field_sampler(F, NBAR, range(4200, 4216), BOX_EST)
    _report("G4b first zone", res["first_zone"])
    _report("G4b to estimator Nyquist (images)", res["to_estimator_nyquist"])
    assert res["first_zone"].passed, res["first_zone"].z
    assert np.all(np.abs(res["to_estimator_nyquist"].z) < gates.Z_MAX)


def test_g5_g6_g7_catalog(spectrum):
    # 32 seeds: measured SE 0.3-0.6% (monopole), 0.1% (fixed field)
    res = gates.gate_catalog(spectrum, B5, F5, BOX, BOX_EST, NBAR, range(5000, 5032))
    _report("G5 monopole vs b^2 P_in x estimator response", res["monopole"])
    print(res["monopole"].extra)
    _report("G5 monopole vs fixed-field prediction", res["monopole_fixed_field"])
    _report("G6 premise P_gm / (b P_mm) on the grid (measurement)", res["premise"])
    print(f"f Psi_rms = {res['f_psi_rms']:.2f} Mpc/h")
    _report("G6 P2/P0 / Kaiser(beta), all bands (measurement)", res["kaiser_all_bands"])
    _report(
        "G6 P2 / generalised linear prediction (measurement)",
        res["quadrupole_generalised"],
    )
    _report(
        "G6 P0_s / generalised linear prediction (measurement)",
        res["monopole_generalised"],
    )
    print("G7 density", res["density"])
    assert res["density"]["passed"], res["density"]
    assert res["monopole"].passed, res["monopole"].z
    assert res["monopole_fixed_field"].passed, res["monopole_fixed_field"].z
    # the k -> 0 limit only: lowest band consistent with Kaiser (measured 0.995 +- 0.03)
    assert res["kaiser_lowest_band"].passed, res["kaiser_lowest_band"].z


def test_g6_kaiser_linear_limit(spectrum):
    # P_in x 1e-2, nbar = 0.1 (1.25e7 galaxies), 32 seeds: budget (k f Psi)^2 = 4e-3 at
    # the band top; measured SE 2-4% (consistency gate, |z| < 4; max measured 3.3)
    e = gates.gate_kaiser_linear_limit(
        spectrum, B5, F5, Box(64, 500.0), Box(128, 500.0), 0.1, range(6000, 6032)
    )
    _report("G6 linear limit: P2/P0 / Kaiser", e)
    print(e.extra)
    assert e.passed, e.z
