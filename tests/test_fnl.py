"""f_NL scale-dependent bias: the M(k) normalisation, the growth integral, the shape
of b(k), and the galaxy-table split (M from the linear table)."""

import warnings

import numpy as np
import pytest

from logunusual import fnl, suite
from logunusual.grid import Box

Z_TABLE = 0.9  # the tests/data table (conftest.PK_TSV)
B5 = 1.76

# Radiation (photons + 3.046 massless neutrinos), Omega_r h^2, for the M(k) bound.
OMEGA_R_H2 = 4.15e-5


def _m_residual(spectrum, png):
    """`M / M_{T=1}` at the table's lowest node (LSS convention); the reference's growth
    comes from the integral, never from `png`, so a wrong `png` cannot cancel."""
    k = spectrum.k[:1]
    om = suite.OMEGA_M_DISTANCE
    D = fnl.growth_md(Z_TABLE, om) / fnl.growth_md(0.0, om)
    m_lim = (2.0 / 3.0) * fnl.C_OVER_H0**2 * k**2 * D / om
    return float(fnl.poisson_M(k, spectrum, png)[0] / m_lim[0])


def _m_bound(spectrum, png):
    """Table M vs the T = 1 limit at the lowest node: radiation, which the integral
    omits, shifts D by order `(1 + z) a_eq`, and T(k) departs from 1 by at most ~2.2 q
    at low k (BBKS's linear coefficient, `q = k / (Om h)` in h/Mpc). Measured on all
    seven v28 tables: 1 - r = 4.5e-4 (z 0.1) to 1.25e-3 (z 1.9), 1.5-3x inside this
    bound; reading the pivot as 0.05 h/Mpc moves r by 0.7%, 4-5x outside it."""
    h = suite.H0_DISTANCE / 100.0
    a_eq = OMEGA_R_H2 / h**2 / png.omega_m
    q_min = spectrum.k[0] / (png.omega_m * h)
    return (1.0 + Z_TABLE) * a_eq + 2.2 * q_min


def test_m_matches_the_large_scale_limit(spectrum):
    png = fnl.LocalPNG(f_nl=1.0)
    r = _m_residual(spectrum, png)
    assert abs(1.0 - r) <= _m_bound(spectrum, png), r


@pytest.mark.parametrize(
    "mutation",
    ["pivot_in_h_per_mpc", "no_nine_25ths", "cmb_normalisation", "A_s_off_by_2pct"],
)
def test_m_check_fails_on_a_wrong_normalisation(spectrum, monkeypatch, mutation):
    png = fnl.LocalPNG(f_nl=1.0)
    if mutation == "pivot_in_h_per_mpc":
        png = fnl.LocalPNG(f_nl=1.0, k_pivot=0.05)
    elif mutation == "no_nine_25ths":
        monkeypatch.setattr(fnl, "_PHI_PER_ZETA_SQ", 1.0)
    elif mutation == "cmb_normalisation":
        object.__setattr__(png, "g0", 1.0)
    else:
        png = fnl.LocalPNG(f_nl=1.0, A_s=1.02 * suite.PRIMORDIAL_AS)
    r = _m_residual(spectrum, png)
    assert abs(1.0 - r) > _m_bound(spectrum, fnl.LocalPNG(f_nl=1.0)), r


def test_growth_integral():
    # Einstein-de Sitter: D = a exactly, so g0 = 1
    for z in (0.0, 0.5, 2.0):
        assert fnl.growth_md(z, 1.0) == pytest.approx(1.0 / (1.0 + z), rel=1e-12)
    # D -> a deep in matter domination; the Lambda term enters as (1 - Om)/Om a^3
    a = 1e-3
    assert fnl.growth_md(1.0 / a - 1.0, 0.3153) * (1.0 / a) == pytest.approx(
        1, abs=1e-8
    )
    g0 = fnl.LocalPNG(f_nl=0.0).g0
    assert 0.7 < g0 < 0.85  # LCDM suppression today for Om ~ 0.3


def test_bias_sign_scaling_and_null_cases(spectrum):
    png = fnl.LocalPNG(f_nl=10.0)
    k = np.array([1e-4, 2e-4, 1e-2])
    db = fnl.delta_b(k, B5, spectrum, png)
    assert np.all(db > 0)  # f_NL > 0 and b > p raise the large-scale bias
    # Delta b ~ 1 / (k^2 T): the ratio over a factor 2 in k is 4 up to T's departure
    h = suite.H0_DISTANCE / 100.0
    t_dev = 2.2 * k[1] / (png.omega_m * h)
    assert db[0] / db[1] == pytest.approx(4.0, rel=t_dev)
    neg = fnl.delta_b(k, B5, spectrum, fnl.LocalPNG(f_nl=-10.0))
    np.testing.assert_array_equal(neg, -db)  # linear in f_NL
    assert np.all(fnl.delta_b(k, 1.0, spectrum, png) == 0.0)  # b = p: no response


def test_zero_fnl_is_the_scalar_bias_bitwise(spectrum):
    k = np.geomspace(1e-4, 1.0, 257)
    old = B5 * B5 * spectrum(k)
    for png in (None, fnl.LocalPNG(f_nl=0.0)):
        np.testing.assert_array_equal(fnl.galaxy_spectrum(spectrum, B5, png)(k), old)


def test_galaxy_spectrum_at_dc_is_zero_and_silent(spectrum):
    P_g = fnl.galaxy_spectrum(spectrum, B5, fnl.LocalPNG(f_nl=-100.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = P_g(np.array([0.0, 1e-3]))
    assert out[0] == 0.0 and out[1] > 0


def test_diagnostics_find_the_zero_crossing(spectrum):
    box = Box(64, 5000.0)
    up = fnl.diagnostics(spectrum, B5, fnl.LocalPNG(f_nl=10.0), box)
    assert up["k_zero"] is None and up["b_kf_over_b"] > 1.0
    down = fnl.diagnostics(spectrum, B5, fnl.LocalPNG(f_nl=-100.0), box)
    assert down["b_kf_over_b"] < 0.0  # b(k) is negative at the fundamental
    kz = down["k_zero"]
    q = round((kz / box.k_f) ** 2)
    assert kz == box.k_f * np.sqrt(q)  # a grid |k|
    png = fnl.LocalPNG(f_nl=-100.0)
    k_prev = box.k_f * np.sqrt(np.arange(1, q))  # every grid |k| below it
    assert np.all(B5 + fnl.delta_b(k_prev, B5, spectrum, png) < 0.0)
    assert B5 + fnl.delta_b(np.array([kz]), B5, spectrum, png)[0] >= 0.0


def test_galaxy_table_takes_m_from_the_linear_table(spectrum):
    # target (b + delta_b[linear])^2 P_gal exactly; M from the galaxy table differs
    from logunusual.pk import PowerSpectrum

    k = np.geomspace(1e-4, 1.0, 257)
    nl = PowerSpectrum(spectrum.k, spectrum.P * (1 + (spectrum.k / 0.2) ** 2))
    png = fnl.LocalPNG(f_nl=50.0)
    got = fnl.galaxy_spectrum(spectrum, B5, png, nl)(k)
    want = (B5 + fnl.delta_b(k, B5, spectrum, png)) ** 2 * nl(k)
    np.testing.assert_array_equal(got, want)
    wrong = (B5 + fnl.delta_b(k, B5, nl, png)) ** 2 * nl(k)
    assert np.abs(got / wrong - 1).max() > 1e-3
    # f_NL = 0: exactly b * b * P_gal
    for p in (None, fnl.LocalPNG(f_nl=0.0)):
        np.testing.assert_array_equal(
            fnl.galaxy_spectrum(spectrum, B5, p, nl)(k), B5 * B5 * nl(k)
        )
