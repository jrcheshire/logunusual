"""M4 statistical gates (slow): the grid identity with the f_NL bias (G13), the
matched-seed catalog ratio (G14), and the grid identity with a halofit galaxy target
(G15). Boxes are ones whose target is attainable at the values used (`ROADMAP.md` M4:
an f_NL or galaxy-table run that clips raises)."""

import pytest

from logunusual import gates
from logunusual.fnl import LocalPNG
from logunusual.grid import Box
from logunusual.pk import PowerSpectrum, check_table_pair
from tests.conftest import DATA

pytestmark = pytest.mark.slow

B5 = 1.76  # bin-5 bias (suite.py)


def _report(name, e):
    print(f"\n{name}: n_real={e.n_real}")
    for k, m, se, z, ok in zip(e.k, e.mean, e.se, e.z, e.se_ok):
        print(f"  k={k:.4f}  ratio={m:.4f} +- {se:.4f}  z={z:+.2f}  se_ok={ok}")


@pytest.mark.parametrize("f_nl, seed0", [(100.0, 8000), (-100.0, 8200)])
def test_grid_identity_with_fnl_bias(spectrum, f_nl, seed0):
    # G3's geometry and seed count (128^3, L 1000, 192 seeds); b(k_f)/b = 1.35 / 0.65
    box = Box(128, 1000.0)
    res = gates.gate_field_identity(
        spectrum, B5, box, range(seed0, seed0 + 192), fnl=LocalPNG(f_nl)
    )
    _report(f"galaxy, f_NL = {f_nl:+g}", res["galaxy"])
    _report(f"matter, f_NL = {f_nl:+g}", res["matter"])
    assert res["galaxy"].extra["clipped_power_fraction_max"] == 0.0
    assert res["galaxy"].passed, res["galaxy"].z
    assert res["matter"].passed, res["matter"].z


@pytest.mark.parametrize("f_nl, seed0", [(100.0, 8400), (-20.0, 8500)])
def test_fnl_matched_seed_catalog_ratio(spectrum, f_nl, seed0):
    # Consistency gate (|z| < 4, no SE floor), one band per kf-shell to k = 0.05.
    # 128^3, L 2000 (dx 15.6, k_f 3.1e-3); b(k_f)/b = 2.30 / 0.74. f_NL = -50 and -100
    # clip here and would raise.
    box, box_est = Box(128, 2000.0), Box(256, 2000.0)
    e = gates.gate_fnl_ratio(
        spectrum,
        B5,
        box,
        box_est,
        1e-3,
        range(seed0, seed0 + 32),
        LocalPNG(f_nl),
        k_max=0.05,
    )
    _report(f"matched ratio / prediction, f_NL = {f_nl:+g}", e)
    print(
        "predicted ratio per band:", [round(x, 4) for x in e.extra["predicted_ratio"]]
    )
    assert e.passed, e.z


def test_grid_identity_with_halofit_galaxy_target():
    # G3's geometry and seed count (128^3, L 1000, 192 seeds), z = 0.9 kh-10 tables:
    # galaxy b^2 P_halofit (sigma^2 5.45), matter P_lin (1.26); 0 clipped modes
    lin = PowerSpectrum.from_tsv(DATA / "matterpower_camb_lin_kmax10_zeff=0.9.tsv")
    hf = PowerSpectrum.from_tsv(DATA / "matterpower_camb_halofit_kmax10_zeff=0.9.tsv")
    assert check_table_pair(lin, hf) == 0.0
    res = gates.gate_field_identity(
        lin, B5, Box(128, 1000.0), range(8600, 8600 + 192), galaxy_table=hf
    )
    _report("galaxy, halofit", res["galaxy"])
    _report("matter, linear", res["matter"])
    assert res["galaxy"].extra["clipped_power_fraction_max"] == 0.0
    assert res["galaxy"].extra["sigma2_galaxy"] > 5.0  # the halofit target, not P_lin
    assert res["galaxy"].passed, res["galaxy"].z
    assert res["matter"].passed, res["matter"].z
