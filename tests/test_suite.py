"""Invariants of the v28 bin table and the seed schedule (the M0 smoke test)."""

import math

import pytest

from logunusual import suite
from logunusual.suite import BIN_SUITE_V28, RADIAL_BUFFER, seed_for


def test_bin_indices_are_1_to_7_in_order():
    assert [b.index for b in BIN_SUITE_V28] == list(range(1, 8))
    assert [b.name for b in BIN_SUITE_V28] == [f"bin{i:02d}" for i in range(1, 8)]


def test_shells_tile_z_and_r_without_gaps():
    for lo, hi in zip(BIN_SUITE_V28[:-1], BIN_SUITE_V28[1:]):
        assert lo.z_max == hi.z_min
        assert lo.rmax == hi.rmin
    assert BIN_SUITE_V28[0].z_min == 0.0 and BIN_SUITE_V28[0].rmin == 0.0
    assert BIN_SUITE_V28[-1].z_max == 2.2


def test_box_holds_the_buffered_shell():
    # The drawn region is the shell padded by RADIAL_BUFFER on both sides, centred
    # on the observer: it must fit inside the box.
    for b in BIN_SUITE_V28:
        assert b.L_box >= 2 * (b.rmax + RADIAL_BUFFER), b.name


def test_cell_size_in_design_band():
    for b in BIN_SUITE_V28:
        assert 7.0 < b.cell < 16.0, (b.name, b.cell)
        assert math.isclose(b.k_nyquist, math.pi / b.cell)


def test_growth_rate_is_astropy_planck18_not_distance_cosmology():
    # f was generated with astropy Planck18 (Om0 = 0.30966 + 0.06 eV neutrino). A
    # bare flat-LCDM Om(z)^0.55 at that Om0 reproduces it to < 0.3%, while the
    # distance cosmology's Om0 = 0.3153 misses by ~0.5%. Pins the documented
    # inconsistency so nobody "fixes" the literals to the other cosmology by
    # accident.
    for b in BIN_SUITE_V28:
        f_astropy_like = suite.omega_m_flat_lcdm(b.z_eff, 0.30966) ** 0.55
        f_distance = suite.omega_m_flat_lcdm(b.z_eff, suite.OMEGA_M_DISTANCE) ** 0.55
        assert abs(b.f - f_astropy_like) < 3e-3, b.name
        assert abs(b.f - f_distance) > 3e-3, b.name
    fs = [b.f for b in BIN_SUITE_V28]
    assert fs == sorted(fs) and all(0 < f < 1 for f in fs)


def test_seed_schedule_matches_prod_v2_and_is_collision_free():
    assert seed_for(0, 1) == 137_000_001
    assert seed_for(0, 4) == 137_000_004  # prod_v2__r00000__bin04
    assert seed_for(99, 7) == 137_099_007
    seeds = {seed_for(r, b.index) for r in range(100) for b in BIN_SUITE_V28}
    assert len(seeds) == 700
    with pytest.raises(ValueError):
        seed_for(1000, 1)
    with pytest.raises(ValueError):
        seed_for(0, 0)


def test_matterpower_file_names_match_chimera_convention():
    assert BIN_SUITE_V28[0].matterpower_file == "data/matterpower_camb_zeff=0.1.tsv"
    assert BIN_SUITE_V28[4].matterpower_file == "data/matterpower_camb_zeff=0.9.tsv"
