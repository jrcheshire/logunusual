"""Sampler contracts: Poisson identities, placement inversion, streaming bit-equality,
RSD translation, seed splitting."""

import numpy as np
import pytest

from logunusual import field, sample
from logunusual.grid import Box


def test_intensity_sums_to_nbar_volume_for_a_lognormal_field(spectrum):
    box = Box(16, 160.0)
    F = field.generate_fields(spectrum, 1.5, box, 4, rsd=False)
    lam, clip, mass = sample.intensity(F.delta_g, 2e-3, box)
    assert clip == 0.0 and mass == 0.0
    assert lam.sum() == pytest.approx(2e-3 * box.volume, rel=1e-12)
    assert np.all(lam >= 0)


def test_intensity_reports_negative_cells():
    box = Box(4, 4.0)
    delta = np.zeros(box.shape)
    delta[0, 0, 0] = -1.5
    lam, clip, mass = sample.intensity(delta, 1.0, box)
    assert clip == pytest.approx(1 / 64) and mass == pytest.approx(0.5 / 64)
    assert lam[0, 0, 0] == 0.0


@pytest.mark.parametrize("p", [1, 2, 3])
def test_placement_inverts_to_counts_and_stays_in_box(p):
    box = Box(6, 12.0)
    rng = np.random.default_rng(5)
    counts = rng.poisson(2.0, size=box.shape)
    xyz, cell = sample.place(counts, box, np.random.default_rng(6), jitter_p=p)
    assert xyz.shape == (counts.sum(), 3) and cell.shape == (counts.sum(),)
    assert np.all(xyz >= 0) and np.all(xyz < box.box_size)
    i, j, k = sample._decode_cells(cell, box.n_mesh)
    assert np.array_equal(np.bincount(cell, minlength=box.n_cells), counts.ravel())
    centre = (np.stack([i, j, k], 1) + 0.5) * box.dx
    d = xyz - centre
    d -= box.box_size * np.round(d / box.box_size)  # periodic distance
    assert np.all(np.abs(d) <= 0.5 * p * box.dx + 1e-12)
    if p == 1:  # uniform in the cell: floor recovers the cell exactly
        assert np.array_equal(np.floor(xyz / box.dx).astype(int).T, np.array([i, j, k]))


@pytest.mark.parametrize("chunk", [1, 7, 36, None])
def test_chunked_placement_is_bit_identical(chunk):
    box = Box(6, 12.0)
    counts = np.random.default_rng(7).poisson(1.5, size=box.shape)
    ref, ref_cell = sample.place(counts, box, np.random.default_rng(8), jitter_p=2)
    parts = list(
        sample.place_chunked(
            counts, box, np.random.default_rng(8), jitter_p=2, chunk_cells=chunk
        )
    )
    xyz = np.concatenate([p[0] for p in parts])
    cell = np.concatenate([p[1] for p in parts])
    assert np.array_equal(xyz, ref) and np.array_equal(cell, ref_cell)


def test_poisson_total_matches_expectation():
    box = Box(16, 16.0)
    lam = np.full(box.shape, 3.0)
    counts = sample.poisson_counts(lam, np.random.default_rng(9))
    tot = lam.sum()
    assert abs(counts.sum() - tot) < 5 * np.sqrt(tot)  # measured ~0.3 sigma


def test_rsd_constant_displacement_translates_z():
    box = Box(4, 40.0)
    xyz = np.random.default_rng(10).random((50, 3)) * box.box_size
    cell = np.zeros(50, dtype=np.int64)
    psi = np.full(box.n_cells, 7.0)
    out = sample.rsd_plane_parallel(xyz.copy(), cell, psi, 0.5, box)
    assert np.allclose(out[:, :2], xyz[:, :2])
    assert np.allclose(out[:, 2], np.mod(xyz[:, 2] + 3.5, box.box_size))


def test_sample_catalog_end_to_end_and_reproducible(spectrum):
    box = Box(16, 160.0)
    F = field.generate_fields(spectrum, 1.5, box, 4, rsd=True)
    a = sample.sample_catalog(F, 3e-3, 11, f=0.8, rsd=True)
    b = sample.sample_catalog(F, 3e-3, 11, f=0.8, rsd=True)
    c = sample.sample_catalog(F, 3e-3, 12, f=0.8, rsd=True)
    real = sample.sample_catalog(F, 3e-3, 11, f=0.8, rsd=False)
    assert np.array_equal(a.xyz, b.xyz) and np.array_equal(a.cell, b.cell)
    assert not np.array_equal(a.xyz, c.xyz)
    # same draw seed: same galaxies, only z moved by the RSD
    assert np.array_equal(a.cell, real.cell)
    assert np.array_equal(a.xyz[:, :2], real.xyz[:, :2])
    assert not np.allclose(a.xyz[:, 2], real.xyz[:, 2])
    assert a.lam_total == pytest.approx(3e-3 * box.volume, rel=1e-12)
    assert abs(a.n_galaxies - a.lam_total) < 5 * np.sqrt(a.lam_total)
    assert a.expected_nbar == pytest.approx(3e-3, rel=1e-12)
    with pytest.raises(ValueError):
        sample.sample_catalog(
            field.generate_fields(spectrum, 1.5, box, 4, rsd=False),
            3e-3,
            1,
            f=0.8,
            rsd=True,
        )


def test_rsd_moves_each_galaxy_by_its_own_cell_displacement(spectrum):
    # Exact: z_s - z_r = f Psi_z[cell] (mod L) galaxy by galaxy; x, y untouched.
    box = Box(16, 160.0)
    F = field.generate_fields(spectrum, 1.5, box, 8, rsd=True)
    f = 0.7
    real = sample.sample_catalog(F, 5e-3, 3, f=f, rsd=False)
    red = sample.sample_catalog(F, 5e-3, 3, f=f, rsd=True)
    psi = np.asarray(F.psi_z).reshape(-1)
    dz = (
        np.mod(red.xyz[:, 2] - real.xyz[:, 2] + 0.5 * box.box_size, box.box_size)
        - 0.5 * box.box_size
    )
    assert np.allclose(dz, f * psi[real.cell], atol=1e-12)
    assert np.array_equal(red.xyz[:, :2], real.xyz[:, :2])
    assert np.all((red.xyz >= 0) & (red.xyz < box.box_size))


def test_split_seed_is_deterministic_and_distinct():
    a = sample.split_seed(137_000_001)
    assert a == sample.split_seed(137_000_001)
    assert a[0] != a[1]
    assert a != sample.split_seed(137_000_002)
