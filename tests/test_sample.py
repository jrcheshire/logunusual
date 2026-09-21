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
    xyz, cell = sample.place(counts, box, 6, jitter_p=p)
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


def test_slab_streams_are_deterministic_distinct_and_local():
    # The same (seed, slab) gives the same numbers; different slabs and different
    # seeds give different numbers; a slab's placement does not move when another
    # slab's counts change (each slab is its own stream).
    a = sample.slab_rng(8, 3).random(4)
    assert np.array_equal(a, sample.slab_rng(8, 3).random(4))
    assert not np.array_equal(a, sample.slab_rng(8, 4).random(4))
    assert not np.array_equal(a, sample.slab_rng(9, 3).random(4))
    box = Box(6, 12.0)
    counts = np.random.default_rng(7).poisson(1.5, size=box.shape)
    ref, ref_cell = sample.place(counts, box, 8, jitter_p=2)
    other = counts.copy()
    other[0] = 0  # empty the first slab
    other[2] += 3
    xyz, cell = sample.place(other, box, 8, jitter_p=2)
    for i in (1, 3, 4, 5):  # untouched slabs: identical galaxies
        assert np.array_equal(xyz[cell // 36 == i], ref[ref_cell // 36 == i])
    assert not np.any(cell // 36 == 0)


@pytest.mark.parametrize("p", [1, 2])
def test_place_slab_order_one_matches_the_general_path(p):
    # jitter_p == 1 draws (n, 3) directly; the general path draws (n, 3, p) and sums.
    # For p == 1 the two must be bitwise the same numbers.
    box = Box(4, 8.0)
    counts = np.random.default_rng(3).poisson(2.0, size=(box.n_mesh, box.n_mesh))
    xyz, cell = sample.place_slab(counts, 1, box, sample.slab_rng(5, 1), jitter_p=p)
    rng = sample.slab_rng(5, 1)
    n = counts.sum()
    u = rng.random((n, 3, p)).sum(axis=2) - 0.5 * (p - 1)
    i, j, k = sample._decode_cells(cell, box.n_mesh)
    ref = (u + np.stack([i, j, k], 1)) * box.dx
    if p > 1:
        ref = np.mod(ref, box.box_size)
    assert np.array_equal(xyz, ref)
    assert np.all(i == 1)


def test_draw_slabs_matches_slabwise_poisson_then_place():
    # The slab stream is used in a fixed order: Poisson over the slab, then uniforms.
    box = Box(6, 12.0)
    lam = np.random.default_rng(2).random(box.shape) * 3.0
    got = list(sample.draw_slabs(lam, box, 11))
    ref = []
    for i in range(box.n_mesh):
        rng = sample.slab_rng(11, i)
        counts = rng.poisson(lam[i])
        xyz, cell = sample.place_slab(counts, i, box, rng)
        if xyz.shape[0]:
            ref.append((xyz, cell))
    assert len(got) == len(ref)
    for (a, ca), (b, cb) in zip(got, ref):
        assert np.array_equal(a, b) and np.array_equal(ca, cb)


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
