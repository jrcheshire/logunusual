"""Shell product contracts: geometry identities, mask and selection identities, radial
RSD identities and its plane-parallel limit, the exact Poisson gate on a uniform
field, streaming bit-equality."""

import numpy as np
import pytest

from logunusual import field, sample, shell, suite
from logunusual.grid import Box
from logunusual.pk import PowerSpectrum


def _mask_h5(path, nside=16, ordering="NESTED", band=0.6, dtype=np.int8):
    import h5py
    import healpy as hp

    m = np.zeros(hp.nside2npix(nside), dtype)
    th, _ = hp.pix2ang(nside, np.arange(m.size), nest=(ordering == "NESTED"))
    m[np.abs(np.cos(th)) < band] = 1
    m[:8] = 0  # a hole, so the map is not a pure band
    with h5py.File(path, "w") as f:
        f.attrs["PIXTYPE"] = "HEALPIX"
        f.attrs["ORDERING"] = ordering
        f.create_dataset("MASK", data=m)
    return m


@pytest.fixture
def mask(tmp_path):
    _mask_h5(tmp_path / "mask.h5")
    return shell.AngularMask.from_h5(tmp_path / "mask.h5")


# ----------------------------------------------------------------------- geometry


def test_shell_validation():
    with pytest.raises(ValueError):
        shell.Shell(10.0, 5.0, 1.0)
    with pytest.raises(ValueError):
        shell.Shell(-1.0, 5.0, 1.0)
    with pytest.raises(ValueError):
        shell.Shell(1.0, 5.0, -1.0)
    s = shell.Shell(10.0, 50.0, 20.0)
    assert s.r_lo == 0.0 and s.r_hi == 70.0
    assert s.volume == pytest.approx(4 / 3 * np.pi * (50.0**3 - 10.0**3))


def test_check_box_rejects_overflow_and_thin_buffer():
    box = Box(16, 160.0)  # dx 10, diagonal 17.3
    shell.Shell(20.0, 50.0, 20.0).check_box(box)
    with pytest.raises(ValueError, match="exceeds"):
        shell.Shell(20.0, 70.0, 20.0).check_box(box)  # 90 > 80
    with pytest.raises(ValueError, match="diagonal"):
        shell.Shell(20.0, 50.0, 15.0).check_box(box)


def test_default_bins_buffer_exceeds_cell_diagonal():
    for b in suite.BIN_SUITE_V28:
        shell.Shell(b.rmin, b.rmax, suite.RADIAL_BUFFER).check_box(
            Box(b.N_grid, float(b.L_box))
        )


def test_cell_window_matches_brute_force_and_shell_volume():
    box = Box(20, 200.0)
    s = shell.Shell(30.0, 70.0, 20.0)
    W = shell.cell_window(box, s)
    ref = np.zeros(box.shape, bool)
    for i in range(20):
        for j in range(20):
            for k in range(20):
                c = (np.array([i, j, k]) + 0.5) * box.dx - 100.0
                r = np.linalg.norm(c)
                ref[i, j, k] = s.r_lo <= r <= s.r_hi
    assert np.array_equal(W, ref)
    # cell count x V_cell vs the continuum volume: the difference is bounded by the
    # volume of the cells cut by the two boundary spheres
    v_cells = W.sum() * box.v_cell
    v_cont = 4 / 3 * np.pi * (s.r_hi**3 - s.r_lo**3)
    layer = 4 * np.pi * (s.r_lo**2 + s.r_hi**2) * np.sqrt(3.0) * box.dx
    assert abs(v_cells - v_cont) < layer


# --------------------------------------------------------------- mask and selection


def test_mask_from_h5_both_orderings_and_validation(tmp_path):
    import healpy as hp

    m_nest = _mask_h5(tmp_path / "n.h5", ordering="NESTED")
    m_ring = _mask_h5(tmp_path / "r.h5", ordering="RING")
    a = shell.AngularMask.from_h5(tmp_path / "n.h5")
    b = shell.AngularMask.from_h5(tmp_path / "r.h5")
    assert a.nested and not b.nested and a.nside == 16 == b.nside
    assert a.fsky == pytest.approx(m_nest.mean()) and b.fsky == pytest.approx(
        m_ring.mean()
    )
    assert a.sha256 != b.sha256 and len(a.sha256) == 64
    # a direction is kept iff its pixel is set, in each ordering
    v = np.random.default_rng(0).standard_normal((500, 3))
    pn = hp.vec2pix(16, v[:, 0], v[:, 1], v[:, 2], nest=True)
    pr = hp.vec2pix(16, v[:, 0], v[:, 1], v[:, 2], nest=False)
    assert np.array_equal(a.contains(v), m_nest[pn].astype(bool))
    assert np.array_equal(b.contains(v), m_ring[pr].astype(bool))
    _mask_h5(tmp_path / "bad.h5", dtype=np.float64)
    import h5py

    with h5py.File(tmp_path / "bad.h5", "a") as f:
        f["MASK"][0] = 0.5
    with pytest.raises(ValueError, match="0/1"):
        shell.AngularMask.from_h5(tmp_path / "bad.h5")
    with h5py.File(tmp_path / "bad2.h5", "w") as f:
        f.attrs["ORDERING"] = "WEIRD"
        f.create_dataset("MASK", data=np.ones(12, np.int8))
    with pytest.raises(ValueError, match="ORDERING"):
        shell.AngularMask.from_h5(tmp_path / "bad2.h5")


def test_select_identity_and_inclusive_edges(mask):
    import healpy as hp

    s = shell.Shell(30.0, 70.0, 10.0)
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-80, 80, (20000, 3))
    keep = shell.select(xyz, s, mask)
    r = np.linalg.norm(xyz, axis=1)
    th = np.arccos(xyz[:, 2] / r)
    ph = np.arctan2(xyz[:, 1], xyz[:, 0])
    ref = (r >= 30.0) & (r <= 70.0) & mask.values[hp.ang2pix(16, th, ph, nest=True)]
    assert np.array_equal(keep, ref)
    assert 0 < keep.sum() < keep.size
    # without a mask only the radial cut acts
    assert np.array_equal(shell.select(xyz, s, None), (r >= 30.0) & (r <= 70.0))
    # inclusive edges, on an axis inside the mask band
    edge = np.array(
        [[30.0, 0.0, 0.0], [70.0, 0.0, 0.0], [np.nextafter(70.0, 80), 0, 0]]
    )
    assert np.array_equal(shell.select(edge, s, None), [True, True, False])


# ------------------------------------------------------------------- radial RSD


def test_rsd_radial_identities():
    rng = np.random.default_rng(2)
    n = 300
    xyz = rng.uniform(-50, 50, (n, 3))
    cell = rng.integers(0, 10, n)
    psi = rng.standard_normal((10, 3)) * 3.0
    f = 0.7
    s = shell.rsd_radial(xyz.copy(), cell, psi, f)
    d = s - xyz
    r = np.linalg.norm(xyz, axis=1)
    xhat = xyz / r[:, None]
    # parallel to x_hat, with the projected displacement as magnitude
    assert np.allclose(np.cross(d, xhat), 0.0, atol=1e-12)
    assert np.allclose(
        np.einsum("ij,ij->i", d, xhat),
        f * np.einsum("ij,ij->i", psi[cell], xhat),
        atol=1e-12,
    )
    # f = 0 is the identity; a galaxy at the origin is left in place
    assert np.array_equal(shell.rsd_radial(xyz.copy(), cell, psi, 0.0), xyz)
    z = np.zeros((1, 3))
    assert np.array_equal(shell.rsd_radial(z.copy(), np.array([0]), psi, f), z)
    # on the z axis the radial shift is the plane-parallel one
    box = Box(4, 40.0)
    ax = np.array([[0.0, 0.0, 12.5], [0.0, 0.0, -7.0]])
    rad = shell.rsd_radial(ax.copy(), np.array([1, 2]), psi, f)
    pp = (
        sample.rsd_plane_parallel(ax.copy() + 20.0, np.array([1, 2]), psi[:, 2], f, box)
        - 20.0
    )
    assert np.allclose(rad[:, :2], 0.0) and np.allclose(rad[:, 2], pp[:, 2], atol=1e-12)


def test_rsd_radial_far_observer_is_plane_parallel():
    # Observer at -D z_hat, D >> |x|: the radial shift tends to f Psi_z z_hat with a
    # difference bounded by 3 f |Psi| |x| / D (x_hat' = z_hat + O(|x|/D)).
    rng = np.random.default_rng(3)
    n = 200
    xyz = rng.uniform(-50, 50, (n, 3))
    cell = np.arange(n)
    psi = rng.standard_normal((n, 3)) * 3.0
    f = 0.8
    pp = xyz.copy()
    pp[:, 2] += f * psi[:, 2]
    prev = None
    for D in (1e3 * 100, 1e4 * 100):
        shifted = xyz.copy()
        shifted[:, 2] += D
        rad = shell.rsd_radial(shifted, cell, psi, f)
        rad[:, 2] -= D
        delta = np.linalg.norm(rad - pp, axis=1)
        bound = 3.0 * f * np.linalg.norm(psi, axis=1) * np.linalg.norm(xyz, axis=1) / D
        assert np.all(delta <= bound)
        if prev is not None:
            assert delta.max() < 0.2 * prev  # scales down with D
        prev = delta.max()


def test_rsd_moves_galaxies_across_the_shell_edges():
    s = shell.Shell(30.0, 70.0, 10.0)
    eps = 1e-3
    psi = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    f = 1.0
    # outward Psi carries a galaxy just inside rmax out; inward Psi brings one just
    # outside back in
    xyz = np.array([[70.0 - eps, 0.0, 0.0], [70.0 + eps, 0.0, 0.0]])
    keep_real = shell.select(xyz.copy(), s, None)
    red = shell.rsd_radial(xyz.copy(), np.array([0, 1]), psi, f)
    keep_red = shell.select(red, s, None)
    assert keep_real.tolist() == [True, False]
    assert keep_red.tolist() == [False, True]


# -------------------------------------------------------- streamed shell sampling


def _uniform_fields(spectrum, box, seed, eps=1e-4):
    scaled = lambda k: eps * spectrum(k)  # noqa: E731
    return field.generate_fields(scaled, 1.5, box, seed, psi_axes="xyz")


def test_sample_shell_uniform_field_is_poisson_in_the_shell(spectrum, mask):
    # With P_in x 1e-4 the field is uniform to 1e-2 in delta, uniform-in-cell
    # placement covers the shell, HEALPix pixels are equal-area: N_kept is Poisson
    # with mean nbar fsky V_shell (exact), and the draws over the window are
    # Poisson with mean sum(lambda).
    box = Box(24, 240.0)  # dx 10
    s = shell.Shell(40.0, 90.0, 20.0)
    nbar = 2e-2
    F = _uniform_fields(spectrum, box, 11)
    parts = []
    for kept, stats in shell.sample_shell(F, s, mask, nbar, 12, f=0.8):
        parts.append(kept)
    xyz = np.concatenate(parts)
    assert xyz.shape[0] == stats.n_kept
    mean = nbar * mask.fsky * s.volume
    z = (stats.n_kept - mean) / np.sqrt(mean)
    assert abs(z) < 4.0, z
    z_draw = (stats.n_drawn - stats.lam_window) / np.sqrt(stats.lam_window)
    assert abs(z_draw) < 4.0, z_draw
    # every kept galaxy satisfies the selection; the radial histogram adds up
    assert np.all(shell.select(xyz, s, mask))
    assert stats.r_hist.sum() == stats.n_kept
    # a galaxy that left the box sits beyond rmax + buffer: never among the kept
    assert np.all(np.abs(xyz).max(axis=1) < 0.5 * box.box_size)
    # the radial profile is flat: sub-shell counts vs their volumes
    v_sub = 4 / 3 * np.pi * np.diff(stats.r_edges**3) * mask.fsky
    zs = (stats.r_hist - nbar * v_sub) / np.sqrt(nbar * v_sub)
    assert np.all(np.abs(zs) < 4.0), zs


@pytest.mark.parametrize("chunk", [1, 37, None])
def test_sample_shell_chunking_is_bit_identical(spectrum, mask, chunk):
    box = Box(16, 160.0)
    s = shell.Shell(20.0, 50.0, 20.0)
    F = field.generate_fields(spectrum, 1.5, box, 21, psi_axes="xyz")

    def run(c):
        parts = []
        for kept, st in shell.sample_shell(F, s, mask, 5e-3, 22, f=0.8, chunk_cells=c):
            parts.append(kept)
        return np.concatenate(parts), st

    a, sa = run(chunk)
    b, sb = run(None)
    assert np.array_equal(a, b)
    assert (sa.n_drawn, sa.n_kept) == (sb.n_drawn, sb.n_kept)
    assert np.array_equal(sa.r_hist, sb.r_hist)


def test_sample_shell_real_space_and_redshift_space_share_the_draw(spectrum):
    box = Box(16, 160.0)
    s = shell.Shell(20.0, 50.0, 20.0)
    F = field.generate_fields(spectrum, 1.5, box, 31, psi_axes="xyz")

    def run(rsd):
        out = []
        for kept, st in shell.sample_shell(F, s, None, 5e-3, 32, f=0.8, rsd=rsd):
            out.append(kept)
        return np.concatenate(out), st

    real, s_real = run(False)
    red, s_red = run(True)
    assert s_real.n_drawn == s_red.n_drawn  # same Poisson draw and placement
    assert np.all(np.linalg.norm(real, axis=1) <= 50.0)
    assert not np.array_equal(real.shape, red.shape) or not np.array_equal(real, red)
    with pytest.raises(ValueError, match="displacement"):
        G = field.generate_fields(spectrum, 1.5, box, 31, psi_axes="z")
        list(shell.sample_shell(G, s, None, 5e-3, 32, f=0.8, rsd=True))


def test_pk_tsv_fixture_spectrum_is_usable(spectrum):
    assert isinstance(spectrum, PowerSpectrum) and spectrum(0.1) > 0
