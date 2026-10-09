"""Shell product contracts: geometry identities, mask and selection identities, radial
RSD identities and its plane-parallel limit, the exact Poisson check on a uniform
field, the radial-buffer requirement, the angular pre-cut, streaming bit-equality."""

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
    # outward Psi carries a galaxy just inside rmax out; inward brings one back in
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
    # P_in x 1e-4: delta uniform to 1e-2, uniform-in-cell placement, equal-area
    # pixels, so N_kept ~ Poisson(nbar fsky V_shell) exactly and the window's draws
    # ~ Poisson(sum lambda).
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
    assert np.all(shell.select(xyz, s, mask))
    assert stats.r_hist.sum() == stats.n_kept
    # a galaxy that left the box sits beyond rmax + buffer: never among the kept
    assert np.all(np.abs(xyz).max(axis=1) < 0.5 * box.box_size)
    # the radial profile is flat: sub-shell counts vs their volumes
    v_sub = 4 / 3 * np.pi * np.diff(stats.r_edges**3) * mask.fsky
    zs = (stats.r_hist - nbar * v_sub) / np.sqrt(nbar * v_sub)
    assert np.all(np.abs(zs) < 4.0), zs


@pytest.mark.parametrize("workers", [3, 8, None])
def test_sample_shell_is_the_same_for_any_thread_count(spectrum, mask, workers):
    box = Box(16, 160.0)
    s = shell.Shell(20.0, 40.0, 35.0)  # buffer above the field-time bound (~25)
    F = field.generate_fields(spectrum, 1.5, box, 21, psi_axes="xyz")

    def run(w):
        parts = []
        for kept, st in shell.sample_shell(F, s, mask, 5e-3, 22, f=0.8, n_workers=w):
            parts.append(kept)
        return np.concatenate(parts), st

    a, sa = run(workers)
    b, sb = run(1)  # inline, no executor
    assert np.array_equal(a, b)
    assert (sa.n_drawn, sa.n_kept, sa.n_left_box) == (
        sb.n_drawn,
        sb.n_kept,
        sb.n_left_box,
    )
    assert np.array_equal(sa.r_hist, sb.r_hist)
    assert sa.n_kept > 0


def _guard_fixture(spectrum):
    box = Box(16, 160.0)  # dx 10, half diagonal 8.66; observer-centred, L/2 = 80
    F = field.generate_fields(spectrum, 1.5, box, 41, psi_axes="xyz")
    return box, F, 0.8


def _centres(box):
    c1 = shell.cell_centres_1d(box)
    return np.stack(np.meshgrid(c1, c1, c1, indexing="ij"), -1).reshape(-1, 3)


def _required_full_grid(box, s, f, psi):
    """`Shell.required_buffer` (the line-of-sight interval test) written
    independently, on the full grid."""
    X = _centres(box)
    r = np.linalg.norm(X, axis=1)
    h = 0.5 * np.sqrt(3.0) * box.dx
    proj = f * np.einsum("ij,ij->i", psi, X) / r
    m = abs(f) * np.linalg.norm(psi, axis=1)
    theta = np.arcsin(np.clip(h / r, 0.0, 1.0))
    chi = np.where(r > h, 2.0 * np.sin(theta / 2.0), 2.0)
    rho = np.stack(
        [
            np.maximum(r - h, 0.0) + np.clip(proj - m * chi, -m, m),
            r + h + np.clip(proj + m * chi, -m, m),
        ]
    )
    s_hi = np.abs(rho).max(0)
    s_lo = np.where(rho[0] * rho[1] <= 0.0, 0.0, np.abs(rho).min(0))
    reach = (s_hi >= s.rmin) & (s_lo <= s.rmax)
    d = np.concatenate(
        [r[reach & (r > s.rmax)] - s.rmax, s.rmin - r[reach & (r < s.rmin)], [0.0]]
    )
    return float(d.max())


def _required_magnitude_bound(box, s, f, psi):
    """The looser magnitude bound: radial shift bounded by `f|Psi_c|`, no direction."""
    r = shell.cell_radius(box).reshape(-1)
    reach = 0.5 * np.sqrt(3.0) * box.dx + f * np.sqrt((psi**2).sum(1))
    outer = (r > s.rmax) & (r - reach <= s.rmax)
    inner = (r < s.rmin) & (r + reach >= s.rmin)
    d = np.concatenate([r[outer] - s.rmax, s.rmin - r[inner], [0.0]])
    return float(d.max())


def _landing_radii(box, cell_xyz, psi_c, f, offsets):
    """`|s|` under radial RSD for points `cell_xyz + offsets` of one cell."""
    x = cell_xyz + offsets
    rx = np.linalg.norm(x, axis=1)
    return np.abs(rx + f * (x @ psi_c) / rx)


def _cell_offsets(box, rng, n_interior=64):
    corners = np.array(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij")).reshape(3, -1).T
    # corners pulled a hair inside: one cell corner is the observer, where r = 0
    return (
        np.concatenate([corners * (1 - 1e-9), rng.uniform(-0.5, 0.5, (n_interior, 3))])
        * box.dx
    )


def _plant(F, cell, value):
    big = np.asarray(F.psi["x"]).copy()
    big[np.unravel_index(cell, F.box.shape)] = value
    F.psi["x"] = big


def test_required_buffer_is_the_exact_feeding_condition(spectrum):
    box, F, f = _guard_fixture(spectrum)
    s = shell.Shell(20.0, 40.0, 35.0)
    psi = F.psi_flat("xyz")
    need = s.required_buffer(box, f, psi)
    assert need == pytest.approx(_required_full_grid(box, s, f, psi), rel=1e-13)
    assert 0.5 * np.sqrt(3.0) * box.dx < need < 35.0, need
    st = None
    for _, st in shell.sample_shell(F, s, None, 5e-3, 42, f=f, n_workers=1):
        pass
    W = shell.cell_window(box, s).reshape(-1)
    assert st.psi_max == np.sqrt((psi[W] ** 2).sum(1)).max()
    assert st.required_buffer == need
    # without RSD only the cell extent matters: at most the half diagonal
    nr = s.required_buffer(box, f, None)
    assert nr == pytest.approx(_required_full_grid(box, s, 0.0, 0.0 * psi), rel=1e-13)
    assert 0.0 < nr <= 0.5 * np.sqrt(3.0) * box.dx
    for _, st in shell.sample_shell(F, s, None, 5e-3, 42, f=f, rsd=False, n_workers=1):
        pass
    assert st.psi_max == 0.0 and st.required_buffer == nr


def test_no_undrawn_cell_reaches_the_shell_at_the_required_buffer(spectrum):
    # Plant radial displacements in undrawn cells both sides of the shell, set the
    # buffer to exactly the requirement, and move the corners and interior samples of
    # EVERY undrawn cell by the radial RSD map: none may land in the shell.
    box, F, f = _guard_fixture(spectrum)
    base = shell.Shell(40.0, 52.0, 18.0)  # r_lo 22, r_hi 70: undrawn cells both sides
    r = shell.cell_radius(box).reshape(-1)
    c1 = shell.cell_centres_1d(box)
    # only the planted cells move, so each side's requirement is theirs alone (the
    # fixture field's own Psi needs 25.9 outside, which would hide the inner side)
    psi3 = {a: np.zeros(box.shape) for a in "xyz"}
    # inner r 16.6 reaches 40 at f|Psi| >= 14.8; outer r 71.2 reaches 52 at >= 10.6
    inner = np.flatnonzero((r > 16.0) & (r < 17.0))[:4]
    outer = np.flatnonzero((r > 71.0) & (r < 72.0))[:4]
    assert inner.size == 4 and outer.size == 4  # the fixture plants on both sides
    assert base.required_buffer(box, f, None) < 40.0 - 16.6  # the inner side binds
    for cells, v in ((inner, 28.0), (outer, -24.0)):
        for cell in cells:
            ijk = np.unravel_index(cell, box.shape)
            x = np.array([c1[q] for q in ijk])
            for a, comp in zip("xyz", v * x / np.linalg.norm(x)):
                psi3[a][ijk] = comp
    F.psi.update(psi3)
    psi = F.psi_flat("xyz")
    need = base.required_buffer(box, f, psi)
    s = shell.Shell(base.rmin, base.rmax, need)
    s.check_box(box)
    undrawn = np.flatnonzero(~shell.cell_window(box, s).reshape(-1))
    assert set(inner) <= set(np.flatnonzero(shell.cell_window(box, s).reshape(-1)))
    offsets = _cell_offsets(box, np.random.default_rng(0))
    idx = np.array(np.unravel_index(undrawn, box.shape)).T
    for cell, (i, j, k) in zip(undrawn, idx):
        sx = _landing_radii(box, np.array([c1[i], c1[j], c1[k]]), psi[cell], f, offsets)
        assert not np.any((sx >= s.rmin) & (sx <= s.rmax)), (cell, sx.min(), sx.max())
    # the guard holds exactly at the requirement and raises just below it
    list(shell.sample_shell(F, s, None, 5e-3, 42, f=f, n_workers=1))
    short = shell.Shell(base.rmin, base.rmax, need * (1.0 - 1e-9))
    with pytest.raises(ValueError, match="this field needs"):
        list(shell.sample_shell(F, short, None, 5e-3, 42, f=f, n_workers=1))


def test_line_of_sight_guard_is_complete_for_any_direction(spectrum):
    # Random-direction displacements in undrawn cells both sides: at buffer =
    # requirement nothing lands in the shell, and it never exceeds the f|Psi_c| bound.
    box, _, f = _guard_fixture(spectrum)
    base = shell.Shell(40.0, 52.0, 18.0)
    X = _centres(box)
    r = np.linalg.norm(X, axis=1)
    rng = np.random.default_rng(5)
    band = np.flatnonzero((r < 30.0) | ((r > 60.0) & (r < 100.0)))
    v = rng.standard_normal((band.size, 3))
    v *= rng.uniform(0.0, 30.0, (band.size, 1)) / np.linalg.norm(v, axis=1)[:, None]
    psi = np.zeros((r.size, 3))
    psi[band] = v
    need = base.required_buffer(box, f, psi)
    assert need == pytest.approx(_required_full_grid(box, base, f, psi), rel=1e-13)
    old = _required_magnitude_bound(box, base, f, psi)
    assert need <= old
    s = shell.Shell(base.rmin, base.rmax, need)
    undrawn = np.flatnonzero(~shell.cell_window(box, s).reshape(-1))
    assert np.intersect1d(undrawn, band).size > 0  # some plants stay undrawn
    offsets = _cell_offsets(box, np.random.default_rng(6))
    for cell in undrawn:
        sx = _landing_radii(box, X[cell], psi[cell], f, offsets)
        assert not np.any((sx >= s.rmin) & (sx <= s.rmax)), (cell, sx.min(), sx.max())


def test_line_of_sight_guard_counts_a_tangential_displacement_through_the_chord():
    # A displacement perpendicular to the centre's line of sight is radial only at
    # off-centre points. A point of the cell lands inside while r_c - h is beyond
    # rmax: only the chord term can admit the cell.
    box = Box(16, 160.0)
    f = 0.8
    X = _centres(box)
    cell = int(np.flatnonzero(np.all(X == [65.0, 5.0, 5.0], axis=1))[0])
    c = X[cell]
    h = 0.5 * np.sqrt(3.0) * box.dx
    psi_c = np.cross(c, [0.0, 0.0, 1.0])
    psi_c *= 100.0 / np.linalg.norm(psi_c)
    assert abs(psi_c @ c) < 1e-9
    offsets = _cell_offsets(box, np.random.default_rng(7), n_interior=4096)
    rho_min = _landing_radii(box, c, psi_c, f, offsets).min()
    r_c = np.linalg.norm(c)
    assert rho_min < r_c - h  # the tangential shift reaches further in than r_c - h
    rmax = 0.5 * (rho_min + r_c - h)
    s = shell.Shell(rmax - 20.0, rmax, 0.0)
    psi = np.zeros((X.shape[0], 3))
    psi[cell] = psi_c
    need = s.required_buffer(box, f, psi)
    assert need == pytest.approx(r_c - rmax, rel=1e-12)
    assert need == pytest.approx(_required_full_grid(box, s, f, psi), rel=1e-13)


def test_an_outward_displacement_beyond_the_window_does_not_count(spectrum):
    box, F, f = _guard_fixture(spectrum)
    s = shell.Shell(20.0, 40.0, 35.0)
    X = _centres(box)
    r = np.linalg.norm(X, axis=1)
    need0 = s.required_buffer(box, f, F.psi_flat("xyz"))
    outside = np.flatnonzero((r > s.r_hi) & (r < s.r_hi + 5.0))[0]
    ijk = np.unravel_index(outside, box.shape)
    for sign, raises in ((+1.0, False), (-1.0, True)):
        # 60 inward from r 79 lands in the shell; 100 would overshoot the observer
        for a, comp in zip("xyz", sign * 60.0 * X[outside] / r[outside]):
            F.psi[a] = np.asarray(F.psi[a]).copy()
            F.psi[a][ijk] = comp
        need = s.required_buffer(box, f, F.psi_flat("xyz"))
        if raises:
            assert need > s.buffer
            with pytest.raises(ValueError, match="this field needs"):
                list(shell.sample_shell(F, s, None, 5e-3, 42, f=f, n_workers=1))
        else:
            assert need == need0
            list(shell.sample_shell(F, s, None, 5e-3, 42, f=f, n_workers=1))


def test_a_large_displacement_counts_only_where_it_can_feed_the_shell(spectrum):
    box, F, f = _guard_fixture(spectrum)
    s = shell.Shell(20.0, 40.0, 35.0)
    r = shell.cell_radius(box).reshape(-1)
    need0 = s.required_buffer(box, f, F.psi_flat("xyz"))
    # inside the shell, far from both edges: drawn, so it cannot change the need
    inside = np.flatnonzero((r > 28.0) & (r < 32.0))[0]
    _plant(F, inside, 60.0)
    assert s.required_buffer(box, f, F.psi_flat("xyz")) == need0
    list(shell.sample_shell(F, s, None, 5e-3, 42, f=f, n_workers=1))
    # the same displacement in an undrawn cell beyond the window (centre
    # (-75, -25, -5): +x is inward, f Psi . c_hat = -45, landing near r 34): it raises
    _plant(F, inside, 0.0)
    outside = np.flatnonzero((r > s.r_hi) & (r < s.r_hi + 5.0))[0]
    _plant(F, outside, 60.0)
    with pytest.raises(ValueError, match="radial buffer 35 Mpc/h is below the"):
        list(shell.sample_shell(F, s, None, 5e-3, 42, f=f, n_workers=1))


def test_mask_dilation_is_the_centre_distance_set():
    import healpy as hp

    nside = 16
    for seed_pix in (0, 1000, 2500):
        v = np.zeros(hp.nside2npix(nside), bool)
        v[seed_pix] = True
        m = shell.AngularMask(v, nested=True)
        for radius in (0.05, 0.2, 0.6):
            got = m.dilated(radius).values
            centre = np.array(hp.pix2vec(nside, seed_pix, nest=True))
            allv = np.column_stack(hp.pix2vec(nside, np.arange(v.size), nest=True))
            ang = np.arccos(np.clip(allv @ centre, -1.0, 1.0))
            assert np.array_equal(got, ang <= radius + 1e-12)
    m = shell.AngularMask(np.zeros(12, bool), nested=False)
    assert not m.dilated(0.3).values.any()
    assert m.dilated(np.pi).values.all()
    full = shell.AngularMask(np.ones(12, bool), nested=False)
    assert full.dilated(0.1).values.all()


def _sparse_mask(nside=16):
    import healpy as hp

    v = np.zeros(hp.nside2npix(nside), bool)
    v[np.arange(0, v.size, 97)] = True  # isolated pixels all over the sky
    v[300:320] = True  # and one small patch
    return shell.AngularMask(v, nested=True)


@pytest.mark.parametrize("which", ["band", "sparse"])
def test_angular_precut_is_an_exact_superset_of_the_feeding_cells(
    spectrum, mask, which
):
    # Every galaxy kept by the radial + mask selection (drawn from the FULL radial
    # window, RSD applied) comes from a cell the pre-cut keeps.
    m = mask if which == "band" else _sparse_mask()
    box = Box(16, 160.0)
    s = shell.Shell(20.0, 40.0, 35.0)
    W = shell.cell_window(box, s)
    P = shell.angular_precut(box, s, m)
    assert not np.any(P & ~W)  # a subset of the radial window
    assert P.sum() < W.sum()
    assert np.array_equal(shell.angular_precut(box, s, None), W)
    f = 0.8
    for seed in (51, 52, 53, 54):
        F = field.generate_fields(spectrum, 1.5, box, seed, psi_axes="xyz")
        lam, _, _ = sample.intensity(F.delta_g, 2e-2, box)
        lam *= W
        psi = F.psi_flat("xyz")
        fed = np.zeros(box.n_cells, bool)
        n_kept = 0
        for xyz, cell in sample.draw_slabs(lam, box, 100 + seed):
            xyz -= 0.5 * box.box_size
            shell.rsd_radial(xyz, cell, psi, f)
            keep = shell.select(xyz, s, m)
            fed[cell[keep]] = True
            n_kept += keep.sum()
        assert n_kept > 0
        assert np.all(P.reshape(-1)[fed]), "a kept galaxy came from a cut cell"


def test_angular_precut_leaves_uncut_slabs_bitwise(spectrum):
    # Each x-slab draws on its own stream, so the pre-cut can change a slab's galaxies
    # only if it drops one of that slab's cells. A mask excluding a cap around +x cuts
    # the high-x slabs only; every other slab must be bitwise unchanged.
    import healpy as hp

    nside = 16
    x, _, _ = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True)
    m = shell.AngularMask(x < 0.5, nested=True)
    box = Box(16, 160.0)
    s = shell.Shell(20.0, 40.0, 35.0)
    W = shell.cell_window(box, s)
    P = shell.angular_precut(box, s, m)
    F = field.generate_fields(spectrum, 1.5, box, 51, psi_axes="xyz")

    def run(pre):
        # one yielded chunk per x-slab, in slab order
        return [
            k
            for k, _ in shell.sample_shell(
                F, s, m, 2e-2, 151, f=0.8, n_workers=1, angular_precut=pre
            )
        ]

    a, b = run(True), run(False)
    assert len(a) == len(b) == box.n_mesh
    cut = (W & ~P).reshape(box.n_mesh, -1).any(axis=1)
    assert cut.any() and not cut.all()
    assert sum(a[i].shape[0] for i in np.flatnonzero(~cut)) > 1000
    for i in np.flatnonzero(~cut):
        assert np.array_equal(a[i], b[i]), i
    assert any(not np.array_equal(a[i], b[i]) for i in np.flatnonzero(cut))


def test_radial_histogram_matches_numpy_including_edges():
    edges = np.linspace(20.0, 50.0, 9)
    rng = np.random.default_rng(4)
    r = np.concatenate(
        [
            rng.uniform(20.0, 50.0, 100_000),
            edges,  # every edge exactly, including both ends
            np.nextafter(edges[1:-1], 0.0),  # just below each interior edge
            np.nextafter(edges[1:-1], np.inf),  # just above
            20.0 + 30.0 * np.arange(1, 8) / 8,  # the edges as a different expression
        ]
    )
    assert np.array_equal(shell.radial_histogram(r, edges), np.histogram(r, edges)[0])
    assert shell.radial_histogram(np.empty(0), edges).sum() == 0


def test_sample_shell_real_space_and_redshift_space_share_the_draw(spectrum):
    box = Box(16, 160.0)
    s = shell.Shell(20.0, 40.0, 35.0)
    F = field.generate_fields(spectrum, 1.5, box, 31, psi_axes="xyz")

    def run(rsd):
        out = []
        for kept, st in shell.sample_shell(F, s, None, 5e-3, 32, f=0.8, rsd=rsd):
            out.append(kept)
        return np.concatenate(out), st

    real, s_real = run(False)
    red, s_red = run(True)
    assert s_real.n_drawn == s_red.n_drawn  # same Poisson draw and placement
    assert np.all(np.linalg.norm(real, axis=1) <= 40.0)
    assert not np.array_equal(real.shape, red.shape) or not np.array_equal(real, red)
    with pytest.raises(ValueError, match="displacement"):
        G = field.generate_fields(spectrum, 1.5, box, 31, psi_axes="z")
        list(shell.sample_shell(G, s, None, 5e-3, 32, f=0.8, rsd=True))


def test_pk_tsv_fixture_spectrum_is_usable(spectrum):
    assert isinstance(spectrum, PowerSpectrum) and spectrum(0.1) > 0
