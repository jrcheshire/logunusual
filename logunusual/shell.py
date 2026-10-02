"""Observer-centred shell product: geometry, cell-level radial window, angular mask,
radial RSD, and the streamed shell sampler (numpy).

Frame: the periodic box is centred on the observer, so a position from
`sample.place_chunked` (in `[0, L)`) is shifted by `-L/2` per axis and cell centres sit
at `(i + 0.5) dx - L/2`. Galaxies are drawn only in cells whose centre lies in the
buffered shell `[max(0, rmin - buffer), rmax + buffer]` (full sky; the window
multiplies the intensity at cell level, the lognormal field itself stays periodic and
unwindowed), displaced radially by their OWN cell's displacement,
`s = x + f (Psi . x / r^2) x`, then kept if `rmin <= |s| <= rmax` (inclusive) and, with
a mask, if the HEALPix pixel of `s` is set. No periodic wrap after the shift: a galaxy
can only leave the box from a buffer cell touching a face, and it is outside
`[rmin, rmax]` either way. The buffer is what lets galaxies cross the shell edges in
both directions under RSD without a density deficit at the edge.
"""

from dataclasses import dataclass
from functools import cached_property
import hashlib
from pathlib import Path

import numpy as np

from . import sample
from .grid import Box


@dataclass(frozen=True)
class Shell:
    rmin: float  # Mpc/h, inclusive
    rmax: float  # Mpc/h, inclusive
    buffer: float  # Mpc/h, drawn on both sides of the shell

    def __post_init__(self):
        if not (0.0 <= self.rmin < self.rmax):
            raise ValueError(f"need 0 <= rmin < rmax, got {self.rmin}, {self.rmax}")
        if self.buffer < 0.0:
            raise ValueError(f"buffer must be >= 0, got {self.buffer}")

    @property
    def r_lo(self) -> float:
        return max(0.0, self.rmin - self.buffer)

    @property
    def r_hi(self) -> float:
        return self.rmax + self.buffer

    @property
    def volume(self) -> float:
        """Full-sky shell volume `4/3 pi (rmax^3 - rmin^3)`."""
        return 4.0 / 3.0 * np.pi * (self.rmax**3 - self.rmin**3)

    def required_buffer(self, box: Box, f: float, psi_flat=None) -> float:
        """The smallest buffer that draws every cell able to feed the shell, for this
        field. A galaxy of cell `c` sits within `h = sqrt(3)/2 dx` of the centre's
        radius `r_c` and radial RSD moves its radius by at most `f |Psi_c|`, so a cell
        beyond the shell can reach it iff `r_c - h - f|Psi_c| <= rmax`, and one inside
        iff `r_c + h + f|Psi_c| >= rmin` (a shift through the observer trips the same
        test). Returns the largest distance from the shell of any such cell, over the
        whole box; `psi_flat` None means no RSD. Checked at field time by
        `sample_shell`."""
        h = 0.5 * np.sqrt(3.0) * box.dx
        n = box.n_mesh
        need = 0.0
        for i in range(n):
            r = slab_radius(box, i).reshape(-1)
            if psi_flat is None:
                reach = h
            else:
                p = psi_flat[i * n * n : (i + 1) * n * n]
                reach = h + abs(float(f)) * np.sqrt(np.einsum("ij,ij->i", p, p))
            outer = (r > self.rmax) & (r - reach <= self.rmax)
            if outer.any():
                need = max(need, float((r[outer] - self.rmax).max()))
            if self.rmin > 0.0:
                inner = (r < self.rmin) & (r + reach >= self.rmin)
                if inner.any():
                    need = max(need, float((self.rmin - r[inner]).max()))
        return need

    def check_box(self, box: Box):
        """The buffered shell must fit inside the box centred on the observer, and
        the buffer must exceed a cell diagonal (so every cell that can feed the shell
        under RSD is drawn)."""
        if self.r_hi > 0.5 * box.box_size + 1e-9:
            raise ValueError(
                f"buffered shell radius {self.r_hi} exceeds L/2 = {0.5 * box.box_size}"
            )
        if self.buffer < np.sqrt(3.0) * box.dx:
            raise ValueError(
                f"buffer {self.buffer} is below a cell diagonal "
                f"{np.sqrt(3.0) * box.dx:.3f}"
            )


def cell_centres_1d(box: Box):
    """Observer-centred cell-centre coordinates along one axis."""
    return (np.arange(box.n_mesh) + 0.5) * box.dx - 0.5 * box.box_size


def cell_radius(box: Box):
    """`|x_centre|` on the (N, N, N) grid, observer at the box centre."""
    c = cell_centres_1d(box)
    r2 = c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2
    return np.sqrt(r2)


def cell_window(box: Box, shell: Shell):
    """Boolean (N, N, N): cell centre within the buffered shell (full sky)."""
    r = cell_radius(box)
    return (r >= shell.r_lo) & (r <= shell.r_hi)


def cell_angular_radius(box: Box, r_centre):
    """Largest angle between a cell's centre direction and any point of the cell:
    `asin(min(1, (sqrt(3)/2 dx) / r_centre))`; `pi` for a cell containing or nearer
    than half a diagonal to the observer."""
    half = 0.5 * np.sqrt(3.0) * box.dx
    r = np.asarray(r_centre, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r > 0.0, half / r, np.inf)
    return np.where(ratio >= 1.0, np.pi, np.arcsin(np.minimum(ratio, 1.0)))


def slab_radius(box: Box, i: int):
    """`|x_centre|` of x-slab `i`, shape (N, N)."""
    c = cell_centres_1d(box)
    return np.sqrt(c[i] ** 2 + c[:, None] ** 2 + c[None, :] ** 2)


def slab_window(box: Box, shell: Shell, mask, i: int, angular: bool = True):
    """Boolean (N, N): the cells of x-slab `i` to draw. Radial: centre within the
    buffered shell. Angular (with a mask and `angular`): radial RSD keeps a galaxy's
    direction, so a galaxy of cell `c` stays within `cell_angular_radius(r_c)` of the
    cell-centre direction and its pixel's centre within a further pixel radius of
    that; the cell is dropped iff the nearest set-pixel centre is farther than
    `cell_angular_radius(r_c) + 2 max_pixrad` from its centre's pixel centre
    (`AngularMask.distance_to_set`). An exact superset of the cells that feed the
    masked shell; `tests/test_shell.py` checks it galaxy by galaxy. Returns
    `(window, n_radial)` with `n_radial` the radial-only cell count."""
    import healpy as hp

    r = slab_radius(box, i)
    W = (r >= shell.r_lo) & (r <= shell.r_hi)
    n_radial = int(np.count_nonzero(W))
    if mask is None or not angular or n_radial == 0:
        return W, n_radial
    idx = np.flatnonzero(W)
    n = box.n_mesh
    c = cell_centres_1d(box)
    pix = hp.vec2pix(
        mask.nside, np.full(idx.size, c[i]), c[idx // n], c[idx % n], nest=mask.nested
    )
    theta = cell_angular_radius(box, r.reshape(-1)[idx]) + 2.0 * hp.max_pixrad(
        mask.nside
    )
    W.reshape(-1)[idx] = mask.distance_to_set[pix] <= theta
    return W, n_radial


def angular_precut(box: Box, shell: Shell, mask):
    """Boolean (N, N, N): `slab_window` stacked over slabs (the drawn cells with a
    mask; `mask=None` gives the radial window)."""
    return np.stack(
        [slab_window(box, shell, mask, i)[0] for i in range(box.n_mesh)], axis=0
    )


_precut = slab_window  # `sample_shell` takes a flag named `angular_precut`


@dataclass(frozen=True)
class AngularMask:
    """A HEALPix map of kept pixels (bool), any NSIDE, NESTED or RING."""

    values: np.ndarray
    nested: bool
    sha256: str | None = None
    path: str | None = None
    dataset: str | None = None

    def __post_init__(self):
        import healpy as hp

        v = np.asarray(self.values).astype(bool)
        object.__setattr__(self, "values", v)
        hp.npix2nside(v.size)  # raises if not a HEALPix map

    @property
    def nside(self) -> int:
        import healpy as hp

        return int(hp.npix2nside(self.values.size))

    @property
    def fsky(self) -> float:
        return float(self.values.mean())

    @classmethod
    def from_h5(cls, path, dataset="MASK") -> "AngularMask":
        """HDF5 with root attributes `PIXTYPE = HEALPIX` and `ORDERING` in
        {`NESTED`, `RING`} and an integer-valued dataset."""
        import h5py

        path = Path(path)
        with h5py.File(path, "r") as f:
            attrs = {
                k: (v.decode() if isinstance(v, bytes) else str(v))
                for k, v in f.attrs.items()
            }
            if attrs.get("PIXTYPE", "HEALPIX") != "HEALPIX":
                raise ValueError(f"{path}: PIXTYPE {attrs.get('PIXTYPE')!r}")
            ordering = attrs.get("ORDERING")
            if ordering not in ("NESTED", "RING"):
                raise ValueError(
                    f"{path}: ORDERING must be NESTED or RING: {ordering!r}"
                )
            values = np.asarray(f[dataset][...])
        if not np.all(np.isin(values, (0, 1))):
            raise ValueError(f"{path}:{dataset} is not a 0/1 map")
        return cls(
            values=values,
            nested=(ordering == "NESTED"),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            path=str(path),
            dataset=dataset,
        )

    @cached_property
    def distance_to_set(self) -> np.ndarray:
        """Per pixel, the angle (radians) from its centre to the nearest SET pixel's
        centre (0 on set pixels; `pi` everywhere if nothing is set). One KD-tree on
        the unit vectors; `dilated(theta)` is `distance_to_set <= theta`."""
        import healpy as hp
        from scipy.spatial import cKDTree

        d = np.zeros(self.values.size)
        on = np.flatnonzero(self.values)
        off = np.flatnonzero(~self.values)
        if on.size == 0:
            d[:] = np.pi
            return d
        if off.size:
            vec_on = np.column_stack(hp.pix2vec(self.nside, on, nest=self.nested))
            vec_off = np.column_stack(hp.pix2vec(self.nside, off, nest=self.nested))
            chord, _ = cKDTree(vec_on).query(vec_off, k=1)
            d[off] = 2.0 * np.arcsin(np.minimum(0.5 * chord, 1.0))
        return d

    def dilated(self, radius: float) -> "AngularMask":
        """The mask grown by `radius` (radians) on pixel CENTRES: set iff the centre
        is within `radius` of a set pixel's centre."""
        return AngularMask(self.distance_to_set <= radius + 1e-12, self.nested)

    def contains(self, xyz):
        """Boolean per row of `xyz` (observer-centred): the pixel of the direction is
        set. Rows at the origin are assigned to whatever pixel healpy returns for the
        zero vector; callers cut on `r` first."""
        import healpy as hp

        pix = hp.vec2pix(self.nside, xyz[:, 0], xyz[:, 1], xyz[:, 2], nest=self.nested)
        return self.values[pix]


def select(xyz, shell: Shell, mask: AngularMask | None):
    """Keep mask: `rmin <= r <= rmax` (inclusive) and, with a mask, pixel set."""
    r = np.sqrt(np.einsum("ij,ij->i", xyz, xyz))
    keep = (r >= shell.rmin) & (r <= shell.rmax)
    if mask is not None:
        keep &= mask.contains(xyz)
    return keep


def rsd_radial(xyz, flat_cell, psi_xyz, f: float):
    """`s = x + f (Psi_cell . x / r^2) x` in place, observer at the origin; a galaxy
    at `r = 0` is left in place. `psi_xyz` is `(n_cells, 3)` (x, y, z order)."""
    v = psi_xyz[flat_cell]
    r2 = np.einsum("ij,ij->i", xyz, xyz)
    dot = np.einsum("ij,ij->i", v, xyz)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(r2 > 0.0, float(f) * dot / r2, 0.0)
    xyz += u[:, None] * xyz
    return xyz


@dataclass
class ShellStats:
    """Per-bin bookkeeping accumulated by `sample_shell`."""

    shell: Shell
    nbar_target: float
    n_window_cells: int  # cells drawn (radial window, cut by the angular pre-cut)
    lam_window: float  # sum of lambda over the drawn cells = expected draws
    n_window_cells_radial: int = 0  # cells of the radial window alone
    n_drawn: int = 0
    n_kept: int = 0
    n_left_box: int = 0  # shifted outside [-L/2, L/2) (diagnostic; never kept)
    r_edges: np.ndarray | None = None
    r_hist: np.ndarray | None = None  # kept galaxies per radial sub-shell
    ic_seed: int = 0
    draw_seed: int = 0
    f: float = 0.0
    psi_max: float = 0.0  # max |Psi| over the drawn (window) cells, Mpc/h
    required_buffer: float = 0.0  # Shell.required_buffer for this field (exact), Mpc/h

    def realized_nbar(self, fsky: float) -> float:
        return self.n_kept / (fsky * self.shell.volume)


def _max_psi_in_window(psi_flat, box: Box, shell: Shell) -> float:
    """`max |Psi|` over the cells of the radial window, slab by slab (no full-grid
    temporary)."""
    n = box.n_mesh
    best = 0.0
    for i in range(n):
        w = slab_window(box, shell, None, i)[0].reshape(-1)
        if not w.any():
            continue
        p = psi_flat[i * n * n : (i + 1) * n * n][w]
        best = max(best, float(np.sqrt(np.einsum("ij,ij->i", p, p).max())))
    return best


def radial_histogram(r, edges):
    """Counts of `r` per bin of the UNIFORM `edges` (as `np.histogram(r, edges)`,
    including its edge corrections; values at `edges[-1]` fall in the last bin) via
    `bincount`, without the sort `np.histogram` does."""
    nb = edges.size - 1
    lo, hi = edges[0], edges[-1]
    idx = ((r - lo) * (nb / (hi - lo))).astype(np.int64)
    idx = np.clip(idx, 0, nb - 1)
    # round-off against the actual edges, as np.histogram does
    idx -= r < edges[idx]
    idx += (r >= edges[idx + 1]) & (idx != nb - 1)
    return np.bincount(idx, minlength=nb).astype(np.int64)


def _ordered_map(fn, items, n_workers: int):
    """`map(fn, items)` on a thread pool, results yielded in input order with at most
    `2 * n_workers` items in flight; inline when `n_workers <= 1`."""
    if n_workers <= 1:
        for x in items:
            yield fn(x)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    items = iter(items)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        pending = deque()
        for x in items:
            pending.append(ex.submit(fn, x))
            if len(pending) >= 2 * n_workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def sample_shell(
    fields,
    shell: Shell,
    mask: AngularMask | None,
    nbar: float,
    draw_seed: int,
    *,
    f: float,
    rsd: bool = True,
    n_workers: int | None = None,
    n_radial_bins: int = 8,
    angular_precut: bool = True,
):
    """Generator over x-slabs of the box: yields `(xyz_kept, stats)` with `stats` the
    running `ShellStats` (the same object each time; final after exhaustion). Positions
    are observer-centred, in redshift space if `rsd`. Each slab is drawn on its own
    stream (`sample.slab_rng`) by `n_workers` threads (default: the core count) and
    the slabs are yielded in order, so the catalog is the same for any thread count.
    Each worker builds its own slab's window (radial, and the angular pre-cut with a
    mask); `n_window_cells` / `lam_window` are complete only after exhaustion."""
    box = fields.box
    shell.check_box(box)
    lam, clip, _ = sample.intensity(fields.delta_g, nbar, box)
    if clip:
        raise ValueError("negative intensity cells: not a lognormal field")
    stats = ShellStats(
        shell=shell,
        nbar_target=float(nbar),
        n_window_cells=0,
        lam_window=0.0,
        r_edges=np.linspace(shell.rmin, shell.rmax, n_radial_bins + 1),
        r_hist=np.zeros(n_radial_bins, dtype=np.int64),
        ic_seed=int(fields.ic_seed),
        draw_seed=int(draw_seed),
        f=float(f),
    )
    psi = fields.psi_flat("xyz") if rsd else None
    stats.psi_max = _max_psi_in_window(psi, box, shell) if rsd else 0.0
    stats.required_buffer = shell.required_buffer(box, f, psi)
    if shell.buffer < stats.required_buffer:
        raise ValueError(
            f"radial buffer {shell.buffer:g} Mpc/h is below the "
            f"{stats.required_buffer:.1f} this field needs: a cell that far from the "
            f"shell [{shell.rmin:g}, {shell.rmax:g}] can displace galaxies into it "
            f"(max f|Psi| in the window {abs(f) * stats.psi_max:.1f})"
        )
    half = 0.5 * box.box_size
    jitter_p = fields.jitter_p
    edges = stats.r_edges
    if n_workers is None:
        n_workers = sample.default_workers()

    if mask is not None and angular_precut:
        mask.distance_to_set  # build once, before the threads share it

    def work(i):
        W, n_radial = _precut(box, shell, mask, i, angular_precut)
        lam_i = lam[i] * W
        win = (int(np.count_nonzero(W)), n_radial, float(lam_i.sum()))
        rng = sample.slab_rng(draw_seed, i)
        counts = rng.poisson(lam_i)
        xyz, flat_cell = sample.place_slab(counts, i, box, rng, jitter_p)
        n_drawn = xyz.shape[0]
        left = 0
        xyz -= half
        if rsd:
            rsd_radial(xyz, flat_cell, psi, f)
            left = int(np.count_nonzero(np.abs(xyz).max(axis=1) >= half))
        r = np.sqrt(np.einsum("ij,ij->i", xyz, xyz))
        keep = (r >= shell.rmin) & (r <= shell.rmax)
        if mask is not None:
            keep &= mask.contains(xyz)
        kept = xyz[keep]
        return kept, n_drawn, left, radial_histogram(r[keep], edges), win

    for kept, n_drawn, left, hist, win in _ordered_map(
        work, range(box.n_mesh), n_workers
    ):
        stats.n_window_cells += win[0]
        stats.n_window_cells_radial += win[1]
        stats.lam_window += win[2]
        stats.n_drawn += n_drawn
        stats.n_left_box += left
        stats.n_kept += kept.shape[0]
        stats.r_hist += hist
        yield kept, stats
