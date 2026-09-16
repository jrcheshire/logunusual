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
    n_window_cells: int
    lam_window: float  # sum of lambda over the window = expected draws
    n_drawn: int = 0
    n_kept: int = 0
    n_left_box: int = 0  # shifted outside [-L/2, L/2) (diagnostic; never kept)
    r_edges: np.ndarray | None = None
    r_hist: np.ndarray | None = None  # kept galaxies per radial sub-shell
    ic_seed: int = 0
    draw_seed: int = 0
    f: float = 0.0

    def realized_nbar(self, fsky: float) -> float:
        return self.n_kept / (fsky * self.shell.volume)


def sample_shell(
    fields,
    shell: Shell,
    mask: AngularMask | None,
    nbar: float,
    draw_seed: int,
    *,
    f: float,
    rsd: bool = True,
    chunk_cells=None,
    n_radial_bins: int = 8,
):
    """Generator over x-slabs of the box: yields `(xyz_kept, stats)` with `stats` the
    running `ShellStats` (the same object each time; final after exhaustion). Positions
    are observer-centred, in redshift space if `rsd`. The draw order and RNG use are
    those of `sample.place_chunked` on the windowed intensity, so any chunking is
    bit-identical."""
    box = fields.box
    shell.check_box(box)
    lam, clip, _ = sample.intensity(fields.delta_g, nbar, box)
    if clip:
        raise ValueError("negative intensity cells: not a lognormal field")
    W = cell_window(box, shell)
    lam *= W
    rng = np.random.default_rng(draw_seed)
    counts = sample.poisson_counts(lam, rng)
    stats = ShellStats(
        shell=shell,
        nbar_target=float(nbar),
        n_window_cells=int(np.count_nonzero(W)),
        lam_window=float(lam.sum()),
        r_edges=np.linspace(shell.rmin, shell.rmax, n_radial_bins + 1),
        r_hist=np.zeros(n_radial_bins, dtype=np.int64),
        ic_seed=int(fields.ic_seed),
        draw_seed=int(draw_seed),
        f=float(f),
    )
    del lam, W
    psi = fields.psi_flat("xyz") if rsd else None
    half = 0.5 * box.box_size
    for xyz, flat_cell in sample.place_chunked(
        counts, box, rng, jitter_p=fields.jitter_p, chunk_cells=chunk_cells
    ):
        stats.n_drawn += xyz.shape[0]
        xyz -= half
        if rsd:
            rsd_radial(xyz, flat_cell, psi, f)
            stats.n_left_box += int(np.count_nonzero(np.abs(xyz).max(axis=1) >= half))
        keep = select(xyz, shell, mask)
        kept = xyz[keep]
        stats.n_kept += kept.shape[0]
        r = np.sqrt(np.einsum("ij,ij->i", kept, kept))
        stats.r_hist += np.histogram(r, bins=stats.r_edges)[0]
        yield kept, stats
