"""Poisson sampling of the lognormal grid field into a galaxy catalog (numpy).

Per cell: `lambda = nbar V_cell (1 + delta_g)` (>= 0 by construction of the field; a
negative cell is a bug and is reported, never silently clipped), counts
`rng.poisson(lambda)`, positions `cell corner + (u_1 + ... + u_p - (p-1)/2) dx` per
axis (order-`p` jitter; `p = 1` is uniform within the cell), plane-parallel RSD
`s_z = z + f Psi_z(cell)` with the galaxy's OWN cell displacement, then periodic wrap.

RNG scheme (one scheme for every sampler in the package): the work unit is one x-slab
(`i` fixed, `N^2` cells) and each slab has its own counter-based stream,
`slab_rng(draw_seed, i)` = `Philox(key = draw_seed | i << 64)`. Within a slab the order
is the Poisson draw over the slab's cells, then the placement uniforms galaxy-major.
A catalog therefore depends on `(draw_seed, field)` alone: not on how many slabs are
processed at once, in which order, or by how many threads. Empty slabs consume no
random numbers, and a slab's draws do not move when another slab's window changes.
"""

from dataclasses import dataclass
import os

import numpy as np

from .grid import Box

RNG_SCHEME = "philox-per-x-slab"  # recorded in the catalog metadata


def default_workers() -> int:
    """Cores this process may run on (the affinity mask where the OS has one)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def slab_rng(draw_seed: int, slab: int) -> np.random.Generator:
    """The stream of x-slab `slab` for `draw_seed`: Philox with the 128-bit KEY
    `draw_seed | slab << 64`, so every (seed, slab) is a distinct stream. The key is
    the stream identifier; the counter is only a position within one stream, and two
    generators that differ in `counter` alone emit the SAME numbers shifted by a few
    blocks (the bug G10 caught on 2026-09-20)."""
    if not 0 <= int(draw_seed) < 2**64 or not 0 <= int(slab) < 2**64:
        raise ValueError("draw_seed and slab must be in [0, 2^64)")
    return np.random.Generator(np.random.Philox(key=int(draw_seed) | int(slab) << 64))


def intensity(delta_g, nbar: float, box: Box):
    """`lambda` per cell (float64, box shape) and the negative-cell diagnostics
    `(clip_fraction, clipped_mass_fraction)`, which must both be 0 for a lognormal
    field; kept as a guard against a future field that is not positive."""
    one_plus = 1.0 + np.asarray(delta_g, dtype=np.float64)
    neg = one_plus < 0.0
    clip_fraction = float(np.count_nonzero(neg) / one_plus.size)
    clipped_mass_fraction = float(-one_plus[neg].sum() / one_plus.size)
    lam = float(nbar) * box.v_cell * np.maximum(one_plus, 0.0)
    return lam, clip_fraction, clipped_mass_fraction


def poisson_counts(lam, rng: np.random.Generator):
    return rng.poisson(np.asarray(lam, dtype=np.float64))


def _decode_cells(flat_cell, n):
    """C-order flat index -> (i, j, k) int64 arrays."""
    k = flat_cell % n
    rest = flat_cell // n
    j = rest % n
    i = rest // n
    return i, j, k


def place_slab(counts_slab, i: int, box: Box, rng: np.random.Generator, jitter_p=1):
    """Positions for x-slab `i` from its per-cell `counts_slab` (`N^2` entries, any
    shape) using `rng` (the slab's stream): `(xyz, flat_cell)` with `xyz` float64 in
    `[0, L)` and `flat_cell` the C-order index into the full grid. Galaxy-major draw
    order; `jitter_p == 1` draws the uniforms directly (same numbers as the order-1
    sum, no length-1 reduction)."""
    n = box.n_mesh
    sub = np.asarray(counts_slab).reshape(-1)
    if sub.size != n * n:
        raise ValueError(f"a slab holds {n * n} cells, got {sub.size}")
    occ = np.flatnonzero(sub)
    if occ.size == 0:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    local = np.repeat(occ, sub[occ]).astype(np.int64, copy=False)
    ng = local.size
    if jitter_p == 1:
        xyz = rng.random((ng, 3), dtype=np.float64)
    else:
        u = rng.random((ng, 3, jitter_p), dtype=np.float64)
        xyz = u.sum(axis=2) - 0.5 * (jitter_p - 1)  # jitter centred on the cell
        del u
    xyz[:, 0] += i
    xyz[:, 1] += local // n
    xyz[:, 2] += local % n
    xyz *= box.dx
    if jitter_p > 1:
        np.mod(xyz, box.box_size, out=xyz)
    return xyz, local + i * n * n


def place_slabs(counts, box: Box, draw_seed: int, *, jitter_p=1):
    """Yield `(xyz, flat_cell)` per non-empty x-slab of a given `counts` grid, each
    slab placed with its own stream (the Poisson draw is not part of this call, so
    the placement uniforms are the FIRST numbers of each slab's stream)."""
    n = box.n_mesh
    counts = np.asarray(counts).reshape(-1)
    if counts.size != box.n_cells:
        raise ValueError("counts must have one entry per cell")
    for i in range(n):
        sub = counts[i * n * n : (i + 1) * n * n]
        if not sub.any():
            continue
        yield place_slab(sub, i, box, slab_rng(draw_seed, i), jitter_p)


def place(counts, box: Box, draw_seed: int, *, jitter_p=1):
    """In-memory placement: concatenation of `place_slabs`."""
    parts = list(place_slabs(counts, box, draw_seed, jitter_p=jitter_p))
    if not parts:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def draw_slab(lam, i: int, box: Box, draw_seed: int, jitter_p=1):
    """Poisson-draw and place x-slab `i` of the intensity grid `lam` on the slab's
    stream: `(xyz, flat_cell)`. The unit every sampler in the package is built from."""
    rng = slab_rng(draw_seed, i)
    counts = rng.poisson(np.asarray(lam[i], dtype=np.float64))
    return place_slab(counts, i, box, rng, jitter_p)


def draw_slabs(lam, box: Box, draw_seed: int, *, jitter_p=1):
    """`draw_slab` over every x-slab, in order, skipping slabs with no galaxies."""
    for i in range(box.n_mesh):
        xyz, flat_cell = draw_slab(lam, i, box, draw_seed, jitter_p)
        if xyz.shape[0]:
            yield xyz, flat_cell


def rsd_plane_parallel(xyz, flat_cell, psi_z_flat, f: float, box: Box):
    """`z -> z + f Psi_z(cell)` in place, then wrap to [0, L)."""
    xyz[:, 2] += float(f) * psi_z_flat[flat_cell]
    np.mod(xyz[:, 2], box.box_size, out=xyz[:, 2])
    return xyz


@dataclass
class Catalog:
    box: Box
    xyz: np.ndarray  # (n, 3) float64, [0, L)
    cell: np.ndarray  # (n,) int64 C-order flat cell index
    nbar_target: float
    lam_total: float  # sum of lambda over the box = expected count
    n_galaxies: int
    clip_fraction: float
    clipped_mass_fraction: float
    ic_seed: int
    draw_seed: int
    f: float
    rsd: bool

    @property
    def realized_nbar(self) -> float:
        return self.n_galaxies / self.box.volume

    @property
    def expected_nbar(self) -> float:
        return self.lam_total / self.box.volume


def sample_catalog(
    fields, nbar: float, draw_seed: int, *, f: float = 0.0, rsd: bool = True
) -> Catalog:
    """Draw one periodic-box catalog from `fields` (a `field.Fields`), serially over
    slabs (this sampler serves the gates; the shell product is the parallel one).
    `rsd` needs `fields.psi_z`."""
    box = fields.box
    lam, clip_fraction, clipped_mass = intensity(fields.delta_g, nbar, box)
    lam_total = float(lam.sum())
    if rsd:
        if fields.psi_z is None:
            raise ValueError("rsd=True but the fields carry no displacement")
        psi_flat = np.asarray(fields.psi_z, dtype=np.float64).reshape(-1)
    parts = []
    cells = []
    for xyz, flat_cell in draw_slabs(lam, box, draw_seed, jitter_p=fields.jitter_p):
        if rsd:
            rsd_plane_parallel(xyz, flat_cell, psi_flat, f, box)
        parts.append(xyz)
        cells.append(flat_cell)
    del lam
    xyz = np.concatenate(parts) if parts else np.empty((0, 3))
    cell = np.concatenate(cells) if cells else np.empty(0, dtype=np.int64)
    return Catalog(
        box=box,
        xyz=xyz,
        cell=cell,
        nbar_target=float(nbar),
        lam_total=lam_total,
        n_galaxies=int(xyz.shape[0]),
        clip_fraction=clip_fraction,
        clipped_mass_fraction=clipped_mass,
        ic_seed=fields.ic_seed,
        draw_seed=int(draw_seed),
        f=float(f),
        rsd=bool(rsd),
    )


def split_seed(seed: int):
    """One user seed -> `(ic_seed, draw_seed)` via `SeedSequence.spawn`, so the field
    and the draw are independent streams and a field can be resampled."""
    children = np.random.SeedSequence(int(seed)).spawn(2)
    return int(children[0].generate_state(1)[0]), int(children[1].generate_state(1)[0])
