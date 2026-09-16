"""Poisson sampling of the lognormal grid field into a galaxy catalog (numpy).

Per cell: `lambda = nbar V_cell (1 + delta_g)` (>= 0 by construction of the field; a
negative cell is a bug and is reported, never silently clipped), counts
`rng.poisson(lambda)` in one vectorised call, positions `cell corner + (u_1 + ... +
u_p - (p-1)/2) dx` per axis (order-`p` jitter; `p = 1` is uniform within the cell),
plane-parallel RSD `s_z = z + f Psi_z(cell)` with the galaxy's OWN cell displacement,
then periodic wrap. Placement is streamed over slabs of cells; the draw order is
galaxy-major, so any chunking reproduces the in-memory result bit for bit (empty
chunks consume no random numbers). Pattern copied from disco-mocks `catalog.py`.
"""

from dataclasses import dataclass

import numpy as np

from .grid import Box


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


def place_chunked(
    counts, box: Box, rng: np.random.Generator, *, jitter_p=1, chunk_cells=None
):
    """Yield `(xyz, flat_cell)` per slab: positions float64 in [0, L) and the C-order
    flat index of each galaxy's cell. `chunk_cells` defaults to one x-slab (N^2)."""
    n = box.n_mesh
    counts = np.asarray(counts).reshape(-1)
    if counts.size != box.n_cells:
        raise ValueError("counts must have one entry per cell")
    if chunk_cells is None:
        chunk_cells = n * n
    dx = box.dx
    L = box.box_size
    offset = 0.5 * (jitter_p - 1)  # so the jitter is centred on the cell for any p
    for c0 in range(0, counts.size, chunk_cells):
        sub = counts[c0 : c0 + chunk_cells]
        occ = np.flatnonzero(sub)
        if occ.size == 0:
            continue
        flat_cell = np.repeat(occ + c0, sub[occ])
        ng = flat_cell.size
        u = rng.random((ng, 3, jitter_p), dtype=np.float64)
        xyz = u.sum(axis=2) - offset
        del u
        i, j, k = _decode_cells(flat_cell, n)
        xyz[:, 0] += i
        xyz[:, 1] += j
        xyz[:, 2] += k
        xyz *= dx
        if jitter_p > 1:
            np.mod(xyz, L, out=xyz)
        yield xyz, flat_cell


def place(counts, box: Box, rng: np.random.Generator, *, jitter_p=1, chunk_cells=None):
    """In-memory placement: concatenation of `place_chunked` (bit-identical)."""
    parts = list(
        place_chunked(counts, box, rng, jitter_p=jitter_p, chunk_cells=chunk_cells)
    )
    if not parts:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


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
    fields,
    nbar: float,
    draw_seed: int,
    *,
    f: float = 0.0,
    rsd: bool = True,
    chunk_cells=None,
) -> Catalog:
    """Draw one catalog from `fields` (a `field.Fields`). `rsd` needs `fields.psi_z`."""
    box = fields.box
    lam, clip_fraction, clipped_mass = intensity(fields.delta_g, nbar, box)
    rng = np.random.default_rng(draw_seed)
    counts = poisson_counts(lam, rng)
    lam_total = float(lam.sum())
    del lam
    if rsd:
        if fields.psi_z is None:
            raise ValueError("rsd=True but the fields carry no displacement")
        psi_flat = np.asarray(fields.psi_z, dtype=np.float64).reshape(-1)
    parts = []
    cells = []
    for xyz, flat_cell in place_chunked(
        counts, box, rng, jitter_p=fields.jitter_p, chunk_cells=chunk_cells
    ):
        if rsd:
            rsd_plane_parallel(xyz, flat_cell, psi_flat, f, box)
        parts.append(xyz)
        cells.append(flat_cell)
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
