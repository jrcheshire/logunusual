"""M5: cap the velocity source -- buffer requirement and galaxy-velocity cross power.

    pixi run python scripts/m5_cap.py [--config configs/v28_halofit.yaml]
        [--bins 1 2 3] [--realizations 0 1 ...] [--nu 5 4.5 4] [--radius 1 2]
        [--out runs/m5/cap.jsonl]

For each (realization, bin): the production field stage (`run.py`'s arguments,
bitwise), then per arm -- uncapped; plain `min(delta_m, delta_max)`; mass-conserving
(the excess above `delta_max` spread over a lattice sphere of R cells) -- Psi from the
arm's source, the line-of-sight `required_buffer` and the `f|Psi_c|` bound, what the arm
touches, and per k_f shell `P(delta_g, delta_src) / P(delta_g, delta_m)` and
`P(delta_src) / P(delta_m)`. Per bin `delta_max(nu) = exp(nu sigma_G - sigma_G^2/2) - 1`
with `sigma_G^2 = log1p(sigma2)` of the matter target. One JSON line per (realization,
bin), appended; rows already in `--out` are skipped (ROADMAP M5).
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from logunusual import field, sample, suite  # noqa: E402
from logunusual.config import RunConfig  # noqa: E402
from logunusual.pk import PowerSpectrum  # noqa: E402
from logunusual.grid import hermitian_weights, k_grid  # noqa: E402
from logunusual.shell import cell_radius, cell_window  # noqa: E402
from logunusual.validate import power_multipoles, shell_edges  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Added minus removed, over the sphere cells only: each share passes through a few
# float additions (round-off ~1e-15 relative), while a lost offset or a wrong share
# normalisation is >= 1/n_sphere of the excess (>= 3% at R = 3). 1e-12 sits between.
MASS_RESIDUAL_MAX = 1e-12
DPSI_GALAXIES = 1.0  # Mpc/h: "touched" = f|Psi| moved by more than this


def sphere_offsets(radius_cells):
    """Integer cell offsets `|d| <= R` (the lattice sphere), shape (n, 3)."""
    m = int(np.floor(radius_cells))
    g = np.arange(-m, m + 1)
    d = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return d[(d**2).sum(1) <= radius_cells**2]


def capped_source(dm, delta_max, offsets=None):
    """`(src, moved, residual)`: `min(dm, delta_max)` as a flat float64 array, plus
    (with `offsets`) each capped cell's excess spread equally over its periodic
    lattice sphere; `moved` is the total excess removed and `residual` (added -
    removed) / removed, read off the sphere cells alone."""
    n = dm.shape[0]
    flat = np.asarray(dm).reshape(-1)
    hot = np.flatnonzero(flat > delta_max)
    excess = flat[hot] - delta_max
    moved = float(excess.sum())
    src = np.minimum(flat, delta_max)
    residual = 0.0
    if offsets is not None and hot.size:
        ijk = np.stack(np.unravel_index(hot, dm.shape), 1)
        share = excess / offsets.shape[0]
        touched = []
        for d in offsets:
            cells = np.ravel_multi_index(((ijk + d) % n).T, dm.shape)
            np.add.at(src, cells, share)
            touched.append(cells)
        cells = np.unique(np.concatenate(touched))
        added = float((src[cells] - np.minimum(flat[cells], delta_max)).sum())
        residual = (added - moved) / moved
    return src, moved, residual


def psi_of(src_k, box):
    """`(n_cells, 3)` float64 Psi from a source's rfft, as `Fields.psi_flat`."""
    out = np.empty((box.n_mesh**3, 3))
    for j, a in enumerate(field.AXES):
        out[:, j] = np.asarray(field.displacement(src_k, box, a)).reshape(-1)
    return out


def psi_norm(psi):
    return np.sqrt(np.einsum("ij,ij->i", psi, psi))


def magnitude_bound(r, h, shell, f, psi):
    """The guard before the line-of-sight test (radial shift bounded by f|Psi_c|)."""
    reach = h + abs(f) * psi_norm(psi)
    outer = (r > shell.rmax) & (r - reach <= shell.rmax)
    inner = (r < shell.rmin) & (r + reach >= shell.rmin)
    d = np.concatenate([r[outer] - shell.rmax, shell.rmin - r[inner], [0.0]])
    return float(d.max())


class ShellP0:
    """`power_multipoles(..., ells=(0,))["P0"]` with the shell binning computed once
    (checked against it on every field)."""

    def __init__(self, box):
        _, _, k_mag = k_grid(box)
        edges = shell_edges(box)
        idx = np.digitize(k_mag, edges) - 1
        self.nb = edges.size - 1
        self.idx = np.where((idx >= 0) & (idx < self.nb), idx, self.nb)
        self.idx = self.idx.astype(np.int32).ravel()
        self.herm = hermitian_weights(box)  # (1, 1, N//2 + 1), broadcast per use
        w = np.broadcast_to(self.herm, k_mag.shape).ravel()
        self.wsum = np.bincount(self.idx, weights=w, minlength=self.nb + 1)
        self.norm = box.volume / box.n_mesh**6

    def __call__(self, a_k, b_k=None):
        a = np.asarray(a_k)
        if b_k is None:
            p = np.abs(a) ** 2
        else:
            p = np.real(a * np.conj(np.asarray(b_k)))
        p *= self.norm
        p *= self.herm
        sums = np.bincount(self.idx, weights=p.ravel(), minlength=self.nb + 1)
        return sums[: self.nb] / self.wsum[: self.nb]


def one(cfg, b, realization, nus, radii, geom):
    t0 = time.perf_counter()
    box, shell = cfg.box(b), cfg.shell(b)
    e = cfg.effective_bin(b)
    f = b.f
    spectrum = PowerSpectrum.from_tsv(cfg.pk_path(b))
    galaxy_table = (
        PowerSpectrum.from_tsv(cfg.pk_galaxy_path(b)) if b.pk_galaxy_file else None
    )
    ic, _ = sample.split_seed(suite.seed_for(realization, b.index, cfg.seed_base))
    F = field.generate_fields(
        spectrum,
        b.b,
        box,
        ic,
        jitter_p=cfg.jitter_p,
        psi_axes="xyz",
        keep_matter=True,
        fnl=cfg.png,
        galaxy_table=galaxy_table,
    )
    t_field = time.perf_counter() - t0
    dm = F.delta_m
    psi0 = F.psi_flat("xyz")
    F.psi.clear()
    offsets, P0, r, h = geom["offsets"], geom["P0"], geom["r"], geom["h"]
    sigma_G = float(np.sqrt(np.log1p(F.diagnostics["matter"]["sigma2"])))
    win = cell_window(box, shell).reshape(-1)
    lam = (e.nbar * box.v_cell * (1.0 + np.asarray(F.delta_g))).reshape(-1)
    lam_window = float(lam[win].sum())

    dg_k = jnp.fft.rfftn(F.delta_g)
    dm_k = jnp.fft.rfftn(dm)
    cross0 = power_multipoles(np.asarray(dg_k), box, ells=(0,), delta_k_b=dm_k)
    cross0_fast = P0(dg_k, dm_k)
    # same sums, products in another order: a few ulps per mode
    assert np.allclose(cross0_fast, cross0["P0"], rtol=1e-12, atol=0.0)
    auto0 = P0(dm_k)

    # nu = inf reproduces production Psi bitwise (the instrument is the production path)
    src, _, _ = capped_source(dm, np.inf)
    src_k = jnp.fft.rfftn(jnp.asarray(src.reshape(box.shape)))
    assert np.array_equal(psi_of(src_k, box), psi0), "nu=inf != production"
    del src, src_k

    def measure(psi):
        need = shell.required_buffer(box, f, psi)
        bound = magnitude_bound(r, h, shell, f, psi)
        assert need <= bound, (need, bound)
        return need, bound

    need0, bound0 = measure(psi0)
    # only `f|Psi - Psi0| > 1 Mpc/h` reads psi0 from here on: float32 (|Psi| < 1e3,
    # error < 1e-4 Mpc/h) halves it
    psi0 = psi0.astype(np.float32)
    row = dict(
        realization=realization,
        bin=b.index,
        n_grid=box.n_mesh,
        dx=box.dx,
        f=f,
        shell=[shell.rmin, shell.rmax, shell.buffer],
        box_room=0.5 * box.box_size - shell.rmax,
        sigma_G=sigma_G,
        delta_m_max=float(dm.max()),
        lam_window=lam_window,
        k=cross0["k"].tolist(),
        n_indep=cross0["n_indep"].tolist(),
        cross0=cross0["P0"].tolist(),
        required_buffer=need0,
        magnitude_bound=bound0,
        arms=[],
    )
    for nu in nus:
        delta_max = float(np.expm1(nu * sigma_G - 0.5 * sigma_G**2))
        for radius in [None] + list(radii):
            off = offsets[radius] if radius is not None else None
            n_cells = 1 if off is None else off.shape[0]
            src, moved, residual = capped_source(dm, delta_max, off)
            if radius is not None:
                assert abs(residual) < MASS_RESIDUAL_MAX, residual
            src_max = float(src.max())
            src_k = jnp.fft.rfftn(jnp.asarray(src.reshape(box.shape)))
            del src
            psi = psi_of(src_k, box)
            need, bound = measure(psi)
            psi -= psi0
            dpsi = abs(f) * psi_norm(psi)
            touched = win & (dpsi > DPSI_GALAXIES)
            row["arms"].append(
                dict(
                    arm="plain" if radius is None else "mass",
                    nu=nu,
                    radius_cells=radius,
                    kernel_cells=n_cells,
                    delta_max=delta_max,
                    cells_capped=int((np.asarray(dm) > delta_max).sum()),
                    mass_moved=moved / box.n_mesh**3,
                    mass_residual_rel=residual,
                    src_max=src_max,
                    required_buffer=need,
                    magnitude_bound=bound,
                    max_f_dpsi=float(dpsi.max()),
                    galaxies_touched=float(lam[touched].sum()),
                    cells_touched_window=int(touched.sum()),
                    cross_ratio=(P0(dg_k, src_k) / cross0_fast).tolist(),
                    auto_ratio=(P0(src_k) / auto0).tolist(),
                )
            )
            del src_k, psi, dpsi, touched
    row["t_field_s"] = t_field
    row["t_total_s"] = time.perf_counter() - t0
    row["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1e9 if sys.platform == "darwin" else 1e6
    )
    return row


def report(o):
    print(
        f"\n=== r{o['realization']} bin {o['bin']}: N {o['n_grid']} dx {o['dx']:.2f}"
        f" sigma_G {o['sigma_G']:.3f} delta_m max {o['delta_m_max']:.0f}"
        f" | uncapped need {o['required_buffer']:.1f} (|Psi| bound "
        f"{o['magnitude_bound']:.1f}), room {o['box_room']:.1f}"
        f" | {o['t_total_s']:.0f} s, field {o['t_field_s']:.1f} s,"
        f" RSS {o['peak_rss_gb']:.1f} GB"
    )
    k = np.array(o["k"])
    lo = k <= 0.02
    for a in o["arms"]:
        cr = np.array(a["cross_ratio"])
        tag = f"{a['arm']:5s} nu {a['nu']:.1f}" + (
            f" R{a['radius_cells']}" if a["radius_cells"] else "   "
        )
        print(
            f"  {tag}: dmax {a['delta_max']:6.1f} capped {a['cells_capped']:>6}"
            f" moved {a['mass_moved']:.1e} | need {a['required_buffer']:6.1f}"
            f" (bound {a['magnitude_bound']:6.1f}) | touched gal "
            f"{a['galaxies_touched'] / o['lam_window']:.1e} | cross-1 k_f "
            f"{cr[0] - 1:+.1e}, max|.| k<=0.02 {np.abs(cr[lo] - 1).max():.1e},"
            f" all k {np.abs(cr - 1).max():.1e}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/v28_halofit.yaml")
    ap.add_argument("--bins", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--realizations", type=int, nargs="+", default=[0])
    ap.add_argument("--nu", type=float, nargs="+", default=[5.0, 4.5, 4.0])
    ap.add_argument("--radius", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--out", default="runs/m5/cap.jsonl")
    args = ap.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["realization"], r["bin"]))
    for b in cfg.bins_by_index(args.bins):
        box = cfg.box(b)
        geom = dict(
            offsets={R: sphere_offsets(R) for R in args.radius},
            P0=ShellP0(box),
            r=cell_radius(box).reshape(-1),
            h=0.5 * np.sqrt(3.0) * box.dx,
        )
        for r in args.realizations:
            if (r, b.index) in done:
                print(f"skip r{r} bin {b.index} (in {out})")
                continue
            row = one(cfg, b, r, args.nu, args.radius, geom)
            report(row)
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        del geom
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
