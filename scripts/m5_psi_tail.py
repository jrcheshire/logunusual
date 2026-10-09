"""M5: what sets the displacement tail that trips the buffer guard (bin 1, r50).

    pixi run python scripts/m5_psi_tail.py [--config configs/v28_halofit.yaml]
        [--bin 1] [--realizations 0 50] [--grid-scale 1] [--out runs/m5/psi_tail.json]

For each realization, the bin's production field stage (`generate_fields`, the run's
arguments, bitwise) and, from the same white noise:
- where max |Psi| sits relative to the densest matter cell, delta_m and the Gaussian
  field's sigma there, |Psi| quantiles, and the cells / expected galaxies with
  f |Psi| above 25-150 Mpc/h;
- a point-mass estimate `delta_peak V_cell / (4 pi d^2)` from the densest cell alone;
- removal: delta_m set to 0 in the top-K cells by delta_m, max f |Psi| recomputed;
- Psi from a Gaussian field with the matter target `P / sinc^2` (the same white noise
  and the same two-point function as the lognormal), and from the lognormal without
  the jitter deconvolution of the matter target (`P` instead of `P / sinc^2`).
Maxima of f |Psi| are over the radial window, as `sample_shell`'s guard takes them.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from logunusual import field, sample, suite  # noqa: E402
from logunusual.config import RunConfig  # noqa: E402
from logunusual.pk import PowerSpectrum, grid_pkG  # noqa: E402
from logunusual.shell import cell_window  # noqa: E402

THRESHOLDS = (25.0, 50.0, 100.0, 150.0)  # Mpc/h, on f |Psi|
QUANTILES = (0.5, 0.99, 0.999, 0.99999)


def psi_mag(delta, box):
    """|Psi| from a real-space density contrast via `field.displacement`."""
    dk = jnp.fft.rfftn(delta)
    comps = [np.asarray(field.displacement(dk, box, a)) for a in field.AXES]
    return np.sqrt(comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2), comps


def periodic_cells(a, b, n):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, n - d)


def tail(mag, f, win):
    return dict(
        max_f_psi_window=float(f * mag[win].max()),
        max_f_psi_box=float(f * mag.max()),
        rms=float(np.sqrt(np.mean(mag**2) / 3.0)),  # per component
        quantiles={str(q): float(np.quantile(mag, q)) for q in QUANTILES},
    )


def one(cfg, b, realization):
    box, shell = cfg.box(b), cfg.shell(b)
    e = cfg.effective_bin(b)
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
        galaxy_table=galaxy_table,
    )
    n, f, win = box.n_mesh, b.f, cell_window(box, shell)
    dm = np.asarray(F.delta_m)
    psi = [np.asarray(F.psi[a]) for a in field.AXES]
    mag = np.sqrt(psi[0] ** 2 + psi[1] ** 2 + psi[2] ** 2)
    lam = e.nbar * box.v_cell * (1.0 + np.asarray(F.delta_g))

    # the Gaussian field behind delta_m, to express the peak in its own sigma
    white_k = jnp.fft.rfftn(jnp.asarray(field.white_noise(box, ic)))
    target_m = field.target_on_grid(spectrum, box, cfg.jitter_p)
    pkG_m, _ = grid_pkG(jnp.asarray(target_m), box, jnp)
    G = np.asarray(field.colour(white_k, pkG_m, box))

    i_peak = np.unravel_index(np.argmax(dm), dm.shape)
    i_psi = np.unravel_index(np.argmax(np.where(win, mag, -1.0)), mag.shape)
    d_cells = periodic_cells(i_psi, i_peak, n)
    d = float(np.sqrt(np.sum(d_cells**2))) * box.dx
    point = (
        float(dm[i_peak] * box.v_cell / (4 * np.pi * d**2)) if d > 0 else float("nan")
    )
    r_c = (np.array(i_psi) + 0.5) * box.dx - 0.5 * box.box_size
    out = dict(
        realization=realization,
        bin=b.index,
        n_grid=n,
        dx=box.dx,
        f=f,
        shell=[shell.rmin, shell.rmax, shell.buffer],
        sigma_G=float(G.std()),
        delta_m_max=float(dm[i_peak]),
        G_peak_sigmas=float((G[i_peak] - G.mean()) / G.std()),
        peak_cell=[int(x) for x in i_peak],
        max_psi_cell=[int(x) for x in i_psi],
        max_psi_cell_r=float(np.linalg.norm(r_c)),
        max_psi_cell_delta_m=float(dm[i_psi]),
        max_psi_to_peak_cells=[int(x) for x in d_cells],
        point_mass_psi_at_max_cell=point,
        production=tail(mag, f, win),
        over_threshold={
            str(t): dict(
                cells=int((f * mag > t).sum()),
                cells_window=int(((f * mag > t) & win).sum()),
                expected_galaxies_window=float(lam[(f * mag > t) & win].sum()),
            )
            for t in THRESHOLDS
        },
        expected_galaxies_window=float(lam[win].sum()),
    )

    order = np.argsort(dm, axis=None)[::-1]
    out["removal"] = {}
    for k in (1, 8, 64):
        cut = dm.copy().reshape(-1)
        cut[order[:k]] = 0.0
        m, _ = psi_mag(jnp.asarray(cut.reshape(dm.shape)), box)
        out["removal"][str(k)] = float(f * m[win].max())

    dL = np.asarray(field.colour(white_k, jnp.asarray(target_m), box))
    mL, compL = psi_mag(jnp.asarray(dL), box)
    out["gaussian_matter"] = tail(mL, f, win)
    out["gaussian_matter"]["corr_with_production"] = [
        float(np.corrcoef(compL[i].ravel(), psi[i].ravel())[0, 1]) for i in range(3)
    ]

    pkG_m0, _ = grid_pkG(jnp.asarray(field.target_on_grid(spectrum, box, 0)), box, jnp)
    d0 = np.asarray(field.coloured_lognormal(white_k, pkG_m0, box))
    m0, _ = psi_mag(jnp.asarray(d0), box)
    out["lognormal_no_deconvolution"] = tail(m0, f, win)
    out["lognormal_no_deconvolution"]["delta_m_max"] = float(d0.max())
    return out


def report(o):
    p = o["production"]
    print(
        f"\n=== r{o['realization']} bin {o['bin']}: N {o['n_grid']} dx {o['dx']:.2f}"
        f" f {o['f']:.4f} shell {o['shell']}"
    )
    print(
        f"  densest matter cell: delta_m {o['delta_m_max']:.1f}, G at "
        f"{o['G_peak_sigmas']:.2f} sigma (sigma_G {o['sigma_G']:.3f})"
    )
    print(
        f"  max |Psi| cell (window): f|Psi| {p['max_f_psi_window']:.1f} Mpc/h "
        f"(box {p['max_f_psi_box']:.1f}), r {o['max_psi_cell_r']:.0f}, "
        f"delta_m {o['max_psi_cell_delta_m']:.1f}, offset from peak "
        f"{o['max_psi_to_peak_cells']} cells; point mass from the peak alone "
        f"f|Psi| {o['f'] * o['point_mass_psi_at_max_cell']:.1f}"
    )
    print(
        f"  |Psi| rms/component {p['rms']:.2f}; quantiles "
        + ", ".join(f"{q}: {v:.1f}" for q, v in p["quantiles"].items())
    )
    for t, v in o["over_threshold"].items():
        print(
            f"  f|Psi| > {float(t):>5.0f}: {v['cells']:>7} cells "
            f"({v['cells_window']} in window), "
            f"{v['expected_galaxies_window']:,.0f} expected galaxies of "
            f"{o['expected_galaxies_window']:,.0f}"
        )
    print(
        "  removal (delta_m -> 0 in top-K cells), max f|Psi| window: "
        + ", ".join(f"K={k}: {v:.1f}" for k, v in o["removal"].items())
    )
    for name in ("gaussian_matter", "lognormal_no_deconvolution"):
        a = o[name]
        extra = (
            f", corr with production {[round(c, 3) for c in a['corr_with_production']]}"
            if "corr_with_production" in a
            else f", delta_m max {a['delta_m_max']:.1f}"
        )
        print(
            f"  {name}: max f|Psi| window {a['max_f_psi_window']:.1f}, rms "
            f"{a['rms']:.2f}, p99.999 {a['quantiles']['0.99999']:.1f}{extra}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/v28_halofit.yaml")
    ap.add_argument("--bin", type=int, default=1)
    ap.add_argument("--realizations", type=int, nargs="+", default=[0, 50])
    ap.add_argument("--grid-scale", type=float, default=None)
    ap.add_argument("--out", default="runs/m5/psi_tail.json")
    args = ap.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    if args.grid_scale is not None:
        cfg = dataclasses.replace(cfg, grid_scale=args.grid_scale)
    (b,) = cfg.bins_by_index([args.bin])
    rows = []
    for r in args.realizations:
        rows.append(one(cfg, b, r))
        report(rows[-1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
