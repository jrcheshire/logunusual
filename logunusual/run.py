"""The realization driver: bins in order, one field stage and one streamed shell draw
per bin, one parquet per realization plus a `summary.json`.

Seeds: `suite.seed_for(realization, bin.index, base)` -> `sample.split_seed` ->
`(ic_seed, draw_seed)`, so every (realization, bin) has its own independent field and
draw streams and a field can be resampled.
"""

import datetime as dt
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

from . import __version__, field, fnl, io, sample, suite  # noqa: E402
from .config import RunConfig  # noqa: E402
from .pk import PowerSpectrum, check_table_pair  # noqa: E402
from .shell import AngularMask, sample_shell  # noqa: E402


def realization_dir(cfg: RunConfig, realization: int) -> Path:
    return cfg.output_dir / cfg.run_name / f"realization_{realization:05d}"


def catalog_path(cfg: RunConfig, realization: int) -> Path:
    return realization_dir(cfg, realization) / "catalog.parq"


def plan_realization(cfg: RunConfig, realization: int, bins=None):
    """Per-bin plan rows (what `generate_realization` will do), for `--dry-run`."""
    rows = []
    for b in cfg.bins_by_index(bins):
        e = cfg.effective_bin(b)
        box = cfg.box(b)
        shell = cfg.shell(b)
        seed = suite.seed_for(realization, b.index, cfg.seed_base)
        ic, draw = sample.split_seed(seed)
        rows.append(
            {
                "index": b.index,
                "name": b.name,
                "n_grid": box.n_mesh,
                "box_size": box.box_size,
                "cell": box.dx,
                "rmin": shell.rmin,
                "rmax": shell.rmax,
                "buffer": shell.buffer,
                "nbar_target": e.nbar,
                "b": b.b,
                "f": b.f,
                "pk": str(cfg.pk_path(b)),
                "pk_galaxy": str(cfg.pk_galaxy_path(b)),
                "seed": seed,
                "ic_seed": ic,
                "draw_seed": draw,
                "expected_kept_fullsky": e.nbar * shell.volume,
            }
        )
    return rows


def _live_bytes():
    import jax

    return sum(int(a.nbytes) for a in jax.live_arrays())


def _global_metadata(cfg: RunConfig, mask, bins, n_workers: int):
    import jax

    meta = {
        "generator": "logunusual",
        "generator_version": __version__,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "machine": platform.machine(),
        "system": platform.system(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "run_name": cfg.run_name,
        "config_hash": cfg.config_hash,
        "seed_base": cfg.seed_base,
        "nbar_scale": cfg.nbar_scale,
        "grid_scale": cfg.grid_scale,
        "radial_buffer": cfg.radial_buffer,
        "jitter_p": cfg.jitter_p,
        "rsd": "radial",
        "rng_scheme": sample.RNG_SCHEME,
        "n_workers": n_workers,
        "angular_precut": cfg.angular_precut,
        "velocity_assignment": "own-cell",
        "frame": "observer at origin; each bin its own periodic box centred there",
        "shell_edges": "inclusive",
        "nbar_overdensity_factor": 1.0,
        "f_nl": float(cfg.f_nl),
        "bins": ",".join(str(b.index) for b in bins),
    }
    png = cfg.png
    if png is not None:
        meta["fnl_convention"] = fnl.CONVENTION
        meta.update({f"fnl_{k}": v for k, v in png.as_dict().items() if k != "f_nl"})
        meta["fnl_g0"] = png.g0
    if mask is None:
        meta["mask"] = "none"
        meta["mask_fsky"] = 1.0
    else:
        meta.update(
            {
                "mask": Path(mask.path).name,
                "mask_dataset": mask.dataset,
                "mask_sha256": mask.sha256,
                "mask_nside": mask.nside,
                "mask_ordering": "NESTED" if mask.nested else "RING",
                "mask_fsky": mask.fsky,
            }
        )
    return meta


def generate_realization(
    cfg: RunConfig, realization: int, *, bins=None, overwrite=False, log=print
):
    """Write `catalog.parq` and `summary.json` for one realization. Returns the
    summary dict; returns `None` (and logs) if the catalog exists and not
    `overwrite`."""
    out = catalog_path(cfg, realization)
    if out.exists() and not overwrite:
        log(f"exists, skipping: {out}")
        return None
    todo = cfg.bins_by_index(bins)
    mask = (
        None
        if cfg.mask_path is None
        else AngularMask.from_h5(cfg.mask_path, cfg.mask_dataset)
    )
    fsky = 1.0 if mask is None else mask.fsky
    n_workers = cfg.n_workers or sample.default_workers()
    png = cfg.png
    log(
        f"run {cfg.run_name}: {len(todo)} bin(s), {n_workers} sampler thread(s)"
        + ("" if png is None else f", f_NL = {png.f_nl:g} ({fnl.CONVENTION})")
    )
    t_start = time.perf_counter()
    summary = {
        "realization": realization,
        "run_name": cfg.run_name,
        "config_hash": cfg.config_hash,
        "path": str(out),
        "bins": [],
    }
    peak_live = 0
    with io.CatalogWriter(
        out,
        _global_metadata(cfg, mask, todo, n_workers),
        row_group_rows=cfg.row_group_rows,
    ) as writer:
        for b in todo:
            e = cfg.effective_bin(b)
            box, shell = cfg.box(b), cfg.shell(b)
            shell.check_box(box)
            spectrum = PowerSpectrum.from_tsv(cfg.pk_path(b))
            galaxy_table = None
            if b.pk_galaxy_file:
                galaxy_table = PowerSpectrum.from_tsv(cfg.pk_galaxy_path(b))
                check_table_pair(spectrum, galaxy_table)
            seed = suite.seed_for(realization, b.index, cfg.seed_base)
            ic, draw = sample.split_seed(seed)
            log(
                f"bin {b.index} ({b.name}): N={box.n_mesh} L={box.box_size:g} "
                f"dx={box.dx:.3f} shell=[{shell.rmin:g}, {shell.rmax:g}] "
                f"nbar={e.nbar:.4g} seed={seed}"
            )
            t0 = time.perf_counter()
            peak = {"b": 0}

            def trace(label):
                peak["b"] = max(peak["b"], _live_bytes())

            F = field.generate_fields(
                spectrum,
                b.b,
                box,
                ic,
                jitter_p=cfg.jitter_p,
                psi_axes="xyz",
                fnl=png,
                galaxy_table=galaxy_table,
                trace=trace,
            )
            t1 = time.perf_counter()
            stats = None
            for kept, stats in sample_shell(
                F,
                shell,
                mask,
                e.nbar,
                draw,
                f=b.f,
                n_workers=n_workers,
                angular_precut=cfg.angular_precut,
                n_radial_bins=cfg.n_radial_bins,
            ):
                writer.write(kept, b.index)
            writer.end_bin()
            t2 = time.perf_counter()
            diag = F.diagnostics
            del F
            peak_live = max(peak_live, peak["b"])
            realized = stats.realized_nbar(fsky)
            row = {
                "index": b.index,
                "name": b.name,
                "z_min": b.z_min,
                "z_max": b.z_max,
                "z_eff": b.z_eff,
                "rmin": shell.rmin,
                "rmax": shell.rmax,
                "box_size": box.box_size,
                "n_grid": box.n_mesh,
                "cell": box.dx,
                "b": b.b,
                "f": b.f,
                "nbar_nominal": b.nbar,
                "nbar_target": e.nbar,
                "nbar_scale": cfg.nbar_scale,
                "realized_nbar": realized,
                "realized_over_target": realized / e.nbar,
                "n_galaxies": stats.n_kept,
                "n_drawn": stats.n_drawn,
                "n_left_box": stats.n_left_box,
                "lam_window": stats.lam_window,
                "n_window_cells": stats.n_window_cells,
                "n_window_cells_radial": stats.n_window_cells_radial,
                "radial_buffer": shell.buffer,
                "psi_max": stats.psi_max,
                "required_buffer": stats.required_buffer,
                "seed": seed,
                "ic_seed": ic,
                "draw_seed": draw,
                "pk_file": b.pk_file,
                "pk_sha256": spectrum.file_hash,
                "pk_galaxy_file": b.pk_galaxy_file or b.pk_file,
                "pk_galaxy_sha256": (
                    spectrum if galaxy_table is None else galaxy_table
                ).file_hash,
                "sigma2_galaxy": diag["galaxy"]["sigma2"],
                "sigma2_matter": diag["matter"]["sigma2"],
                "xi_min_galaxy": diag["galaxy"]["xi_min"],
                "clipped_power_fraction": diag["galaxy"]["clipped_power_fraction"],
                "psi_rms": diag["psi_rms"],
                "r_edges": stats.r_edges.tolist(),
                "r_hist": stats.r_hist.tolist(),
                "t_field_s": t1 - t0,
                "t_sample_s": t2 - t1,
                "peak_live_jax_bytes": peak["b"],
            }
            if png is not None:
                d = diag["fnl"]
                row["fnl_delta_b_kf"] = d["delta_b_kf"]
                row["fnl_b_kf_over_b"] = d["b_kf_over_b"]
                row["fnl_k_zero"] = d["k_zero"]
            summary["bins"].append(row)
            writer.add_metadata(
                {
                    io.bin_key(b.index, k): (
                        json.dumps(v) if isinstance(v, dict) else v
                    )
                    for k, v in row.items()
                    if k not in ("r_edges", "r_hist")
                }
            )
            log(
                f"  drawn {stats.n_drawn:,}  kept {stats.n_kept:,}  "
                f"realized/target {realized / e.nbar:.4f}  "
                f"field {t1 - t0:.1f} s  sample {t2 - t1:.1f} s"
            )
    summary["n_galaxies"] = int(sum(r["n_galaxies"] for r in summary["bins"]))
    summary["wall_s"] = time.perf_counter() - t_start
    summary["peak_live_jax_bytes"] = peak_live
    summary["file_bytes"] = out.stat().st_size
    (out.parent / "summary.json").write_text(json.dumps(summary, indent=1))
    log(
        f"wrote {out} ({summary['n_galaxies']:,} galaxies, "
        f"{summary['file_bytes'] / 2**30:.2f} GiB, {summary['wall_s']:.0f} s)"
    )
    return summary
