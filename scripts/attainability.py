"""Whether each bin's target is reachable by a lognormal, over f_NL values and galaxy
tables; writes `runs/attainability/<mode>_<stamp>.json`.

    pixi run python scripts/attainability.py attain [--fnl -100 -10 -1 1 10 100]
        [--bins 1 7] [--grid-scale 2] [--pk-template NAME] [--pk-galaxy-template NAME]

For every v28 bin at its production grid (times `--grid-scale`, rounded as
`RunConfig.effective_bin` does), the galaxy target `b(k)^2 P_gal / sinc^2` goes through
the field stage's own P -> P_G conversion (no sampling). Each row: `xi_min`, sigma^2,
clipped P_G modes and their power fraction, `b(k_f) / b` and the k where b(k) changes
sign. f_NL = 0 is the reference row; one `matter` row per bin is `P / sinc^2`.
`--pk-template` names the linear table and `--pk-galaxy-template` the galaxy one
(`{z:g}` is the bin's z_eff; defaults: the bin's `pk_file`, then the linear table);
the pair is checked by `pk.check_table_pair`, as in a run. With f_NL != 0 or a galaxy
table, a clipped galaxy row is a target a run refuses (`docs/construction.md`).
"""

import argparse
import dataclasses
import datetime as dt
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp  # noqa: E402

from logunusual import config, field, fnl, suite  # noqa: E402
from logunusual.pk import PowerSpectrum, check_table_pair, grid_pkG  # noqa: E402


def attain(args):
    rows = []
    bins = [b for b in suite.BIN_SUITE_V28 if b.index in args.bins]
    cfg = dataclasses.replace(config.default_config(), grid_scale=args.grid_scale)
    pk_dir = Path(args.pk_dir)
    for b in bins:
        box = cfg.box(b)
        lin_name = args.pk_template.format(z=b.z_eff) if args.pk_template else b.pk_file
        spectrum = PowerSpectrum.from_tsv(pk_dir / lin_name)
        gal_name = lin_name
        galaxy_table = None
        if args.pk_galaxy_template:
            gal_name = args.pk_galaxy_template.format(z=b.z_eff)
            galaxy_table = PowerSpectrum.from_tsv(pk_dir / gal_name)
            check_table_pair(spectrum, galaxy_table)
        print(
            f"{b.name}: N = {box.n_mesh}, L = {b.L_box}, dx = {box.dx:.3f}, "
            f"b = {b.b}, z_eff = {b.z_eff}, linear {lin_name}, galaxy {gal_name}"
        )
        base = {
            "bin": b.index,
            "N": box.n_mesh,
            "L": b.L_box,
            "b": b.b,
            "pk_file": lin_name,
            "pk_galaxy_file": gal_name,
        }
        t0 = time.perf_counter()
        target = field.target_on_grid(spectrum, box, 1)
        _, diag = grid_pkG(jnp.asarray(target), box, jnp)
        del target, _  # free the grids before the next row (1024^3)
        rows.append(
            {**base, "field": "matter", **diag, "t_s": time.perf_counter() - t0}
        )
        print(
            f"  matter           xi_min {diag['xi_min']:+.3e}  "
            f"sigma2 {diag['sigma2']:.4f}  clipped {diag['n_clipped']:>9,} "
            f"({diag['clipped_power_fraction']:.2e})"
        )
        for f_nl in [0.0] + list(args.fnl):
            png = None if f_nl == 0 else fnl.LocalPNG(f_nl=f_nl)
            t0 = time.perf_counter()
            target = field.target_on_grid(
                fnl.galaxy_spectrum(spectrum, b.b, png, galaxy_table), box, 1
            )
            _, diag = grid_pkG(jnp.asarray(target), box, jnp)
            del target, _
            row = {
                **base,
                "field": "galaxy",
                "f_nl": f_nl,
                **diag,
                **(
                    {"b_kf_over_b": 1.0, "k_zero": None, "delta_b_kf": 0.0}
                    if png is None
                    else fnl.diagnostics(spectrum, b.b, png, box)
                ),
                "t_s": time.perf_counter() - t0,
            }
            rows.append(row)
            kz = "-" if row["k_zero"] is None else f"{row['k_zero']:.2e}"
            print(
                f"  f_NL {f_nl:+7g}  xi_min {row['xi_min']:+.3e}  "
                f"sigma2 {row['sigma2']:.4f}  clipped {row['n_clipped']:>9,} "
                f"({row['clipped_power_fraction']:.2e})  "
                f"b(k_f)/b {row['b_kf_over_b']:+.3f}  k_zero {kz}"
            )
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["attain"])
    ap.add_argument("--fnl", type=float, nargs="+", default=[-100, -10, -1, 1, 10, 100])
    ap.add_argument("--bins", type=int, nargs="+", default=list(range(1, 8)))
    ap.add_argument("--grid-scale", type=float, default=1.0)
    ap.add_argument("--pk-dir", default="data")
    ap.add_argument("--pk-template", default=None)
    ap.add_argument("--pk-galaxy-template", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    res = attain(args)
    res.update(
        {
            "mode": args.mode,
            "convention": fnl.CONVENTION,
            "png_defaults": fnl.LocalPNG(f_nl=0.0).as_dict(),
            "host": platform.node(),
            "wall_s": time.perf_counter() - t0,
        }
    )
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or f"runs/attainability/{args.mode}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out} ({res['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
