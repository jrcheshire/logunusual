"""`logunusual` command line: `run`, `check`, `default-config`."""

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")


def parse_realizations(spec: str):
    """`a:b` (half-open) or a single integer -> list of ints."""
    if ":" in spec:
        a, b = spec.split(":", 1)
        a = int(a) if a else 0
        b = int(b)
        if b <= a:
            raise argparse.ArgumentTypeError(f"empty realization range {spec!r}")
        return list(range(a, b))
    return [int(spec)]


def cmd_run(args):
    from .config import RunConfig
    from .run import generate_realization, plan_realization

    cfg = RunConfig.from_yaml(args.config)
    if args.output_dir is not None:
        from dataclasses import replace

        cfg = replace(cfg, output_dir=args.output_dir)
    print(
        f"run {cfg.run_name}: config hash {cfg.config_hash[:12]}, "
        f"{len(cfg.bins_by_index(args.bins))} bin(s), realizations {args.realizations}"
    )
    if args.dry_run:
        for r in args.realizations:
            print(f"realization {r:05d}:")
            for row in plan_realization(cfg, r, args.bins):
                print(
                    f"  bin {row['index']} {row['name']}: N={row['n_grid']} "
                    f"L={row['box_size']:g} dx={row['cell']:.3f} "
                    f"shell=[{row['rmin']:g}, {row['rmax']:g}] +-{row['buffer']:g} "
                    f"nbar={row['nbar_target']:.4g} seed={row['seed']} "
                    f"expected full-sky kept ~{row['expected_kept_fullsky']:.3g} "
                    f"pk={row['pk']}"
                )
        return 0
    for r in args.realizations:
        print(f"=== realization {r:05d}")
        generate_realization(cfg, r, bins=args.bins, overwrite=args.overwrite)
    return 0


def cmd_check(args):
    from . import io

    rc = 0
    for path in args.catalog:
        rep = io.check_layout(path)
        meta = io.read_metadata(path)
        print(
            f"{path}: {rep.n_rows:,} rows, {rep.n_row_groups} row groups of "
            f"{rep.row_group_rows}, bins {rep.bins}"
        )
        print(
            f"  generator {meta.get('generator')} {meta.get('generator_version')} "
            f"on {meta.get('host')} ({meta.get('jax_backend')}), "
            f"config {str(meta.get('config_hash'))[:12]}, mask fsky "
            f"{meta.get('mask_fsky')}"
        )
        print(
            f"  {'bin':>4} {'rows':>14} {'nbar_target':>12} {'realized':>12} "
            f"{'ratio':>7} {'N':>5} {'L':>7}"
        )
        for b in rep.bins:
            m = io.bin_metadata(meta, b)
            tgt = float(m.get("nbar_target", "nan"))
            rea = float(m.get("realized_nbar", "nan"))
            print(
                f"  {b:>4} {rep.rows_per_bin[b]:>14,} {tgt:>12.4e} {rea:>12.4e} "
                f"{rea / tgt:>7.4f} {m.get('n_grid', '?'):>5} "
                f"{m.get('box_size', '?'):>7}"
            )
        if rep.ok:
            print("  layout: ok")
        else:
            rc = 1
            for p in rep.problems:
                print(f"  PROBLEM: {p}")
    return rc


def cmd_default_config(args):
    from .config import default_config

    print(default_config().to_yaml(), end="")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="logunusual", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate realizations from a YAML config")
    r.add_argument("--config", required=True)
    r.add_argument(
        "--realizations",
        required=True,
        type=parse_realizations,
        help="`a:b` (half-open) or a single index",
    )
    r.add_argument("--bins", type=int, nargs="+", default=None)
    r.add_argument("--output-dir", default=None, help="override the config's")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("check", help="verify a catalog's layout and print its bins")
    c.add_argument("catalog", nargs="+")
    c.set_defaults(func=cmd_check)

    d = sub.add_parser("default-config", help="print the default (v28) config as YAML")
    d.set_defaults(func=cmd_default_config)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
