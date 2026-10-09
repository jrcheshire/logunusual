"""Periodic-box statistical checks with per-band tables; writes
`runs/gates_box/<stamp>_n<N>.json`.

    pixi run python scripts/gates_box.py [--n 128] [--L 1000] [--seeds 32]
        [--seeds-shot 64] [--band-seeds 32] [--nbar 3e-3]

The same `gates` functions back `tests/test_gates_box.py`, and the defaults are the
test's settings; this script runs other configurations and keeps every number. Bars
and their derivation: `docs/validation.md`.
"""

import argparse
import datetime as dt
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

from logunusual import field, gates, suite  # noqa: E402
from logunusual.grid import Box  # noqa: E402
from logunusual.pk import PowerSpectrum  # noqa: E402


def report(name, e):
    if e is None:
        print(f"\n{name}: (no band)")
        return
    print(f"\n{name}: n_real={e.n_real} passed={e.passed}")
    for k, m, se, z, ok in zip(e.k, e.mean, e.se, e.z, e.se_ok):
        flag = "" if ok else "  (SE > tol/3)"
        print(f"  k={k:.4f}  ratio={m:.4f} +- {se:.4f}  z={z:+.2f}{flag}")
    for key, val in e.extra.items():
        print(f"  {key}: {val}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=float, default=1000.0)
    ap.add_argument("--est-factor", type=int, default=2)
    ap.add_argument(
        "--seeds", type=int, default=32, help="catalog and Kaiser-limit realizations"
    )
    ap.add_argument(
        "--seeds-shot", type=int, default=64, help="uniform-field shot-noise draws"
    )
    ap.add_argument(
        "--band-seeds",
        type=int,
        default=32,
        help="seed count the shot-noise BANDS are designed for; more draws than this "
        "through the same bands is how a super-Gaussian scatter clears the SE floor",
    )
    ap.add_argument("--seeds-field", type=int, default=192)
    ap.add_argument("--nbar", type=float, default=3e-3)
    ap.add_argument(
        "--bin", type=int, default=5, help="v28 bin whose b, f, TSV are used"
    )
    ap.add_argument("--pk", default=None, help="TSV path (default: the bin's file)")
    ap.add_argument("--out", default="runs/gates_box")
    args = ap.parse_args()

    b = suite.BIN_SUITE_V28[args.bin - 1]
    spectrum = PowerSpectrum.from_tsv(args.pk or b.matterpower_file)
    box = Box(args.n, args.L)
    box_est = Box(args.n * args.est_factor, args.L)
    print(f"generator {box}  estimator {box_est}  b={b.b} f={b.f:.4f} nbar={args.nbar}")
    print(f"host {platform.node()} {platform.machine()} {platform.system()}")
    out = {
        "config": vars(args)
        | {
            "b": b.b,
            "f": b.f,
            "pk_hash": spectrum.file_hash,
            "host": platform.node(),
            "machine": platform.machine(),
        }
    }

    t = time.perf_counter()
    r = gates.gate_field_identity(
        spectrum, b.b, box, range(3000, 3000 + args.seeds_field)
    )
    report("galaxy grid identity", r["galaxy"])
    report("matter grid identity", r["matter"])
    out["grid_identity"] = {k: v.as_dict() for k, v in r.items()}
    print(f"[grid identity {time.perf_counter() - t:.0f} s]")

    t = time.perf_counter()
    e = gates.gate_uniform_shot(
        box,
        box_est,
        args.nbar,
        range(4000, 4000 + args.seeds_shot),
        band_seeds=args.band_seeds,
    )
    report("uniform-field shot noise vs Jing", e)
    out["uniform_shot"] = e.as_dict()
    F = field.generate_fields(spectrum, b.b, box, 4100, rsd=False)
    r = gates.gate_fixed_field_sampler(F, args.nbar, range(4200, 4200 + 16), box_est)
    report("fixed-field sampler, first zone", r["first_zone"])
    report("fixed-field sampler, to estimator Nyquist", r["to_estimator_nyquist"])
    out["fixed_field_sampler"] = {k: v.as_dict() for k, v in r.items()}
    print(f"[shot noise and sampler {time.perf_counter() - t:.0f} s]")

    t = time.perf_counter()
    r = gates.gate_catalog(
        spectrum, b.b, b.f, box, box_est, args.nbar, range(5000, 5000 + args.seeds)
    )
    report("monopole vs b^2 P_in x estimator response", r["monopole"])
    report("monopole vs fixed-field prediction", r["monopole_fixed_field"])
    report("RSD premise P_gm/(b P_mm) (measurement)", r["premise"])
    print(f"  f Psi_rms = {r['f_psi_rms']:.2f} Mpc/h")
    report("RSD P2/P0 over Kaiser, all bands (measurement)", r["kaiser_all_bands"])
    report("RSD P2/P0 over Kaiser, lowest band (gated)", r["kaiser_lowest_band"])
    report(
        "RSD P2 over generalised prediction (measurement)", r["quadrupole_generalised"]
    )
    report(
        "RSD P0_s over generalised prediction (measurement)", r["monopole_generalised"]
    )
    print("\ndensity", r["density"])
    out["catalog"] = {
        k: (v.as_dict() if isinstance(v, gates.Ensemble) else v) for k, v in r.items()
    }
    print(f"[catalog {time.perf_counter() - t:.0f} s]")

    t = time.perf_counter()
    e = gates.gate_kaiser_linear_limit(
        spectrum,
        b.b,
        b.f,
        Box(args.n // 2, args.L / 2),
        Box(args.n, args.L / 2),
        0.1,
        range(6000, 6000 + args.seeds),
    )
    report("Kaiser linear limit (P_in x 1e-2): P2/P0 over Kaiser", e)
    out["kaiser_linear_limit"] = e.as_dict()
    print(f"[Kaiser linear limit {time.perf_counter() - t:.0f} s]")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = Path(args.out) / f"{stamp}_n{args.n}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
