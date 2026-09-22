"""Run the M1 statistical gates with full tables and write `runs/m1_gates/<stamp>.json`.

    pixi run python scripts/m1_gates.py [--n 128] [--L 1000] [--seeds 48] [--nbar 3e-3]

The same functions back `tests/test_gates_m1.py`; this script exists to run bigger
configurations and keep the numbers (the JSON is gitignored; quote them in the PR).
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
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument(
        "--band-seeds",
        type=int,
        default=None,
        help="seed count the BANDS are designed for (default: --seeds); more seeds "
        "than this through the same bands is how a super-Gaussian scatter clears "
        "the SE floor",
    )
    ap.add_argument("--seeds-field", type=int, default=192)
    ap.add_argument("--nbar", type=float, default=3e-3)
    ap.add_argument(
        "--bin", type=int, default=5, help="v28 bin whose b, f, TSV are used"
    )
    ap.add_argument("--pk", default=None, help="TSV path (default: the bin's file)")
    ap.add_argument("--out", default="runs/m1_gates")
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
    report("G3 galaxy grid identity", r["galaxy"])
    report("G3 matter grid identity", r["matter"])
    out["G3"] = {k: v.as_dict() for k, v in r.items()}
    print(f"[G3 {time.perf_counter() - t:.0f} s]")

    t = time.perf_counter()
    e = gates.gate_uniform_shot(
        box,
        box_est,
        args.nbar,
        range(4000, 4000 + args.seeds),
        band_seeds=args.band_seeds,
    )
    report("G4a uniform shot vs Jing", e)
    out["G4a"] = e.as_dict()
    F = field.generate_fields(spectrum, b.b, box, 4100, rsd=False)
    r = gates.gate_fixed_field_sampler(F, args.nbar, range(4200, 4200 + 16), box_est)
    report("G4b fixed field, first zone", r["first_zone"])
    report("G4b fixed field, to estimator Nyquist", r["to_estimator_nyquist"])
    out["G4b"] = {k: v.as_dict() for k, v in r.items()}
    print(f"[G4 {time.perf_counter() - t:.0f} s]")

    t = time.perf_counter()
    r = gates.gate_catalog(
        spectrum, b.b, b.f, box, box_est, args.nbar, range(5000, 5000 + args.seeds)
    )
    report("G5 monopole vs b^2 P_in x estimator response", r["monopole"])
    report("G5 monopole vs fixed-field prediction", r["monopole_fixed_field"])
    report("G6 premise P_gm/(b P_mm) (measurement)", r["premise"])
    print(f"  f Psi_rms = {r['f_psi_rms']:.2f} Mpc/h")
    report("G6 P2/P0 over Kaiser, all bands (measurement)", r["kaiser_all_bands"])
    report("G6 P2/P0 over Kaiser, lowest band (gated)", r["kaiser_lowest_band"])
    report(
        "G6 P2 over generalised prediction (measurement)", r["quadrupole_generalised"]
    )
    report(
        "G6 P0_s over generalised prediction (measurement)", r["monopole_generalised"]
    )
    print("\nG7 density", r["density"])
    out["G5_G6_G7"] = {
        k: (v.as_dict() if isinstance(v, gates.Ensemble) else v) for k, v in r.items()
    }
    print(f"[G5-7 {time.perf_counter() - t:.0f} s]")

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
    report("G6 linear limit (P_in x 1e-2): P2/P0 over Kaiser", e)
    out["G6_linear_limit"] = e.as_dict()
    print(f"[G6 linear limit {time.perf_counter() - t:.0f} s]")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = Path(args.out) / f"{stamp}_n{args.n}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
