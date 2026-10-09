"""Shell-product statistical checks with per-band tables; writes
`runs/gates_shell/<stamp>_n<N>.json`.

    pixi run python scripts/gates_shell.py [--n 128] [--L 1000] [--seeds 16]
        [--nbar 3e-3] [--shell RMIN RMAX BUFFER]

The same `gates` functions back `tests/test_gates_shell.py`. The shell has bin 5's
thickness scaled into the box, placed so a buffer covering the field's requirement fits
under L/2 (`--shell`); the mask is an equatorial band, fsky ~ 0.7. Bars:
`docs/validation.md`.
"""

import argparse
import datetime as dt
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

from logunusual import gates, shell, suite  # noqa: E402
from logunusual.grid import Box  # noqa: E402
from logunusual.pk import PowerSpectrum  # noqa: E402


def report(name, e):
    print(f"\n{name}: n_real={e.n_real} passed={e.passed}")
    for k, m, se, z in zip(e.k, e.mean, e.se, e.z):
        print(f"  x={k:.4f}  ratio={m:.4f} +- {se:.4f}  z={z:+.2f}")
    for key, val in e.extra.items():
        print(f"  {key}: {val}")


def band_mask(nside=32, band=0.7):
    import healpy as hp

    m = np.zeros(hp.nside2npix(nside), np.int8)
    th, _ = hp.pix2ang(nside, np.arange(m.size), nest=True)
    m[np.abs(np.cos(th)) < band] = 1
    return shell.AngularMask(values=m, nested=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=float, default=1000.0)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seeds-response", type=int, default=64)
    ap.add_argument(
        "--band-seeds",
        type=int,
        default=32,
        help="seed count the 1x-mesh bands are sized for",
    )
    ap.add_argument("--nbar", type=float, default=3e-3)
    ap.add_argument("--bin", type=int, default=5, help="v28 bin whose b, f, TSV")
    ap.add_argument(
        "--shell",
        type=float,
        nargs=3,
        default=(300.0, 371.1, 90.0),
        metavar=("RMIN", "RMAX", "BUFFER"),
        help="shell in the box; the buffer must cover the field's line-of-sight "
        "requirement (the draw raises otherwise)",
    )
    ap.add_argument("--pk", default=None, help="TSV path (default: the bin's file)")
    ap.add_argument("--out", default="runs/gates_shell")
    args = ap.parse_args()

    b = suite.BIN_SUITE_V28[args.bin - 1]
    spectrum = PowerSpectrum.from_tsv(args.pk or b.matterpower_file)
    box = Box(args.n, args.L)
    sh = shell.Shell(*args.shell)
    sh.check_box(box)
    mask = band_mask()
    print(f"generator {box}  shell {sh}  b={b.b} f={b.f:.4f} nbar={args.nbar}")
    print(f"host {platform.node()} {platform.machine()} {platform.system()}")
    out = {
        "config": vars(args)
        | {
            "b": b.b,
            "f": b.f,
            "shell": [sh.rmin, sh.rmax, sh.buffer],
            "fsky": mask.fsky,
            "pk_hash": spectrum.file_hash,
            "host": platform.node(),
            "machine": platform.machine(),
        }
    }

    t = time.perf_counter()
    r = gates.gate_shell_density(
        spectrum, b.b, b.f, box, sh, mask, args.nbar, range(7000, 7000 + args.seeds)
    )
    report("shell N_kept / (nbar fsky V_shell)", r["total"])
    report("shell n(r) / nbar per sub-shell", r["profile"])
    print("shell draws", r["draws"])
    out["shell_density"] = {
        "total": r["total"].as_dict(),
        "profile": r["profile"].as_dict(),
        "draws": r["draws"],
    }
    print(f"[shell density {time.perf_counter() - t:.0f} s]")

    t = time.perf_counter()
    r = gates.gate_catalog(
        spectrum,
        b.b,
        b.f,
        box,
        box,
        args.nbar,
        range(8000, 8000 + args.seeds_response),
        band_seeds=args.band_seeds,
    )
    report("1x mesh: monopole vs b^2 P_in x response", r["monopole"])
    report("1x mesh: monopole vs fixed-field prediction", r["monopole_fixed_field"])
    out["one_x_mesh_response"] = {
        "monopole": r["monopole"].as_dict(),
        "monopole_fixed_field": r["monopole_fixed_field"].as_dict(),
    }
    print(f"[1x mesh response {time.perf_counter() - t:.0f} s]")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = Path(args.out) / f"{stamp}_n{args.n}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
