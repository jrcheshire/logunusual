"""M3: device memory, CPU-vs-CUDA agreement and per-stage wall of the field stage.

    pixi run -e gpu python scripts/device.py ladder
    pixi run -e gpu python scripts/device.py ladder --bins bin05 --dtype f32
    pixi run -e gpu python scripts/device.py ulp --n 64 128 256
    pixi run -e gpu python scripts/device.py wall --bin bin05

`ladder` walks the v28 bins at their production `(N_grid, L_box, b)` -- bin01..bin05
cover every distinct grid size, since bin06 and bin07 also run at 512^3 and a
footprint is set by shape, not by which spectrum is on the grid. It runs one
subprocess per point (a JAX process keeps no resettable peak counter) and reports TWO
instruments: live `jax.live_arrays()` bytes, the M1 G9 instrument, and the device
allocator's `peak_bytes_in_use`, which also counts XLA intra-op scratch and is
therefore what decides whether a grid fits. An out-of-memory point is a RESULT: the
child reports `oom` and exits 0, and only a non-OOM failure is an error.

`--dtype f32` runs the SHIPPED field stage at `dtype="f32"` (`field.resolve_dtype`).
It replaced a replica of the stage's array sequence, `probe_field_arrays`, which had
to earn its number with a self-check against the real f64 peak and failed it (1.084 at
256^3, job 1958); a knob in `field.py` needs no such licence.

`ulp` compares the two backends inside ONE process, so a single jaxlib build is on
both sides and the difference is the backend, not the wheel.

`wall` blocks on every live array at each step boundary, which serialises JAX's async
dispatch: the per-stage numbers are upper bounds and their sum exceeds the unblocked
end-to-end time, which is reported alongside.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

FIXTURE_PK = "tests/data/matterpower_camb_zeff=0.9.tsv"
GIB = 2**30


def resolve_pk(bin_):
    """`data/<pk_file>` if the checkout has it, else the committed bin05 fixture.

    `data/` is gitignored, so a fresh clone carries only the z_eff = 0.9 spectrum.
    A footprint is set by shape, so the fallback is sound for `ladder`; `wall`
    prints which file it used.
    """
    for path in (f"data/{bin_.pk_file}", f"tests/data/{bin_.pk_file}"):
        if os.path.exists(path):
            return path
    return FIXTURE_PK


# --------------------------------------------------------------------------- ladder

LADDER_CHILD = r"""
import os, sys, json, resource, threading, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import jax.numpy as jnp
import numpy as np
from logunusual import field
from logunusual.grid import Box, k_components
from logunusual.pk import PowerSpectrum, grid_pkG

n, L, pk_file, dtype = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
B, name, use_jit = float(sys.argv[5]), sys.argv[6], sys.argv[7] == "jit"
spectrum = PowerSpectrum.from_tsv(pk_file)


def live_bytes():
    return sum(int(a.nbytes) for a in jax.live_arrays())


def device_peak():
    try:
        s = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return None, None
    return s.get("peak_bytes_in_use"), s.get("bytes_limit")


class Peak:
    def __init__(self):
        self.label, self.peak, self.peak_label, self.stop = "start", 0, "start", False
        self.trace = []

    def cb(self, label):
        self.trace.append((label, live_bytes()))
        self.label = label

    def poll(self):
        while not self.stop:
            b = live_bytes()
            if b > self.peak:
                self.peak, self.peak_label = b, "after '" + self.label + "'"
            time.sleep(0.001)


report = {
    "bin": name, "n": n, "dtype": dtype, "L": L, "b": B, "pk": pk_file,
    "jit": use_jit,
}
try:
    # Warm-up at a small N: the CUDA context allocates. Under jit this does NOT warm
    # the compilation -- the shape is different, so the measured leg compiles its own.
    field.generate_fields(spectrum, B, Box(32, L), 0, jit=use_jit)
    report["warmup_device_peak_bytes"] = device_peak()[0]

    def measure(fn, label):
        '''Peak live JAX bytes over one leg, in its own poller.'''
        pk = Peak()
        pk.label = label
        th = threading.Thread(target=pk.poll, daemon=True)
        th.start()
        try:
            res = fn(pk)
            jax.block_until_ready(res)
        finally:
            pk.stop = True
            th.join()
        return pk

    # One point per process: `peak_bytes_in_use` is a process-wide high-water mark
    # with no reset, so two dtypes in one process could only give a one-sided bound.
    def body(pk):
        F = field.generate_fields(
            spectrum, B, Box(n, L), 1,
            keep_matter=True, psi_axes="xyz", dtype=dtype, jit=use_jit, trace=pk.cb,
        )
        assert F.delta_g.dtype == (jnp.float32 if dtype == "f32" else jnp.float64)
        return [F.delta_g, F.delta_m] + list(F.psi.values())

    p = measure(body, dtype)

    dp, limit = device_peak()
    report.update(
        ok=True,
        oom=False,
        live_peak_bytes=p.peak,
        live_peak_after=p.peak_label,
        device_peak_bytes=dp,
        device_bytes_limit=limit,
        trace=[(l, b) for l, b in p.trace],
    )
except Exception as exc:
    text = f"{type(exc).__name__}: {exc}"
    is_oom = ("RESOURCE_EXHAUSTED" in text) or ("out of memory" in text.lower())
    dp, limit = device_peak()
    report.update(
        ok=not is_oom,
        oom=is_oom,
        error=text[:600],
        device_peak_bytes=dp,
        device_bytes_limit=limit,
    )

report["cell_bytes_f64"] = n**3 * 8
report["ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
    1 if sys.platform == "darwin" else 1024
)
report["device"] = str(jax.devices()[0])
print("@@JSON@@" + json.dumps(report))
sys.exit(0 if (report["ok"] or report["oom"]) else 1)
"""


def run_ladder(args):
    from logunusual import suite

    by_name = {b.name: b for b in suite.BIN_SUITE_V28}
    bins = [by_name[nm] for nm in args.bins]

    def child(n, b, pk, dt, jit):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                LADDER_CHILD,
                str(n),
                str(b.L_box),
                pk,
                dt,
                str(b.b),
                b.name,
                "jit" if jit else "nojit",
            ],
            capture_output=True,
            text=True,
        )
        got = [x for x in proc.stdout.splitlines() if x.startswith("@@JSON@@")]
        return (json.loads(got[-1][len("@@JSON@@") :]) if got else None), proc

    rows = []
    for b in bins:
        for dt in args.dtype:
            n, pk = b.N_grid, resolve_pk(b)
            rep, proc = child(n, b, pk, dt, args.jit)
            if rep is None:
                print(
                    f"{b.name} N = {n} [{dt}]: child produced no report "
                    f"(exit {proc.returncode})"
                )
                print(proc.stderr[-2000:])
                continue
            rows.append(rep)
            n3 = rep["cell_bytes_f64"]
            if rep.get("oom"):
                lim = rep.get("device_bytes_limit")
                tail = f" -- allocator limit {lim / GIB:.2f} GiB" if lim else ""
                print(f"\n{b.name} N = {n} [{dt}]: OUT OF MEMORY (a result){tail}")
                print(f"   {rep['error'].splitlines()[0][:200]}")
                continue
            if not rep.get("ok"):
                print(
                    f"\n{b.name} N = {n} [{dt}]: FAILED (not OOM) -- "
                    f"{rep.get('error', '')[:300]}"
                )
                continue
            dp = rep.get("device_peak_bytes")
            lv = rep.get("live_peak_bytes", 0)
            tag = f"{dt}, jit" if args.jit else dt
            print(f"\n{b.name}  N = {n}  L = {b.L_box:g}  [{tag}]  {rep['device']}")
            print(
                f"   live JAX arrays  {lv / GIB:6.2f} GiB  "
                f"({lv / n3:.2f} x N^3 f64)  {rep.get('live_peak_after', '')}"
            )
            if dp:
                print(
                    f"   device allocator {dp / GIB:6.2f} GiB  "
                    f"({dp / n3:.2f} x N^3 f64)   <- decides the fit"
                )
                if rep.get("device_bytes_limit"):
                    print(
                        f"   allocator limit  "
                        f"{rep['device_bytes_limit'] / GIB:6.2f} GiB"
                    )
            print(f"   host ru_maxrss   {rep['ru_maxrss_bytes'] / GIB:6.2f} GiB")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.out}")


# ------------------------------------------------------------------------------ ulp


def ulp_distance(a, b):
    """Elementwise |a - b| in units of the local ULP, at the larger magnitude."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.spacing(np.maximum(np.abs(a), np.abs(b)))
    scale = np.where(scale == 0, np.spacing(np.float64(0)), scale)
    return np.abs(a - b) / scale


def run_ulp(args):
    import jax

    from logunusual import field
    from logunusual.grid import Box
    from logunusual.pk import PowerSpectrum

    spectrum = PowerSpectrum.from_tsv(args.pk)
    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("no CUDA device visible to JAX -- ulp mode needs both backends")
        return
    print(f"CPU: {cpu}   CUDA: {gpu}   jax {jax.__version__}\n")

    rows = []
    for n in args.n:
        out = {}
        for name, dev in (("cpu", cpu), ("gpu", gpu)):
            with jax.default_device(dev):
                F = field.generate_fields(
                    spectrum, 1.76, Box(n, args.L), 1, keep_matter=True, psi_axes="xyz"
                )
                out[name] = {
                    "delta_g": np.asarray(F.delta_g),
                    "delta_m": np.asarray(F.delta_m),
                    **{f"psi_{a}": np.asarray(F.psi[a]) for a in "xyz"},
                }
                del F
        row = {"n": n, "fields": {}}
        print(f"N = {n}")
        for key in out["cpu"]:
            a, b = out["cpu"][key], out["gpu"][key]
            u = ulp_distance(a, b)
            rms = float(np.sqrt(np.mean(a**2)))
            rec = {
                "max_ulp": float(u.max()),
                "median_ulp": float(np.median(u)),
                "p99_ulp": float(np.percentile(u, 99)),
                "max_abs": float(np.abs(a - b).max()),
                "rms_of_field": rms,
                "bitwise": bool(np.array_equal(a, b)),
            }
            row["fields"][key] = rec
            print(
                f"   {key:8s} max {rec['max_ulp']:10.1f} ulp   p99 "
                f"{rec['p99_ulp']:6.1f}   median {rec['median_ulp']:5.1f}   "
                f"max|d| {rec['max_abs']:.3e}  (field rms {rms:.3e})"
                + ("   BITWISE" if rec["bitwise"] else "")
            )
        rows.append(row)
        print()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.out}")


# ----------------------------------------------------------------------------- wall


def run_wall(args):
    import time

    import jax

    from logunusual import field, suite
    from logunusual.grid import Box
    from logunusual.pk import PowerSpectrum

    b = {x.name: x for x in suite.BIN_SUITE_V28}[args.bin]
    pk = args.pk or resolve_pk(b)
    spectrum = PowerSpectrum.from_tsv(pk)
    box = Box(b.N_grid, b.L_box)
    print(
        f"{b.name}: N = {b.N_grid}, L = {b.L_box} Mpc/h, b = {b.b}, z_eff = {b.z_eff}"
    )
    print(f"device {jax.devices()[0]}   pk {pk}   jit {args.jit}\n")

    field.generate_fields(spectrum, b.b, Box(32, b.L_box), 0, jit=args.jit)  # warm-up

    marks = []

    def cb(label):
        jax.block_until_ready(list(jax.live_arrays()))
        marks.append((label, time.perf_counter()))

    t0 = time.perf_counter()
    F = field.generate_fields(
        spectrum, b.b, box, 1, keep_matter=True, psi_axes="xyz", jit=args.jit, trace=cb
    )
    jax.block_until_ready([F.delta_g, F.delta_m] + list(F.psi.values()))
    t_blocked = time.perf_counter() - t0

    stages = []
    prev = t0
    for label, t in marks:
        stages.append((label, t - prev))
        print(f"   {label:12s} {t - prev:7.2f} s")
        prev = t
    print(f"\n   field stage, serialised by the per-step blocks: {t_blocked:.2f} s")

    t1 = time.perf_counter()
    F2 = field.generate_fields(
        spectrum, b.b, box, 2, keep_matter=True, psi_axes="xyz", jit=args.jit
    )
    jax.block_until_ready([F2.delta_g, F2.delta_m] + list(F2.psi.values()))
    t_free = time.perf_counter() - t1
    print(f"   field stage, unblocked end to end:              {t_free:.2f} s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "bin": b.name,
                    "n": b.N_grid,
                    "L": b.L_box,
                    "jit": args.jit,
                    "stages": stages,
                    "blocked_s": t_blocked,
                    "unblocked_s": t_free,
                    "device": str(jax.devices()[0]),
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.out}")


# ------------------------------------------------------------------------------ pkg


def run_pkg(args):
    """Split one `target_on_grid` + `grid_pkG` call into host work, upload, FFTs and
    device syncs.

    The pieces are the REAL library calls in the order `field.generate_fields` makes
    them: `radius_index`, `pk_on_grid`'s table + gather, `jitter_power_window`,
    `grid_xi`, `grid_pk_from_xi` are each timed where they are called. The assembled
    result is then compared bitwise against `grid_pkG(target_on_grid(...))` in the
    same process, so a split that has drifted from the shipped path cannot be reported
    as one.
    """
    import time

    import jax
    import jax.numpy as jnp

    from logunusual import field, suite
    from logunusual.grid import Box, jitter_power_window
    from logunusual.pk import (
        PowerSpectrum,
        grid_pk_from_xi,
        grid_xi,
        grid_pkG,
        radius_index,
    )

    b = {x.name: x for x in suite.BIN_SUITE_V28}[args.bin]
    pk_file = args.pk or resolve_pk(b)
    spectrum = PowerSpectrum.from_tsv(pk_file)
    box = Box(b.N_grid, b.L_box)
    jitter_p = args.jitter_p
    print(
        f"{b.name}: N = {b.N_grid}, L = {b.L_box} Mpc/h, b = {b.b}, z_eff = {b.z_eff}"
    )
    print(f"device {jax.devices()[0]}   pk {pk_file}   jitter_p {jitter_p}\n")

    # Warm-up: compilation, the CUDA context and the spline's first call all allocate.
    field.generate_fields(spectrum, b.b, Box(32, b.L_box), 0)

    arms = {"pkG_g": (lambda k: b.b * b.b * spectrum(k)), "pkG_m": spectrum}

    def split_once(target):
        """One timed pass. Returns `(pieces, pkG, diag)`; `pieces` is a list of
        `(label, kind, seconds)` with kind in host / upload / device / sync."""
        pieces = []
        t = [time.perf_counter()]

        def mark(label, kind, result=None):
            if kind in ("upload", "device"):
                jax.block_until_ready(result)
            now = time.perf_counter()
            pieces.append((label, kind, now - t[0]))
            t[0] = now

        # --- host: field.target_on_grid == pk_on_grid / jitter_power_window
        q = radius_index(box)
        mark("radius_index", "host")
        n2 = box.n_mesh // 2
        table = target(box.k_f * np.sqrt(np.arange(3 * n2 * n2 + 1, dtype=np.float64)))
        table = np.asarray(table, dtype=np.float64)
        mark("spline_table", "host")
        P = table[q]
        P[0, 0, 0] = 0.0
        mark("gather", "host")
        del q, table
        if jitter_p:
            W = jitter_power_window(box, jitter_p)
            mark("jitter_window", "host")
            P = P / W
            del W
            mark("window_divide", "host")

        # --- device: pk.grid_pkG
        Pd = jnp.asarray(P)
        mark("upload", "upload", Pd)
        del P
        xi = grid_xi(Pd, box, jnp)
        mark("irfftn_xi", "device", xi)
        del Pd
        xi_min = float(xi.min())
        sigma2 = float(xi[0, 0, 0])
        mark("sync_xi_min_sigma2", "sync")
        if xi_min <= -1.0:
            raise ValueError(
                f"xi(x) reaches {xi_min} <= -1: no lognormal has this P(k)"
            )
        xiG = jnp.log1p(xi)
        mark("log1p", "device", xiG)
        del xi
        pkG = grid_pk_from_xi(xiG, box, jnp)
        mark("rfftn_pkG", "device", pkG)
        del xiG
        pkG = pkG.at[0, 0, 0].set(0.0)
        neg = pkG < 0
        mark("dc_and_compare", "device", (pkG, neg))
        n_clipped = int(neg.sum())
        neg_power = float(jnp.where(neg, -pkG, 0.0).sum())
        pos_power = float(jnp.where(neg, 0.0, pkG).sum())
        mark("sync_clip_diagnostics", "sync")
        pkG = jnp.where(neg, 0.0, pkG)
        mark("clip", "device", pkG)
        diag = {
            "xi_min": xi_min,
            "sigma2": sigma2,
            "n_clipped": n_clipped,
            "clipped_power_fraction": neg_power / pos_power if pos_power > 0 else 0.0,
        }
        return pieces, pkG, diag

    rows = []
    for arm, target in arms.items():
        runs = []
        for rep in range(args.repeat):
            pieces, pkG, diag = split_once(target)
            runs.append(pieces)
            total = sum(s for _, _, s in pieces)
            print(f"   {arm} run {rep + 1}/{args.repeat}: {total:7.3f} s")
            if rep == 0:
                # Self-check: the composed split must BE the shipped call, bitwise.
                ref, diag_ref = grid_pkG(
                    field.target_on_grid(target, box, jitter_p), box, jnp
                )
                bitwise = bool(np.array_equal(np.asarray(pkG), np.asarray(ref)))
                diag_same = all(diag[k] == diag_ref[k] for k in diag)
                del ref
                print(
                    "      self-check vs grid_pkG(target_on_grid(...)): "
                    + ("BITWISE" if bitwise else "MISMATCH -- split is VOID")
                    + ("" if diag_same else "; DIAGNOSTICS DIFFER")
                )
            del pkG
        labels = [(lab, kind) for lab, kind, _ in runs[0]]
        med = {
            lab: float(np.median([r[i][2] for r in runs]))
            for i, (lab, kind) in enumerate(labels)
        }
        by_kind = {}
        for lab, kind in labels:
            by_kind[kind] = by_kind.get(kind, 0.0) + med[lab]
        total = sum(med.values())
        print(f"\n   {arm}: {total:.3f} s (median of {args.repeat})")
        for i, (lab, kind) in enumerate(labels):
            vals = [r[i][2] for r in runs]
            print(
                f"      {lab:22s} {kind:6s} {med[lab]:7.3f} s  "
                f"{100 * med[lab] / total:5.1f}%   "
                f"[{min(vals):.3f}, {max(vals):.3f}]"
            )
        for kind in ("host", "upload", "device", "sync"):
            if kind in by_kind:
                print(
                    f"      {'= ' + kind:22s} {'':6s} {by_kind[kind]:7.3f} s  "
                    f"{100 * by_kind[kind] / total:5.1f}%"
                )
        print()
        rows.append(
            {
                "arm": arm,
                "bitwise_self_check": bitwise,
                "diagnostics_match": diag_same,
                "median_s": med,
                "by_kind_s": by_kind,
                "total_s": total,
                "runs": [[list(p) for p in r] for r in runs],
                "diagnostics": diag,
            }
        )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "bin": b.name,
                    "n": b.N_grid,
                    "L": b.L_box,
                    "b": b.b,
                    "jitter_p": jitter_p,
                    "pk": pk_file,
                    "repeat": args.repeat,
                    "device": str(jax.devices()[0]),
                    "arms": rows,
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("ladder", help="peak device memory per production bin")
    p.add_argument(
        "--bins",
        nargs="+",
        default=["bin01", "bin02", "bin03", "bin04", "bin05"],
        help="bin06/bin07 also run at 512^3, so bin05 stands for all three",
    )
    p.add_argument("--dtype", nargs="+", default=["f64"], choices=["f64", "f32"])
    p.add_argument(
        "--jit", action="store_true", help="compile the stage's pure device functions"
    )
    p.add_argument("--out")
    p.set_defaults(func=run_ladder)

    p = sub.add_parser("ulp", help="CPU vs CUDA agreement of the field stage")
    p.add_argument("--n", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--L", type=float, default=5000.0)
    p.add_argument("--pk", default=FIXTURE_PK)
    p.add_argument("--out")
    p.set_defaults(func=run_ulp)

    p = sub.add_parser("wall", help="per-stage wall for one production bin")
    p.add_argument("--bin", default="bin05")
    p.add_argument("--pk")
    p.add_argument("--jit", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=run_wall)

    p = sub.add_parser("pkg", help="split the P -> P_G step into host / FFT / sync")
    p.add_argument("--bin", default="bin05")
    p.add_argument("--pk")
    p.add_argument("--jitter-p", type=int, default=1, dest="jitter_p")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out")
    p.set_defaults(func=run_pkg)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
