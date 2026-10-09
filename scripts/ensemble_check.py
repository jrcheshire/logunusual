"""Integrity and density checks over an ensemble's `summary.json` files.

    pixi run python scripts/ensemble_check.py RUN_DIR [--manifest PATH] [--n 100]
        [--out runs/ensemble/ensemble_check.json]

RUN_DIR holds `realization_NNNNN/summary.json` (the catalogs need not be present);
`--manifest` is the job's `sha256sum` manifest, which gets one line per realization
whose catalog passed `logunusual check`.

Integrity: realizations 0..n-1 all present and no others, one config hash, every
per-bin seed (and ic / draw seed) distinct, the manifest lists exactly those n
catalogs. Density: per bin, the ensemble mean of `realized_over_target` against
`1 + edge`, SE from the scatter over realizations; `edge = 3 sigma_s^2 (rmax - rmin) /
(rmax^3 - rmin^3)` is the second-order excess from the radial RSD shift across a
curved shell (sigma_s = f times the rms of one displacement component). |z| <= 3
passes. Exits 0 only if both pass; bars in `docs/validation.md`.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(run_dir: Path):
    found = sorted(run_dir.glob("realization_*/summary.json"))
    by_r = {}
    for p in found:
        s = json.loads(p.read_text())
        by_r[s["realization"]] = s
    return by_r


def edge_excess(b: dict) -> float:
    sigma_s = b["f"] * math.sqrt(np.mean(np.square(list(b["psi_rms"].values()))))
    rmin, rmax = b["rmin"], b["rmax"]
    return 3.0 * sigma_s**2 * (rmax - rmin) / (rmax**3 - rmin**3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="runs/ensemble/ensemble_check.json")
    args = ap.parse_args()

    by_r = load(Path(args.run_dir))
    want = set(range(args.n))
    missing = sorted(want - set(by_r))
    extra = sorted(set(by_r) - want)
    hashes = {s["config_hash"] for s in by_r.values()}
    seeds = {k: [] for k in ("seed", "ic_seed", "draw_seed")}
    for s in by_r.values():
        for b in s["bins"]:
            for k in seeds:
                seeds[k].append(b[k])
    n_bins_total = sum(len(s["bins"]) for s in by_r.values())
    seeds_ok = all(len(set(v)) == n_bins_total for v in seeds.values())

    manifest_ok = None
    if args.manifest:
        names = [ln.split()[-1] for ln in Path(args.manifest).read_text().splitlines()]
        listed = sorted(int(x.split("realization_")[1][:5]) for x in names)
        manifest_ok = listed == sorted(want)
        print(
            f"manifest: {len(names)} entries, exactly realizations 0..{args.n - 1}:"
            f" {manifest_ok}"
        )

    integrity_ok = not missing and not extra and len(hashes) == 1 and seeds_ok
    integrity_ok = integrity_ok and manifest_ok is not False
    print(
        f"realizations: {len(by_r)} found, missing {missing or 'none'}, "
        f"extra {extra or 'none'}"
    )
    print(f"config hashes: {sorted(h[:12] for h in hashes)}")
    print(f"seeds distinct over {n_bins_total} bin-realizations: {seeds_ok}")
    print(f"integrity: {'PASS' if integrity_ok else 'FAIL'}")

    rows, density_ok = [], True
    print(
        f"\n{'bin':>3} {'N mean':>13} {'N sd':>10} {'ratio mean':>11} {'SE':>8} "
        f"{'edge':>8} {'z':>6} {'psi_max':>8} {'req/buf':>8} {'clipped':>9}"
    )
    for idx in sorted({b["index"] for s in by_r.values() for b in s["bins"]}):
        bs = [b for s in by_r.values() for b in s["bins"] if b["index"] == idx]
        ratio = np.array([b["realized_over_target"] for b in bs])
        ngal = np.array([b["n_galaxies"] for b in bs], dtype=float)
        se = ratio.std(ddof=1) / math.sqrt(len(ratio))
        edge = float(np.mean([edge_excess(b) for b in bs]))
        z = float((ratio.mean() - 1.0 - edge) / se)
        density_ok = density_ok and bool(abs(z) <= 3.0)
        req = max(b["required_buffer"] / b["radial_buffer"] for b in bs)
        row = dict(
            index=idx,
            n=len(bs),
            n_galaxies_mean=ngal.mean(),
            n_galaxies_sd=ngal.std(ddof=1),
            ratio_mean=ratio.mean(),
            ratio_se=se,
            edge=edge,
            z=z,
            psi_max=max(b["psi_max"] for b in bs),
            required_over_buffer=req,
            clipped_power_fraction=max(b["clipped_power_fraction"] for b in bs),
        )
        rows.append(row)
        print(
            f"{idx:>3} {row['n_galaxies_mean']:>13,.0f} {row['n_galaxies_sd']:>10,.0f} "
            f"{row['ratio_mean']:>11.5f} {se:>8.5f} {edge:>8.1e} {z:>6.2f} "
            f"{row['psi_max']:>8.1f} {req:>8.3f} {row['clipped_power_fraction']:>9.1e}"
        )
    print(f"density: {'PASS' if density_ok else 'FAIL'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                run_dir=str(args.run_dir),
                n=args.n,
                missing=missing,
                extra=extra,
                config_hashes=sorted(hashes),
                seeds_distinct=seeds_ok,
                manifest_ok=manifest_ok,
                integrity_ok=integrity_ok,
                density_ok=density_ok,
                bins=rows,
            ),
            indent=1,
        )
    )
    print(f"\nwrote {out}")
    raise SystemExit(0 if integrity_ok and density_ok else 1)


if __name__ == "__main__":
    main()
