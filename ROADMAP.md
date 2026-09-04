# logunusual roadmap (master plan)

This file fixes scope and the acceptance gate of each milestone. Each milestone gets its
own detailed plan (plan mode) in the session that opens it; file-level steps planned here
would go stale. Gates are derived from identities, self-convergence or Poisson statistics,
not picked, and are not relaxed without explicit approval.

## Why

Measured 2026-09-03 on the M4 Max, Julia reference at production settings:

| stage | bin 1 (192^3) | bin 5 (512^3) |
|---|---|---|
| field (FFTs, P_G, exp, velocities) | 1.5 s | 8 s |
| galaxy draw (single-threaded loop) | 33 s | 104 s |
| constraint randoms at 10x nbar | 30 s | 64 s |
| P(k) estimator | 6.5 s | 35 s |
| total wall / peak RSS | 95 s / 35 GB | 255 s / 75 GB |

The field stage is ~3% of wall. The draw is per-galaxy work that is really per-cell work.
The randoms and estimator are not consumed by the catalog product and carry ~50 GB of the
peak. A vectorised generator should do a 7-bin realization on the laptop in minutes at
10-15 GB and run unchanged on CUDA. The same interface then takes a nonlinear input P(k)
and f_NL scale-dependent bias.

## Non-goals

MPI / multi-node; bispectrum or higher-order fidelity beyond what a lognormal field has;
constraint randoms; an in-generator estimator suite; an MLX backend; reproducing the
prod_v2 1.28x over-density.

## M0 -- Bootstrap  [done 2026-09-04]

Repo skeleton (pixi, hatchling, default + gpu envs), `suite.py` (v28 bin table, seed
schedule, constants) with 7 invariant tests, `CLAUDE.md`, this file, `docs/landscape.md`
with citations checked against their abstract pages.

## M1 -- Periodic-box core

**Build:** `pk.py` (TSV loader, power-law extrapolation, P -> xi -> log1p -> P_G by
FFTLog, plus the 3D-FFT route Henry uses for gridded input), `field.py` (coloured Gaussian
field, lognormal transform with box-mean normalisation, sinc^-p voxel-window correction,
displacement field), `sample.py` (per-cell intensity, Poisson counts, triangular-jitter
placement, 3x3x3 cell-velocity stencil, plane-parallel RSD), `validate.py` (periodic CIC
P(k) monopole/quadrupole with shot subtraction; Poisson self-check). Fast tests on
32^3-64^3 boxes; `slow` marker for the statistical gates. CI (lint + tests) added here.

**Gates:**
- P_G route agreement: FFTLog vs 3D-FFT P_G(k) agree to the FFTLog's own self-convergence
  (halve the log-grid spacing and take the change as the tolerance), and to Henry's
  TwoFAST P_G(k) dumped from Julia for the bin-5 TSV over the k range the grid uses.
- Monopole: P_0(k) / (b^2 P_in(k)) = 1 within the ensemble standard error over
  k < k_Nyq / 2, with the seed count chosen so the SE resolves 2%.
- Quadrupole: P_2 / P_0 equals the Kaiser prediction at the same tolerance
  (plane-parallel).
- Density: realized / target nbar = 1 within Poisson on the box.
- Reproducibility: same seed -> bit-identical catalog on CPU across two runs; CPU vs CUDA
  agreement recorded in ULPs, not asserted bitwise.
- Memory: peak *allocation* (not RSS) of the 512^3 field stage measured and recorded with
  the grid count it corresponds to.

## M2 -- Shell product and drop-in integration

**Build:** `shell.py` (shell padded by 150 Mpc/h, HEALPix NSIDE-128 NESTED int8 mask at
cell level, observer at the origin, radial RSD), `io.py` (streamed parquet, pinned 2^20
row groups, bin-contiguous concatenation, provenance metadata), the 7-bin driver in
`suite.py` with the prod_v2 seed schedule, `cli.py`.

**Gates:**
- Schema test identical to chimera's `tests/test_mock_coverage_thinning.py` fixture:
  `x, y, z: double`, `bin: int8`, bins ascending and contiguous, row groups byte-contiguous
  per bin.
- `pk_bin_patches.py` runs unmodified on a logunusual realization: shell nbar / nominal =
  1.00 within Poisson in every bin; T^2(k) shape consistent with the Julia bin-5 curve
  (same grid ceiling); its Poisson self-check passes.
- `diagnose_realization.py` passes; `contaminate_catalog.py` + `run_field_level.py` run
  end to end on one realization.
- The single chimera PR of this milestone: `--nbar-overdensity-factor` becomes
  metadata-driven (default 1.0 when `realized_nbar` is stamped), and the three copies of
  the v28 table import `logunusual.suite`.

## M3 -- Performance and memory

Per-bin wall and peak allocation on the laptop, on deneb's RTX 3050 (8 GB: float32 fields
or slab FFTs), and one TACC GPU node, tabulated against the Julia baseline above. Targets
are measured, not asserted: a 512^3 bin under 15 GB; a 7-bin realization on the laptop in
single-digit minutes with parquet I/O as the floor. Any regression-guarding number lives
in a test that states what it measured and on which machine.

## M4 -- Physics upgrades (each opens its own plan session)

- **Nonlinear input P(k):** halofit TSVs (a flag in `make_matter_power.py`) together
  with the 1024^3 grid the finer scales need; at 512^3 halofit alone changes little
  because the grid Nyquist binds first.
- **f_NL scale-dependent bias in the input galaxy P(k):**
  `b(k) = b + 2 (b - p) f_NL delta_c / M(k)` with the Poisson factor `M(k)` (copy from
  disco-mocks `cosmology.poisson_M`). Gate: low-k P_0(f_NL) / P_0(0) matches the injected
  ratio; the P_G inversion stays non-negative where the grid uses it. Scope: two-point
  only (null tests, covariance shape); a lognormal is not an f_NL bispectrum mock.
- On demand only: Finger-of-God scatter (`sigma_psi`), deterministic-count sampling
  (`minimize_shotnoise`), multi-tracer.

## M5 -- Ensemble production

Replace or supplement prod_v2 (100 realizations). Realization count, machine, and whether
prod_v2 is retired are JC's call; not scheduled here.

## Known risks / open questions

- The prod_v2 1.28x over-density is consistent with the LogNormalGalaxies 0.9.4 -> 0.10
  voxel-window-correction default flip (count ratios 1.263 / 1.281 on bins 1 / 5) but not
  proven; a 0.11.0 arm with the correction off would close it. JC's call, also on
  reporting it to Henry.
- prod_v2's growth rates come from astropy Planck18 (Om0 = 0.30966) while its distances
  use Om0 = 0.3153; `suite.py` keeps the literals for drop-in fidelity. Reconciling is an
  M4 decision.
- The input P(k) is linear and truncated at kh = 1; the honest reach of the mocks is set
  by the grid (ell ~150-300 sample-weighted at 512^3), see chimera memory
  `project_lognormal_vs_disco_resolution`.
