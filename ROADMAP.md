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

## M1 -- Periodic-box core  [done 2026-09-04]

**Built:** `grid.py`, `pk.py` (grid-native P -> P_G), `field.py` (JAX, eager),
`sample.py`, `validate.py` (CIC multipoles, Jing shot noise, coherent-alias estimator
response), `gates.py`; 44 fast tests + 5 slow gates (`tests/test_gates_m1.py`), CI
(lint + full suite), `scripts/m1_{gates,reproducibility,memory}.py`. Construction and
the measurements behind it: `CLAUDE.md` "Construction".

**Gates as run (128^3, L = 1000, dx = 7.8, bin-5 b and f, nbar 3e-3, estimator mesh
2x; bands hold >= n_min independent modes so SE <= 2%/3):**

| gate | statistic | band | result (2026-09-04, M4 laptop) |
|---|---|---|---|
| G1 | grid Hankel pair, round trip, 2nd-order expansion, spline nodes/tails | -- | exact (fast tests: atol 1e-10, rtol 1e-12, eps^3 scaling) |
| G2 | Gaussian colouring `<P(G)>/P_G` | all shells, 32^3 x 48 | max |z| < 4.5, mean z^2 = 1.2 (fast test) |
| G3 | lognormal grid identity `<P(delta)>/P_grid`, galaxy and matter | every band to k_Nyq, 192 seeds | max |z| 2.0 (galaxy), 2.2 (matter); SE 0.2-0.5%; 0 clipped modes |
| G4a | uniform catalog / Jing shot, deconvolved | to k_Nyq, 32 seeds | max |z| 2.3; SE 0.1-0.6% |
| G4b | fixed field: catalog / exact prediction incl. coherent aliases | to k_Nyq (and to est. Nyquist), 16 draws | max |z| 3.3 in 52 bands; SE 0.1-0.3% |
| G5 | real-space `(P0 - shot) / (b^2 P_in x response)` | k < k_Nyq/2, 32 seeds | max |z| 2.1; SE 0.3-0.6%; vs fixed-field prediction max |z| 1.9, SE 0.1% |
| G6 | `P2/P0 / Kaiser(f/b)`, P_in x 1e-2 arm (64^3, L 500, nbar 0.1) | k < k_Nyq/2, 32 seeds | max |z| 3.3, SE 2-4%; nonlinear budget 0.4% |
| G6 | same at production amplitude, lowest band | k = 0.035 | 0.995 +- 0.031 (measurement: 1.25x Kaiser by k = 0.19; `P_gm/(b P_mm)` 0.993 -> 0.965) |
| G7 | `sum counts` vs `sum lambda`; `sum lambda / nbar V` | 32 seeds | z mean -0.03, std 0.78; exact to 1e-15 |
| G8 | two processes, same seed | 32^3, 64^3 | identical bytes on macOS-arm64 (asserted on Linux in CI) |
| G9 | peak live JAX bytes, field stage | 128^3, 256^3 | 5.5 x N^3 float64 both (during the matter-field step; 0.69 GiB at 256^3, so 5.5 GiB projected at 512^3); XLA scratch not seen; 512^3 and CUDA in M3 |


**Gate re-derivations (approved by JC 2026-09-04; nothing was loosened, two premises were wrong):**
- *Monopole target.* The catalog has `b^2 P_in` in its first zone, but a CIC estimator
  on a finite mesh sees the lattice-periodic images coherently (`validate.effective_window`,
  exact, separable); the gate compares to `b^2 P_in x estimator_response`, and the
  fixed-field gate G4b verifies that response to the estimator Nyquist.
- *Quadrupole.* "P2/P0 = Kaiser over k < k_Nyq/2 at the same tolerance" assumed linear
  RSD; at production amplitude the mapping is nonlinear (`f Psi_rms` = 3.2 Mpc/h, so
  `k f Psi` = 0.6 at k = 0.19) and the galaxy-matter correlation of the lognormal pair is
  below 1 at finite k. Kaiser is asserted in a scaled-amplitude arm (`P_in x 1e-2`, where
  the nonlinear term is budgeted at `(k f Psi_rms)^2`), in the lowest production band,
  and MEASURED elsewhere (tables in the gate script output).
- *Reproducibility.* Bitwise asserted on Linux (CI); characterised on macOS.
- *CPU vs CUDA in ULPs:* moved to M3 (needs a deneb session; decided 2026-09-04).

## M2 -- Shell product  [done 2026-09-04]

Direction (JC, 2026-09-04): this is a lognormal mock code, not a survey code. The v28
table and a survey mask are default INPUTS; no other repository is touched, depended
on, or used as a gate; the catalog layout is this package's own format spec; every
gate is an identity, a limit, or a Poisson statement.

**Built:** `field.py` (three displacement components, `psi_axes`), `shell.py`
(observer-centred box, cell-level buffered radial window, `AngularMask` for any
HEALPix NSIDE in NESTED or RING, inclusive shell cut, own-cell radial RSD, streamed
`sample_shell`), `io.py` (`CatalogWriter` with pinned row groups that never mix bins,
atomic rename, `check_layout`, `read_bin`), `config.py` (`RunConfig`, YAML, bins
default to the suite, `nbar_scale` / `grid_scale`, `config_hash`), `run.py`
(`generate_realization`, `summary.json`, per-bin metadata), `cli.py` (`run`, `check`,
`default-config`; console script), `configs/v28_default.yaml`; 74 fast tests + 2 slow
gates (`tests/test_gates_m2.py`), `scripts/m2_gates.py`. Construction: `CLAUDE.md`
"Shell product" and "Catalog format".

**Gates (fast, exact):** divergence identity `sum_i k_i Psi_i,k = i delta_k`; the z
path bitwise equal to M1; cell window vs brute force and vs the continuum shell volume
(bounded by the boundary-cell layer); selection identity against an independent
`ang2pix` reference with inclusive edges; radial RSD parallel to `x_hat` with the
projected displacement, `f = 0` identity, on-axis equal to plane-parallel, and the
far-observer limit `|Delta| <= 3 f |Psi| |x| / D` shrinking with `D`; galaxies cross
the shell edges in both directions; **uniform-field Poisson gate**: with `P_in x 1e-4`
the kept count is `Poisson(nbar fsky V_shell)` exactly, `|z| < 4`, through
`sample_shell` and through the whole driver (window, RSD, mask, writer); draws over
the window vs `sum(lambda)`; chunk invariance (bit-identical positions, byte-identical
files); layout self-check, metadata round trip, empty bin; two processes -> identical
bytes (asserted on Linux, characterised on macOS); CLI dry-run writes nothing.

**Slow gates (128^3, L = 1000, dx = 7.8, bin-5 b and f, nbar 3e-3; the shell is bin
5 scaled by 1/5: [386.4, 457.5] +- 30, mask an equatorial band fsky 0.70):**

| gate | statistic | band | result (2026-09-04, M4 laptop) |
|---|---|---|---|
| G10 | `N_kept / (nbar fsky V_shell)` at production amplitude | 16 seeds | 0.9964 +- 0.0040 (z = -0.9); SE is the shell-scale sample variance, consistency gate |
| G10 | `n(r) / nbar` in 8 sub-shells incl. both edges | 16 seeds | max |z| 2.5, SE 0.6-0.9%; edge sub-shells 0.9955 +- 0.0075 and 0.9984 +- 0.0092 |
| G10 | draws over the window vs `sum lambda` | 16 seeds | z mean +0.03, std 0.98; 3 galaxies left the box (all beyond rmax + buffer, never kept) |
| G11 | 1x mesh: `(P0 - shot) / (b^2 P_in x response)` | k < k_Nyq/2, 64 seeds on the 32-seed bands of G5 | max |z| 2.0; SE 0.2-0.5% (with bands re-derived per seed count, 32/48/64 seeds all left one band at 0.68-0.74% SE: the lognormal scatter is super-Gaussian, so `band_seeds` fixes the bands and the seeds bring the SE down) |
| G11 | 1x mesh: vs the fixed-field prediction | k < k_Nyq/2, 64 seeds | max |z| 2.5; SE 0.08-0.11% (|z| up to 43 before the sign fix) |

**Finding (G11).** M1's coherent-alias formula omitted the sign `(-1)^(r n)` that the
half-cell offset of the cell centres puts on alias image `n` of a mesh with ratio `r`.
Every sign is + for an even ratio, so the 2x gates (G4b, G5) could not see it; on the
generator's own mesh the dominant image subtracts and the deconvolved power reads 6%
low at half the Nyquist (shell average; 11% along an axis). With the sign the
fixed-field arm went from |z| up to 43 to max 2.8. `validate.effective_window` carries
the sign; pinned by a fast test against the hand sum.

**Acceptance run (2026-09-04, M4 Max laptop, CPU, `runs/v28_full.yaml`: the seven-bin
default at full density with the survey mask, realization 0):** 651,427,904 galaxies,
15.04 GiB parquet, **187 s wall**; host peak RSS 19.3 GB (`/usr/bin/time -l`), peak
live JAX bytes 7.0 GiB = **7.0 x N^3 float64** in every bin (512^3 bins; delta_g +
delta_m_k + three Psi + FFT scratch). `logunusual check`: layout ok, realized/target
1.0077, 1.0018, 0.9988, 0.9991, 1.0004, 0.9993, 1.0011 for bins 1-7. Per bin (field /
sample): 192^3 1.0 / 9.2 s; 256^3 1.5 / 27.8 s; 384^3 3.8 / 24.6 s; 448^3 6.0 / 29.2 s;
512^3 8.2 / 30.7 s, 8.0 / 21.8 s, 7.9 / 7.2 s. The sample stage (Poisson draw,
placement, radial RSD, mask lookup, parquet) is 80% of the wall; 1.55e9 galaxies drawn
for 6.5e8 kept (the full-sky buffered window vs the masked shell). Against the Julia
reference at bin 5 (255 s / 75 GB for one bin): 39 s / 7 GiB live. Bins 1-2 had 87 and
40 galaxies leave the box (beyond `rmax + buffer`, never kept); the others 0.

**Not gated in M2:** the redshift-space statistics of the radial mapping (a
wide-angle estimator is not part of the package); the geometry is exact by the tests
above and the plane-parallel physics is M1 G6.

## M3 -- Performance and memory

Per-bin wall and peak allocation on the laptop, on deneb's RTX 3050 (6 GB: float32 fields
or slab FFTs), and one TACC GPU node, tabulated against the Julia baseline above. CPU vs
CUDA agreement of the field stage recorded in ULPs (moved here from M1); `jax.jit` of
the field stage with a bitwise-before-jit check. Targets
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
  voxel-window-correction default flip (count ratios 1.263 / 1.281 on bins 1 / 5), and
  M1 measured the mechanism: the post-transform `sinc^-2` deconvolution leaves 27% of the
  expected galaxies in cells with `1 + delta < 0` at bin-5 settings. Not proven on
  prod_v2 itself; a 0.11.0 arm with the correction off would close it. JC's call, also
  on reporting it to Henry.
- Attainability of the deconvolved target (`P/sinc^2`, uniform placement) for the v28
  bins at their production grids (measured 2026-09-04, `grid_pkG` diagnostics): bins 1-6
  zero clipped modes, `xi_min` > -0.004; bin 7 (b = 3.29, dx 15.6) clips 5 of 67M modes
  carrying 1e-8 of the power. Grid sigma^2 of the galaxy field 2.3-3.3, matter 0.26-2.8.
- prod_v2's growth rates come from astropy Planck18 (Om0 = 0.30966) while its distances
  use Om0 = 0.3153; `suite.py` keeps the literals for drop-in fidelity. Reconciling is an
  M4 decision.
- The input P(k) is linear and truncated at kh = 1; the honest reach of the mocks is set
  by the grid (ell ~150-300 sample-weighted at 512^3), see chimera memory
  `project_lognormal_vs_disco_resolution`.
