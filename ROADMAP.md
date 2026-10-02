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

**Slow gates (128^3, L = 1000, dx = 7.8, bin-5 b and f, nbar 3e-3; the shell was bin
5 scaled by 1/5, [386.4, 457.5] +- 30, mask an equatorial band fsky 0.70; since
2026-09-20 the shell is [300, 371.1] +- 90, see the note under the table):**

| gate | statistic | band | result (2026-09-04, M4 laptop) |
|---|---|---|---|
| G10 | `N_kept / (nbar fsky V_shell)` at production amplitude | 16 seeds | 0.9964 +- 0.0040 (z = -0.9); SE is the shell-scale sample variance, consistency gate |
| G10 | `n(r) / nbar` in 8 sub-shells incl. both edges | 16 seeds | max |z| 2.5, SE 0.6-0.9%; edge sub-shells 0.9955 +- 0.0075 and 0.9984 +- 0.0092 |
| G10 | draws over the window vs `sum lambda` | 16 seeds | z mean +0.03, std 0.98; 3 galaxies left the box (all beyond rmax + buffer, never kept) |
| G11 | 1x mesh: `(P0 - shot) / (b^2 P_in x response)` | k < k_Nyq/2, 64 seeds on the 32-seed bands of G5 | max |z| 2.0; SE 0.2-0.5% (with bands re-derived per seed count, 32/48/64 seeds all left one band at 0.68-0.74% SE: the lognormal scatter is super-Gaussian, so `band_seeds` fixes the bands and the seeds bring the SE down) |
| G11 | 1x mesh: vs the fixed-field prediction | k < k_Nyq/2, 64 seeds | max |z| 2.5; SE 0.08-0.11% (|z| up to 43 before the sign fix) |

**G10 re-run, 2026-09-20 (per-slab streams, buffer guard, angular pre-cut; shell
[300, 371.1] +- 90).** The M3 buffer guard (`Shell.required_buffer` = half diagonal
+ `f max|Psi|` over the drawn cells) measured 40-68 Mpc/h across 16 seeds at this
grid, so the 30 Mpc/h the scaled production buffer gave was not sufficient, and only
42.5 fit between `rmax` and `L/2`: the shell moved inward with the same thickness and
a 90 Mpc/h buffer. Results: total 1.0085 +- 0.0063 (z +1.35); profile max |z| 2.3
(two adjacent outer sub-shells at 1.024 and 1.016, the same realizations in both);
draws z mean -0.03, std 1.20; nothing left the box. The 2026-09-04 rows above stand as
measured under the old geometry; nothing in them is re-asserted.

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

## M3 -- Performance and memory  [done 2026-10-01]

Per-bin wall and peak allocation on the laptop, on deneb's RTX 3050 (6 GB: float32 fields
or slab FFTs), and one TACC GPU node, tabulated against the Julia baseline above. CPU vs
CUDA agreement of the field stage recorded in ULPs (moved here from M1); `jax.jit` of
the field stage with a bitwise-before-jit check. Targets
are measured, not asserted: a 512^3 bin under 15 GB; a 7-bin realization on the laptop in
single-digit minutes with parquet I/O as the floor. Any regression-guarding number lives
in a test that states what it measured and on which machine.

### Measured on deneb's RTX 3050, 2026-09-15 (jobs 1956 / 1958, commit `0229230`)

The card reports 6144 MiB. JAX's allocator limit is **4.25 GiB** by default and
**5.38 GiB** at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. Peak per bin, float64, from
`scripts/m3_device.py ladder`:

| bin | N | live arrays (G9) | device allocator | fits 4.25 | fits 5.38 |
|---|---|---|---|---|---|
| bin01 | 192 | 0.40 GiB (7.54x) | 0.64 GiB (12.08x) | yes | yes |
| bin02 | 256 | 0.94 GiB (7.53x) | 1.51 GiB (12.06x) | yes | yes |
| bin03 | 384 | 3.17 GiB (7.52x) | 4.75 GiB (11.27x) | no | **yes** |
| bin04 | 448 | -- | > 5.38 GiB | no | no |
| bin05-07 | 512 | -- | > 5.38 GiB | no | no |

**The allocator peak, not G9's live-array count, is what decides the fit**: XLA's
intra-op scratch adds 50-60% on top, and the fraction falls with grid size (1.60x at
192^3 and 256^3, 1.50x at 384^3). The 7.0-7.5 x N^3 float64 figure recorded from the
laptop is a true live-array count and is not the requirement.

**Float32 is not a rescue for the 512^3 bins.** The f32 ladder ran through a probe
that reproduces the field stage's array sequence, and the probe failed its own
self-check (1.084 against the real stage's f64 allocator peak at 256^3, exact), so its
numbers are UPPER BOUNDS, not measurements: bin03 <= 2.75 GiB and bin04 <= 4.20 GiB
both fit, bin05 exceeded 5.38 GiB. Discounting the 8.4% still leaves 512^3 over the
limit, so the three 512^3 bins need chunked/slab FFTs or a larger card in either
precision. A real answer needs a dtype knob in `field.py`, not a probe.

**Wall, float64, unblocked end to end:** bin01 0.46 s, bin02 0.97 s. The laptop CPU
(M4 Max) does bin01 in 0.43 s, so this card gives **no speedup** on the field stage --
expected, since consumer Ampere runs float64 at a small fraction of its float32 rate.
Cross-machine, so directional rather than a controlled comparison.

Two instrument notes worth keeping: `wall` OOMed at bin03 although the ladder fits it,
because its per-step `block_until_ready` over `live_arrays()` allocates -- at the edge
of the card the instrument changes the outcome. And the live-array poller has a ~10%
spread run to run, so only the device allocator can adjudicate a percent-level band.

**CPU vs CUDA (job 1956, `runs/m3/ulp.json`), one process, one jaxlib:** median 6-12
ulp, p99 400-4000, max |diff| 4e-15 to 1.6e-13 on fields of rms 0.18-3.8, i.e. ~1e-14
relative. **Not bitwise.** The max-ulp figures (up to 1.9e9) are an artefact of the
metric at zero crossings and should not be quoted.

### Measured on a TACC Vista GH200 node, 2026-09-16/17 (jobs 999776 / 1002399, commits `cb2fdf1` / `b04ddf5`)

GH200 120GB (97871 MiB reported; JAX allocator limit 71.25 GiB), partition `gh`, one
node. Job 999776 ran `scripts/m3_device.py ladder` / `wall` / `ulp` over all seven bins;
job 1002399 (`scripts/m3_vista_b.sbatch`) ran one full seven-bin realization. Logs
`logunusual-m3-vista-999776.log`, `logunusual-m3-vista-b-1002399.log`; JSON under
`runs/m3/vista_*.json`, `runs/m3/vista_b_1002399_summary.json`.

**Every bin fits, and the allocator peak is 12.0 x N^3 float64 at every size:**

| bin | N | device allocator peak | x N^3 f64 | field stage, unblocked | laptop M4 Max |
|---|---|---|---|---|---|
| bin01 | 192 | 0.64 GiB | 12.09 | 0.36 s | 0.43 s |
| bin02 | 256 | 1.51 GiB | 12.07 | 0.83 s | |
| bin03 | 384 | 5.08 GiB | 12.04 | 2.71 s | |
| bin04 | 448 | 8.06 GiB | 12.04 | 4.29 s | |
| bin05 | 512 | 12.03 GiB | 12.03 | 6.34 s | ~8 s |
| bin06 / bin07 | 512 | 12.03 GiB | 12.03 | 6.37 / 6.39 s | |

deneb's 11.27x at 384^3 was the outlier, not the trend. **The field stage gains only
~1.2x over the laptop at 512^3.** Its block-serialised breakdown (7.95 s total):
white_noise 0.68, pkG_g 3.30, delta_g 0.29, pkG_m 2.61, psi_x/y/z 0.47/0.28/0.25. The
two P(k) -> P_G conversions are ~75% of the stage; their internal split (host spline
`target_on_grid`, upload, FFTs, `float()` syncs) is NOT measured, so no cause is named.

**CPU vs CUDA (`runs/m3/vista_ulp.json`, N = 64/128/256):** median 6-20 ulp (120 on
delta_m at 64^3), max |diff| 4e-15 to 1.2e-13. Same picture as deneb; not bitwise.

**Full seven-bin realization (job 1002399, 72 affinity cores, `OMP_NUM_THREADS=72`):**
651,427,904 galaxies, 15.04 GiB, realized/target 0.9988-1.0077 per bin, identical in
count and size to the laptop's acceptance run (ROADMAP M2). Wall **259 s against the
laptop's 187 s.**

| bin | N | field | sample | drawn / kept |
|---|---|---|---|---|
| bin01 | 192 | 5.3 s | 14.1 s | 2.8x |
| bin02 | 256 | 3.9 s | 40.6 s | 2.3x |
| bin03 | 384 | 7.2 s | 34.8 s | 2.4x |
| bin04 | 448 | 6.1 s | 41.5 s | 2.5x |
| bin05 | 512 | 8.2 s | 43.1 s | 2.6x |
| bin06 | 512 | 6.3 s | 29.9 s | 1.9x |
| bin07 | 512 | 6.3 s | 10.4 s | 2.1x |

The sample stage is ~214 s of the 259. Bash `time` gives 4m48 real against 4m07 user
with 72 threads available, i.e. **under one core busy on average: the sample stage ran
serially, and the thread count bought nothing.** The 259 vs 187 s gap is therefore a
single-core Grace vs M4 Max comparison, and the GPU node loses it. The field stage on
the GPU (3.9-8.2 s per bin, in-run) matches the standalone `wall` numbers above.

### The sample stage, rebuilt (2026-09-20, laptop M4 Max, 16 cores)

Measured on bin 2 (256^3, survey mask, 133M kept), one realization, no parquet write:

| sample stage | wall | drawn |
|---|---|---|
| M2 code, serial (one stream, `np.histogram`, whole-grid counts) | 22.7 s | 311M |
| per-slab streams, 1 thread | 19.8 s | 311M |
| per-slab streams, 8 threads | 3.2 s | 311M |
| per-slab streams, 16 threads | 2.4 s | 311M |
| + angular pre-cut, 16 threads | 2.1 s | 230M |

The M2 profile (cProfile, 22.7 s): placement 5.4, `rsd_radial` 4.9, `select` 3.9 (of
which `vec2pix` 2.7), `np.histogram` 2.9 (its sort), cell decode 1.1. numpy and healpy
release the GIL in every one of these, so the stage is thread-parallel once each slab
has its own RNG stream.

**RNG scheme.** One x-slab is the work unit; its stream is
`Philox(key = draw_seed | slab << 64)`, Poisson over the slab's cells then placement
uniforms galaxy-major (`sample.slab_rng`, `draw_slab`). A catalog depends on
`(draw_seed, field)` alone: bitwise the same for any thread count (test) and across
processes (test, on Linux). Every seed's catalog changed relative to M2. **The first
version keyed the stream by the seed and set `counter = slab`; Philox's counter is a
position within ONE stream, so adjacent slabs emitted the same numbers shifted by a
block. Every fast test passed; G10's draw-count Poisson z came out with std 9.4 per
realization. The slab index belongs in the 128-bit key; G10 now gives std 1.07-1.20.**

**Buffer guard.** `Shell.required_buffer(box, f, psi_max) = sqrt(3)/2 dx + f max|Psi|`
over the drawn cells, checked at field time; `sample_shell` raises below it and
records `psi_max` / `required_buffer` per bin. The guard found the M2 test fixtures
short (20 Mpc/h against 25-31 needed at 16^3; G10's 30 against 40-68 at 128^3) and
measured the production case: bin 2, seed 0, `max|Psi|` over the window 154 Mpc/h
(`psi_rms` 5.2; the lognormal matter field's tail, growing with the cell count: 25 at
16^3, 45 at 64^3, 70-90 at 128^3), so 113 Mpc/h of the 150 buffer is needed; bin 7
needs ~42. The production default stays 150 (JC, 2026-09-20); shrinking it would buy
1.66 -> 1.56x of shell volume on bin 2 and 1.48 -> 1.14x on bin 7, not more.

**Angular pre-cut.** Radial RSD preserves direction, so a cell is dropped iff the
nearest set-pixel centre is farther than `cell_angular_radius(r_c) + 2 max_pixrad`
from its centre's pixel (`AngularMask.distance_to_set`, one KD-tree per mask). Exact
superset of the feeding cells, checked galaxy by galaxy (band and sparse masks). With
the survey mask (fsky 0.713) the drawn cells are 0.73-0.75 of the radial window at
every bin, draws 311M -> 230M on bin 2 (drawn/kept 2.33 -> 1.72), and the windows are
built per slab inside the workers, so no N^3 window array exists.

**Full suite under the new scheme (2026-09-20, `pixi run test`, 4:41): 85 passed, 1
failed -- G4a's lowest band (k = 0.035) had SE 0.69% against the 0.67% floor with 32
seeds (ratio 1.009 +- 0.007, z +1.3; every band |z| < 2.1; every other band's SE
inside the floor).** Resolved 2026-09-21 with G11's `band_seeds` treatment, below.

### The P -> P_G step, split (2026-09-21, laptop M4 Max, `m3_device.py pkg`)

The two `target_on_grid` + `grid_pkG` conversions are ~70-75% of the field stage on
both machines, and their internal split was the last thing assumed rather than
measured. `pkg` mode times the REAL library calls in the order `generate_fields` makes
them -- `k_grid`, the spectrum call, `jitter_power_window`, the upload, `grid_xi`,
`log1p`, `grid_pk_from_xi`, the clip -- and then compares the assembled result
**bitwise** against `grid_pkG(target_on_grid(...))` in the same process, so a split
that has drifted from the shipped path cannot be reported as one. Both arms at every
bin below: BITWISE, diagnostics identical.

Median of 5, bin05 (512^3), CPU backend; `[min, max]` over the runs in the JSON:

| piece | kind | s | % of the step |
|---|---|---|---|
| `k_grid` | host | 0.086 | 3.1 |
| **spectrum spline on the grid** | **host** | **1.674** | **59.3** |
| `jitter_power_window` + divide | host | 0.088 | 3.1 |
| upload to device | upload | 0.008 | 0.3 |
| `irfftn` (xi) | device | 0.659 | 23.3 |
| `log1p` | device | 0.101 | 3.6 |
| `rfftn` (P_G) | device | 0.089 | 3.2 |
| DC / compare / clip | device | 0.028 | 1.0 |
| the five `float()`/`int()` syncs | sync | 0.088 | 3.1 |
| | | **2.82** | |

**Host work is 65.5% of the conversion and the spline evaluation alone is 59%**, at
every grid size measured (bin01 64%, bin02 64%, bin03 63%, bin05 65.5%). The upload is
0.3% and the five device syncs together are 3.1%, so neither transfer nor
synchronisation is the story. Only `irfftn_xi` is unstable run to run (0.17-0.66 s at
512^3, bimodal); every host piece repeats to under 2%, and even at the worst device
draw the host share stays above 60%.

**Laptop field stage for the same bin (`wall`, bin05): 8.31 s blocked, 8.09 s
unblocked** -- white_noise 0.52, white_k 0.08, pkG_g 2.95, delta_g 0.33, pkG_m 2.86,
delta_m 0.28, delta_m_k 0.06, psi_x/y/z 0.40/0.38/0.37. The two conversions are 5.81 s,
**70% of the stage**, matching the GH200's ~75%.

**This names the cause of the GH200's 1.2x.** 65.5% of 70% is **45% of the whole field
stage running on the host in numpy/scipy, where no accelerator can touch it**, and 27%
of the stage is the scipy spline call alone. On Vista the device half of each
conversion collapses while the host half runs on a Grace core -- slower single-core
than the M4 Max by ~1.4x, the factor the serial seven-bin run measured (259 vs 187 s) --
so the GH200's 3.30 s / 2.61 s conversions are consistent with being almost entirely
host-bound. That last step is arithmetic across two machines, not a controlled
measurement; the split itself is measured.

**Headroom, measured here; adopted 2026-09-30 (see "The spline as a per-radius table"
below).** `|k|^2` on the rfft
grid is `k_f^2 (i^2 + j^2 + l^2)` with the integer bounded by `3 (N/2)^2`, so the
spline can be tabulated once on every attainable radius and gathered with the integer
as the index -- no sort. Probed at 256^3 and 512^3: **11.7x on the spline piece at
both** (512^3: 1.668 -> 0.143 s), which would take the laptop field stage from 8.31 s
to ~5.3 s. It is **not bitwise**: 13% of cells differ by up to 2.0e-15 relative (median
exactly 0), because `sqrt(i^2+j^2+l^2) k_f` and `sqrt((i k_f)^2 + ...)` round
differently. Every catalog would change, so adopting it is the same class of call as
the per-slab Philox scheme. (Going through `np.unique` instead is a LOSS: the sort of
67.6M keys costs 4.1 s against the 1.7 s it saves.)

### A real float32 knob, and what widens an f32 field stage (2026-09-21)

`generate_fields(..., dtype="f64" | "f32")` sets the precision of the DEVICE arrays;
`field.resolve_dtype` is the table. Three things it deliberately does not do:

- **The white noise is still drawn in float64** and cast on the way to the device. A
  numpy generator consumes its stream differently per dtype, so drawing natively would
  change the realization rather than its precision, and the f32 run would no longer be
  comparable to the f64 one cell by cell.
- **`Fields.psi_flat` still returns float64** -- the sampler's interface does not move.
- **It is not a `RunConfig` field.** f32 changes every catalog, so putting it in the
  config would change the hash of every existing config for a knob no production run
  selects. It stays an API / instrument argument until it is a production choice.

**What actually widens an f32 field stage is `k_components`, not `1j`.** It returns
numpy float64, and with x64 enabled a single float64 array in
`1j * comps / k2 * delta_k` drags the whole expression to complex128 and the psi
irfftn back to float64. `1j * float32` on its own is complex64 (jax 0.10.1, measured
here). `displacement_k` therefore takes its working dtype FROM `delta_k` rather than
from an argument, so the two can never disagree; nothing else in the stage needed an
edit, and the float64 path is textually unchanged.

**Gates.** `scripts/m1_reproducibility.py --n 64` IDENTICAL on macOS after the change;
fast suite 83 passed. Four tests in `tests/test_field.py`: `resolve_dtype` rejects
anything else; f64 is the default (bitwise on Linux, characterised on macOS, per the
in-process reproducibility test above); every returned array is float32 under `f32`
and `psi_flat` is still float64; and the f32 field agrees with the f64 one to **2-22
float32 eps of the field's own maximum** at (N, L) = (32, 320), (64, 640), (32, 1000),
i.e. sigma2 from 0.6 to 3.0, with no growth in N -- gated at 100 eps32. The tolerance
is on max|field|, not the rms: a lognormal's peak sits 30-100x its rms, so an
rms-normalised tolerance measures that ratio rather than the arithmetic.

**Footprint, laptop `ladder` (live-array instrument; the CPU backend has no
allocator):**

| bin | N | f64 | f32 |
|---|---|---|---|
| bin01 | 192 | 7.02 x N^3 f64 | **3.51** |
| bin02 | 256 | 7.53 x N^3 f64 | **3.51** |

Exactly half, at both sizes. **The allocator peak in f32 is still unmeasured**, and it
is the one that decides a fit: XLA intra-op scratch added 50-60% on top in f64 and
nothing says it halves. That needs one short GPU ladder run; until then the f64 table
above is the only statement about what fits a card. `probe_field_arrays` and its
self-check machinery are deleted -- `--dtype f32` now runs the shipped stage.
(Measured 2026-09-22, job 1014508, below: the f32 allocator peak is half the f64 one.)

### `jax.jit` of the field stage, and the bitwise check (2026-09-21)

`generate_fields(..., jit=False)` compiles the stage's two pure device functions,
`coloured_lognormal` and `displacement` (`box` and `axis` static; `Box` is a frozen
dataclass so it hashes). Nothing else in the stage can go inside a jit: the spectrum
spline runs on the host, and `grid_pkG` needs values BACK from the device for its
diagnostics and its `xi <= -1` raise.

**It is not bit-preserving, and that is why it is off by default.** At 64^3, 37-88% of
cells move; at 256^3, 88%. The move is round-off and nothing more: normalised to the
field's own maximum it is **0.25-4.0 float64 eps**, flat across 8 seeds at N = 64 and
96. The per-CELL relative figure reaches 1e-10 at 64^3 and 2e-8 at 256^3, but that is
the metric blowing up at zero crossings -- the same artefact this file already flags
for the ULP table -- and it should not be quoted as the size of the effect. Against
it, eager repeated twice gave **zero** differing cells on this machine, so the jit
difference is the compiler, not the platform.

**It does not make the stage faster: 1.07x at 64^3 and 0.95x at 256^3** (laptop CPU,
median of 3). That follows from the split above -- 45% of the stage is host numpy and
scipy, which no compiler in JAX can reach, and the device half was already one FFT per
step with little to fuse.

**It does not change the live-array peak either** (7.02 / 3.51 x N^3 f64 at bin01 and
bin02, identical eager and jit, both dtypes). The peak that would move is XLA's
intra-op scratch, which added 50-60% on top in the deneb f64 table, and **a CPU
backend has no allocator to report it**. That is the one open question jit leaves, and
it is the same run the f32 allocator question needs: one short GPU `ladder` over
f64/f32 x eager/jit. `ladder --jit` is in the instrument for it. (Measured 2026-09-22,
job 1014508, below: jit takes 2.0 x N^3 f64 off the f64 allocator peak, and is 0.93x
eager on the 512^3 wall.)

The k grid does NOT get baked into the executable as a constant, which was the risk of
making `box` static: at 512^3 `displacement_jit`'s `memory_analysis` reports
generated code 0.000 GiB against 1.004 GiB of argument and 1.004 GiB of temp.

### G4a on fixed bands (2026-09-21)

`gate_uniform_shot` takes `band_seeds`, the split `gate_catalog` has had since M2, and
G4a now runs **64 seeds on the 32-seed bands**. The lowest band goes from SE 0.69%
against the 0.667% floor to **0.51%** -- 24% margin -- and its ratio tightens from
1.009 +- 0.007 (z +1.3) to 1.0004 +- 0.0051 (z +0.07). Gate wall 53 s.

**Nothing was loosened, and the obvious alternative would not have worked.** `TOL/3`
is not a tightness setting: TOL is 2% and SE <= TOL/3 is what lets the gate resolve a
2% bias at 3 sigma, so relaxing it would let the gate pass a sampler carrying exactly
the bias it advertises as excluded. Adding seeds with the bands left to re-derive does
nothing either, because the seed count CANCELS -- `n_min = ceil(9 / (TOL^2 n_real))`,
so the Gaussian SE is pinned at TOL/3 whatever `n_real` is. Measured on G4a's own
geometry, the worst band's Gaussian SE is 0.576% at 32 seeds, 0.587% at 48, 0.621% at
64 and **0.643% at 128**: it rises, because the bands narrow faster than the seeds
help. The floor is therefore met exactly by construction with ~15% headroom and no
more, at any seed count, and a Poisson-sampled ratio's scatter is super-Gaussian
enough to spend it.

Why it is always the LOWEST band: at high k a single k-shell already holds twice
`n_min`, so those bands sit far under the floor, while at low k the shells are thin and
merging lands just over `n_min` (overshoot 1.40, 1.43, 2.01, 1.34 for the first four
bands at 32 seeds). The low bands have the least mode overshoot and the most
super-Gaussian scatter, both pushing the same way.

G4a now uses the same bands as G11, so the two gates stay comparable.

### Allocator peaks f64/f32 x eager/jit, on a Vista GH200 (2026-09-22, job 1014508, commit `00d0344`)

`scripts/m3_vista_alloc.sbatch`, partition `gh`, 72 affinity cores, allocator limit
71.25 GiB; log `logunusual-m3-alloc-1014508.log`, JSON `runs/m3/vista_alloc_*.json`.
The device check and the dtype check (the f32 leg returns float32 from `cuda:0`) passed
first; every phase rc 0. One ladder point per process, so each peak is its own.

**Device allocator peak, x N^3 float64** (GiB at 512^3 in brackets):

| bin | N | f64 eager | f64 jit | f32 eager | f32 jit |
|---|---|---|---|---|---|
| bin01 | 192 | 12.59 | 10.05 | 6.30 | 5.03 |
| bin02 | 256 | 12.57 | 10.04 | 6.29 | 5.02 |
| bin03 | 384 | 12.04 | 10.03 | 6.28 | 5.02 |
| bin04 | 448 | 12.04 | 10.02 | 6.02 | 5.08 |
| bin05 | 512 | 12.03 (12.03) | 10.02 (10.02) | 6.02 (6.02) | 5.07 (5.07) |

- **float32 halves the allocator peak**, scratch included: 0.500 of f64 at 448^3 and
  512^3 (0.52 at 384^3).
- **jit takes 2.0 x N^3 f64 off the f64 peak** (12.03 -> 10.02 at 512^3, 0.83x) and
  ~1 x N^3 off the f32 one; f32 + jit is 0.42 of f64 eager at 512^3.
- f64 eager at 384^3-512^3 reproduces job 999776 to four digits. At 192^3 and 256^3 it
  reads 12.59 / 12.57 against 999776's 12.09 / 12.07 (+0.5 x N^3, ~4%); not
  investigated -- it is a small-grid term and changes no fit. The live-array column
  (7.5-8.5x f64 eager in both jobs) moves between runs by up to 1 x N^3 at the same
  grid, which is the poller's known spread.

**What fits.** On this card, every arm at every grid, with >5x headroom; the "512^3
bin under 15 GB" target is met by f64 eager (12.03 GiB). For deneb's RTX 3050 the
table gives a PREDICTION, not a measurement -- deneb read 11.27x at 384^3 where this
card reads 12.04x, so the scratch term is card-dependent: at 512^3 only f32 + jit
(5.07 GiB) comes under the 5.38 GiB limit at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`,
by 6%; nothing at 512^3 comes under the default 4.25 GiB. 448^3 f32 eager (4.03 GiB)
is under 4.25 by 5%.

**The P -> P_G split on Grace (`pkg`, median of 5, both arms BITWISE against the
shipped call, diagnostics identical):**

| | 256^3 laptop | 256^3 GH200 | 512^3 laptop | 512^3 GH200 |
|---|---|---|---|---|
| conversion (pkG_g) | 0.38 s | 0.34 s | 2.82 s | 2.57 s |
| host | 64% | 97.3% | 65.5% | 98.0% |
| spline alone | 0.22 s | 0.29 s | 1.67 s | 2.25 s |
| device | 0.12 s | 0.004 s | 0.88 s | 0.014 s |

On the node the device half of the conversion is 30-60x faster than the laptop's and
is now ~1% of it; the spline runs 1.3x SLOWER on one Grace core than on the M4 Max.
At 512^3 the two conversions' host work is 5.0 s of the 6.27 s unblocked field stage
(spline alone 4.5 s), measured on the node rather than scaled from the laptop. **The
GH200's ~1.2x is the host spline**, and the spline is now the largest single piece of
the GPU field stage.

**512^3 wall, eager vs jit (`wall`, one run each):** unblocked 6.27 s eager, 5.81 s
jit (0.93x; eager's unblocked spread across jobs 999776 / 1014508 is 6.27-6.39 s).
The gain is on the device steps (delta_g 0.31 -> 0.10 s blocked); the conversions do
not move (3.16 / 2.63 s eager, 3.16 / 2.62 jit). The blocked jit leg includes its own
512^3 compile (the N = 32 warm-up compiles a different shape); the unblocked leg reuses it.

### The spline as a per-radius table (2026-09-30)

`pk_on_grid` now calls the spectrum once on `k_f sqrt(q)` for every integer
`q <= 3 (N/2)^2` and gathers by `pk.radius_index` (`i^2 + j^2 + l^2`, int32, in
`fftfreq` / `rfftfreq` order). It takes any callable of `|k|`, as before, so both field
conversions and the gate predictions in `gates.py` use it. It is **not bitwise**
against evaluating on `k_grid`'s `|k|`, so every catalog changes, at the last bit;
nothing in the catalog records the scheme (decided 2026-09-30).

**Gate (`tests/test_pk.py`, derived, not picked).** The index must equal
`rint((|k| / k_f)^2)` exactly. The two `|k|` constructions differ by <= 2 eps at
every size measured (N = 16-256, `|dk/k|` max 2.00 eps, p99 1.0-1.5); every v28 table
has `max |dlnP/dlnk|` = 2.38 over its grid's k range. P = exp(spline(ln k)), so beyond
the slope term each side rounds by ~2 ulp of ln P (ln P ~ 10, whose ulp is 8 eps in P).
Bound: `slope_max * |dk/k| + 4 ulp(max |ln P|)`, ~37 eps (8.2e-15) at (32, 320).
Measured: **max 9 eps, median 0, 13-16% of cells moved**, on bin01 and bin02 with
their own tables. Mutations fail it: an off-by-one radius by 0.15, `k_f (1 + 1e-6)`
by 2.4e-6. `m3_device.py pkg` times `radius_index` / `spline_table` / `gather` in
place of `k_grid` / `spline_eval` and stays BITWISE against the shipped call.
`m1_reproducibility --n 64` IDENTICAL.

Full suite 99 passed / 0 failed (4:50), `check-format` + `lint` clean.

**Timed on the shipped code (laptop M4 Max, idle, bin05 512^3; JSON
`runs/m3/pkg_bin05_table.json`, `wall_laptop_bin05_table.json`):**

| | 2026-09-21 (direct) | 2026-09-30 (table) |
|---|---|---|
| k magnitude + spectrum on the grid | 0.086 + 1.674 = 1.76 s | 0.023 + 0.003 + 0.117 = 0.14 s (index, table, gather) |
| one conversion (pkG_g, median of 5) | 2.82 s, host 65.5% | 1.20 s, host 19.2% |
| field stage, blocked / unblocked | 8.31 / 8.09 s | **5.16 / 4.62 s** |

12.3x on the piece the table replaces, 2.35x on each conversion, 1.6-1.75x on the
stage (one `wall` run each side). The conversion is now 73% device on the laptop, and
`irfftn_xi` is still the one bimodal piece (0.16-0.68 s). (Measured on the GH200
2026-09-30, job 1039040, below: 5.8x on each conversion, 3.1x on the stage.)

### Seven-bin realization on the laptop, re-measured (2026-09-30, commit `44ee60a`)

`logunusual run` with the default v28 config, full density, the SPHEREx fiducial mask
(fsky 0.7127), realization 0, 16 sampler threads, M4 Max idle. `logunusual check`:
layout ok.

| bin | N | field | sample | drawn / kept | realized / target |
|---|---|---|---|---|---|
| bin01 | 192 | 0.8 s | 1.9 s | 2.09x | 1.0079 |
| bin02 | 256 | 1.1 s | 5.6 s | 1.72x | 1.0018 |
| bin03 | 384 | 2.4 s | 5.0 s | 1.74x | 0.9988 |
| bin04 | 448 | 5.0 s | 11.7 s | 1.82x | 0.9990 |
| bin05 | 512 | 7.7 s | 12.3 s | 1.91x | 1.0004 |
| bin06 | 512 | 7.1 s | 9.8 s | 1.40x | 0.9994 |
| bin07 | 512 | 7.3 s | 4.0 s | 1.52x | 1.0010 |

**82 s wall against 187 s (2026-09-04), 2.3x**; 651,439,871 galaxies (a new draw
under the per-slab streams; 651,427,904 before), 15.04 GiB, peak RSS 13.8 GB
(`/usr/bin/time -l`) against the 19.3 GB recorded then. Field stage 31 s of the 82,
sample stage (with the streamed write) 50 s. The draw over-shoot fell from 2.4x to
1.4-2.1x with the angular pre-cut. The headline target, a seven-bin realization in
single-digit minutes on the laptop, is met at under a minute and a half.

### Seven-bin realization on a Vista GH200, re-measured (2026-09-30, job 1039040, commit `0684ce1`)

`scripts/m3_vista_c.sbatch`, partition `gh`, one node, 72 affinity cores, 72 sampler
threads; log `logunusual-m3-vista-c-1039040.log`, JSON `runs/m3/vista_c_*.json`. The
device check passed first; every phase rc 0; `logunusual check`: layout ok.

**The spline table on the node, bin05 512^3** (`pkg` median of 5, both arms BITWISE
against the shipped call, diagnostics identical; `wall` one run):

| | job 1014508 (direct) | job 1039040 (table) | laptop (table) |
|---|---|---|---|
| spectrum on the grid | 2.25 s (spline) | 0.26 s (index 0.026, table 0.004, gather 0.231) | 0.14 s |
| one conversion (pkG_g) | 2.57 s, host 98.0% | **0.44 s**, host 88.3% | 1.20 s, host 19.2% |
| field stage, blocked / unblocked | 7.74 / 6.27 s | **3.55 / 2.01 s** | 5.16 / 4.62 s |

5.8x on each conversion and 3.1x on the unblocked stage; the GH200 field stage is now
2.3x the laptop's. The conversion is still host-bound on the node: the gather alone is
52% of it (0.23 s, twice the laptop's 0.12 s) and `jitter_window` + divide another 29%;
the device pieces total 0.014 s.

**Full seven-bin realization: 82 s (81.9) against 259 s on job 1002399 and the
laptop's 82 s (82.1) on the same code.**

| bin | N | field GH200 | field laptop | sample GH200 | sample laptop |
|---|---|---|---|---|---|
| bin01 | 192 | 2.2 s | 0.8 s | 2.7 s | 1.9 s |
| bin02 | 256 | 2.3 s | 1.1 s | 8.2 s | 5.6 s |
| bin03 | 384 | 6.7 s | 2.4 s | 7.5 s | 5.0 s |
| bin04 | 448 | 5.9 s | 5.0 s | 9.2 s | 11.7 s |
| bin05 | 512 | 6.8 s | 7.7 s | 9.8 s | 12.3 s |
| bin06 | 512 | 1.9 s | 7.1 s | 8.6 s | 9.8 s |
| bin07 | 512 | 1.9 s | 7.3 s | 4.0 s | 4.0 s |
| total | | 27.8 s | 31.5 s | 50.1 s | 50.4 s |

**The realization is the laptop's, count for count:** 651,439,871 galaxies, and per bin
the kept, drawn and left-the-box counts and both seeds agree exactly; `psi_max` agrees
to <= 3.4e-15 relative and `sigma2_galaxy` to <= 3e-16, the CPU-vs-CUDA last-bit
difference recorded above. 15.04 GiB, realized/target 0.9988-1.0079.

Two observations, neither measured:

- **A first-use cost per grid size on the GPU.** Bins 6 and 7 (the second and third
  512^3 bins) spend 1.9 s in the field stage, matching the standalone unblocked 2.01 s;
  bin05, the first 512^3 bin, spends 6.8 s, and bin03 (the only 384^3) 6.7 s. The
  laptop shows no such step (bins 5-7 at 7.1-7.7 s). Scaling the 512^3 2.0 s by
  N^3 log N puts bins 1-5's work at ~4.4 s against the 24.1 s they spend, so on that
  scaling ~20 s of the 82 is first-use cost. The pattern fits a per-shape compile or
  FFT-plan cost; the cause is not measured.
- **The sample stage does not scale past the laptop's cores.** 50.1 s on 72 threads
  against 50.4 s on 16; per bin the GH200 is faster on bins 4-6, slower on 1-3, equal on
  7. Bash `time` for the whole process: 1m42.5 real, 3m42.5 user, so 2.2 cores busy on
  average (field stage, startup and the streamed write included). What serialises it
  is not measured.

**M3 targets met:** a 512^3 bin under 15 GB (12.03 GiB, f64 eager, GH200) and a
seven-bin realization on the laptop in single-digit minutes (82 s). The two
observations above are open performance questions, not M3 deliverables.

## M4 -- Physics upgrades (each opens its own plan session)

- **Nonlinear input P(k):** halofit TSVs (`scripts/make_pk_tables.py`) together
  with the finer grids (up to 1024^3) the finer scales need; at the production grids
  halofit alone changes little because the grid Nyquist binds first. Construction and
  gates below.
- **f_NL scale-dependent bias in the input galaxy P(k):**
  `b(k) = b + 2 (b - p) f_NL delta_c / M(k)` with the Poisson factor `M(k)`. Gate: low-k
  P_0(f_NL) / P_0(0) matches the injected ratio; the P_G inversion stays non-negative
  where the grid uses it. Scope: two-point only (null tests, covariance shape); a
  lognormal is not an f_NL bispectrum mock. Construction and gates below.
- On demand only: Finger-of-God scatter (`sigma_psi`), deterministic-count sampling
  (`minimize_shotnoise`), multi-tracer.

### f_NL scale-dependent bias: construction and gates (planned 2026-10-01)

**Convention: LSS** (JC, 2026-10-01): `M(k, z) = (2/3) (c/H0)^2 k^2 T(k) D(z) / Om`
with `D(0) = 1`, so `f_NL^LSS = f_NL^CMB / g0`, `g0 = D_md(0)` (growth normalised to `a`
in matter domination; ~0.78 for Om = 0.3153). Supported and gated for `|f_NL| <= 100`.

**M(k) comes from the bin's own P(k) table.** With `Phi = (3/5) zeta` in matter
domination, `P(k, z) = M_CMB(k, z)^2 P_Phi(k)`, `P_Phi = (9/25) 2 pi^2 A_s k^-3
(k / k_pivot)^(n_s - 1)`, so `M_CMB = sqrt(P / P_Phi)` exactly and z enters through the
table; `M = M_CMB / g0`. Inputs: `A_s`, `n_s`, `k_pivot` of the tables (v28: 2.1e-9,
0.9649, 0.05/Mpc) and Om for g0. Only the galaxy target changes,
`b^2 P -> b(k)^2 P`, `b(k) = b + 2 (b - p) f_NL delta_c / M(k)`, p = 1, delta_c = 1.686;
the matter field and the velocities do not. f_NL = 0 is bitwise the M3 code, and the
f_NL keys enter the config hash only when nonzero.

**Gates (thresholds derived in the tests that carry them):**

| gate | statistic | where |
|---|---|---|
| G12 | `M_CMB / [(2/3)(c/H0)^2 k^2 D_md / Om]` -> 1 at the lowest table node (the T = 1 limit, growth from the flat-LCDM integral); tolerance from T(k)'s own low-k departure plus the neutrino / radiation terms the integral omits. Mutations must fail: pivot 0.05/Mpc read as h/Mpc (0.7%), a missing 9/25, CMB normalisation where LSS is meant. Growth integral: EdS gives g0 = 1; D -> a at high z. b(k): sign, `Delta b ~ k^-2` at low k, `b = p` gives none, f_NL = 0 bitwise. | fast, `tests/test_fnl.py` |
| G13 | G3's grid identity `<P(delta_g)> / P_grid` with the b(k) target, f_NL = +-100, every band to the Nyquist, SE <= 2%/3 | slow, 128^3, L 1000, bin-5 b |
| G14 | matched seeds (same ic and draw): per-realization real-space `(P0 - shot)` ratio f_NL / f_NL = 0 against `<b(k)^2 P R> / <b^2 P R>` (R the estimator response), low k | slow, a box large enough for the signal (L 2000-3000, dx ~ 16) |

**Attainability (measurement, before the slow gates):** every v28 bin at its
production grid, f_NL in {-100, -10, -1, 1, 10, 100}: `xi_min`, sigma^2, clipped modes
and power fraction of P_G, `b(k_f) / b`, and the k where b(k) crosses zero (f_NL < 0).
Any clipping is reported as a finding; nothing handles it automatically.

**Attainability, measured (2026-10-01, laptop, `scripts/m4_fnl.py attain`, 40 s, peak
RSS 12 GB; `runs/m4/attain_20261001-111917.json`).** Clipped galaxy P_G modes per bin
at its production grid:

| bin | N, L | -100 | -10 | -1 | 0 | +1 | +10 | +100 |
|---|---|---|---|---|---|---|---|---|
| 1 | 192, 1500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 256, 2500 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 384, 3500 | 17 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 448, 4500 | 41 | 5 | 0 | 0 | 0 | 0 | 0 |
| 5 | 512, 5000 | 85 | 5 | 0 | 0 | 0 | 0 | 0 |
| 6 | 512, 7000 | 486 | 33 | 5 | 0 | 0 | 0 | 0 |
| 7 | 512, 8000 | 1540 | 94 | 13 | 5 | 0 | 0 | 0 |

**Positive f_NL is attainable everywhere; negative f_NL is not, on the lowest shells.**
At f_NL < 0, b(k) falls toward zero at low k (and crosses it: bin 5 at -100 has
`b(k_f)/b = -6.8`, zero at k = 3.8e-3). A lognormal at the grids' sigma^2 of 2.3-3.3
cannot carry power that low: the higher orders of `log(1 + xi)` add a white term that
floors the realizable power, so P_G goes negative on exactly those shells. The clipped
power FRACTION (1e-8 to 2e-5) hides how wrong the result would be there. The exact
ensemble power of the clipped field against the target, per shell (|k| in units of k_f):
bin 3 at -100, 8.6x at sqrt2 and 4.0x at sqrt3; bin 5 at -10, 8.6x at the fundamental;
bin 5 at -100, 1.6x-306x over 2.45-3.6 (the peak where b(k) crosses zero); bin 7 at -1,
3.9x at 1 and 1.37x at sqrt2; bin 7 at -10, 2x-230x over 1.7-3.5. Every other shell is
exact.

**Decision (JC, 2026-10-01): with f_NL != 0 any clipped galaxy P_G mode raises in the
field stage** (`field.generate_fields`), before the bin is sampled. Gaussian runs keep
reporting clipping (bin 7's fundamental, 1.109x; see Known risks). An f_NL < 0 run is
therefore possible only where it is exact: at -1 everywhere but bins 6-7, at -10 in
bins 1-3, at -100 in bin 1. The gate boxes follow: G13 on G3's geometry (128^3, L 1000)
clips at neither +100 nor -100; G14's 128^3, L 2000 box clips at -50 and -100, so G14
runs at +100 and -20.

**Gates as run (2026-10-01, laptop; `tests/test_fnl.py`, `tests/test_gates_m4.py`):**

| gate | result |
|---|---|
| G12 | `1 - M / M_{T=1}` = 7.7e-4 at k = 1e-4 (z 0.9) against the derived bound 1.59e-3; on all seven v28 tables 4.5e-4 (z 0.1) to 1.25e-3 (z 1.9), each inside its own bound by 1.5-3x. Mutations fail: pivot read as h/Mpc -6.2e-3, A_s +2% 1.1e-2, CMB normalisation 0.21, no 9/25 0.40. |
| G13 | 128^3, L 1000, b 1.76, 192 seeds, 60 bands to k_Nyq, 0 clipped modes. f_NL = +100 (`b(k_f)/b` 1.35): galaxy max |z| 3.25, matter 2.73; f_NL = -100 (0.65): galaxy 2.19, matter 2.54. SE 0.05-0.6%, floor met in every band. |
| G14 | 128^3, L 2000, estimator 256^3, b 1.76, nbar 1e-3, 32 matched seeds, one band per kf-shell to k = 0.05. f_NL = +100: max |z| 1.97; k_f measured/predicted 1.074 +- 0.040 on a predicted 3.48x. f_NL = -20: max |z| 2.16; k_f 0.987 +- 0.024 on a predicted 0.69x. |

**G14 is a consistency gate (|z| < 4, no SE floor)** (JC, 2026-10-01). The repository's
2%/3 floor is sized to the catalog power itself, and here it would apply to the ratio
measured/predicted whatever the size of the f_NL effect in that shell. At k_f, 32 seeds
already resolve f_NL = +100's 248% boost to ~6% of the boost. The lowest shells stay
noisy under matched seeds. At sigma^2 ~ 3, part of the low-k lognormal power is a white
term built from the high-k modes; it is nearly the same in both arms but does not scale
with the shell's own amplitude, so the ratio does not cancel there (per-realization
scatter ~26% at k_f, ~10% at 2 k_f). Meeting the floor per shell would take ~1500 seeds
at k_f. The exact, floored statement is G13; the sampler, linear in the intensity, is
already gated by G4b/G5.

Observed, not investigated: G13's galaxy field at f_NL = +100 reads coherently low
above k = 0.2 (mean -0.26% over 32 bands, mean z -2.1; max |z| 3.25 at k = 0.35), where
G3 on the same geometry reads +0.10% and G13 at -100 reads -0.11%. With nothing clipped
the grid identity is exact, and the lognormal's high-k scatter is correlated across k
(rare peaks), so this is within the gate.

Full suite 117 passed (8:36), `check-format` + `lint` clean, `m1_reproducibility --n
64` IDENTICAL.

### Nonlinear input P(k): construction and gates (planned 2026-10-01; descoped to the production grids the same day)

**Two tables per bin; only the galaxy target goes nonlinear** (JC, 2026-10-01).
`Bin.pk_file` stays the linear table and drives the matter field, the velocities and
M(k): continuity `Psi_k = i k / k^2 delta_m,k` is a linear relation, and
`P = M^2 P_Phi` holds only in linear theory. A new `Bin.pk_galaxy_file` (empty = the
same table) sets the galaxy target, `b(k)^2 P_gal`, with `b(k)` still built from the
linear table. The field is drawn from one white noise as before; with distinct tables
the matter field and velocities are bitwise those of the linear run. `pk_galaxy_file`
enters the config hash only when set, so every existing config and catalog is
unchanged.

**Tables** (`scripts/make_pk_tables.py`, pixi env `tables`, osx-arm64 only:
conda-forge `camb 1.5.9` build `py312h904ef0c_0`, the build that made the v28 tables
in the emufab env; linux-aarch64 has no 1.5.9). Cosmology and call sequence are
`make_matter_power.py`'s (Planck 2018, `delta_tot`). Nonlinear: CAMB
`halofit_version = "takahashi"` (Takahashi et al. 2012, JC). CAMB's implementation
carries the Bird et al. 2012 massive-neutrino terms (`fnu = omnuh2 / omm0` in `beta`,
the halo term and `plinaa`; read in the CAMB 1.6.1 `halofit.f90` source, the env runs
the 1.5.9 binary) and applies the correction only above `Min_kh_nonlinear` = 0.005
h/Mpc. New tables run kh 1e-4 to 10 at 12501
nodes (the v28 tables' 2500 per decade):
`matterpower_camb_{lin,halofit}_kmax10_zeff={z}.tsv`. The 2x grids' corners reach
k = 1.39 (bin 1), past the v28 tables' kh = 1.

**Grids: the production grids** (JC, 2026-10-01). Planned as `grid_scale = 2` for
every bin, then per-bin clip-free grids (270 / 450 / 720 / 896 / 1024^3); both are
recorded below as measured dead ends (halofit unattainable at 2x in bins 1-3, and the
displacement tail outruns the radial buffer on every finer grid tried). On the
production grids halofit is attainable in bins 1-6 and is 1.06x (bin 7) to 1.89x
(bin 1) the linear power at the Nyquist (1.01x-1.18x at half of it). **A run with a
galaxy table raises on any clipped galaxy P_G mode** (JC), as an f_NL run does: the
clipped field's excess power spreads to every k. **Bin 7 keeps the linear galaxy
target in `configs/v28_halofit.yaml`** (2026-10-01): its 5 fundamental
modes clip with either table, so a galaxy table would raise there, and halofit is
1.06x linear at its Nyquist.

**Checks and gates (thresholds derived in the tests or scripts that carry them):**

| check | statistic | where |
|---|---|---|
| P1 | the table script regenerates the seven v28 linear tables (kh 1e-4 to 1, 10001 nodes) byte for byte; otherwise the max relative difference is reported and nothing proceeds | `make_pk_tables.py --check`, recorded here |
| pair | `pk.check_table_pair`: a common lowest k node, and `|P_gal / P_lin - 1|` there within `k0^2 sigma_v^2` (the leading low-k nonlinear correction, `P_13 -> -k^2 sigma_v^2 P_lin`, from the linear table; JC); a z-mismatched pair (adjacent bins) fails by > 10x the bound. Called at load in `run.py` | fast, `tests/test_pk.py` |
| wiring | `pk_galaxy_file == pk_file` is bitwise the single-table stage; with distinct tables `delta_m` / `psi` are bitwise the linear run's and `delta_g` the single-table run's on the galaxy table; with f_NL the target is exactly `(b + delta_b_lin)^2 P_gal` | fast, `tests/test_field.py`, `tests/test_fnl.py` |
| G15 | G3's grid identity with the halofit galaxy target on G3's geometry (128^3, L 1000, dx 7.8, bin-5 b, z 0.9 kh-10 tables, 192 seeds), galaxy and matter, every band to the Nyquist, SE <= 2%/3 | slow, `tests/test_gates_m4.py` |

(G16, the linear matter identity on a finer cell, was dropped with the finer grids; on
G3's geometry the matter field is G3's, and G15 runs it alongside.)

No new catalog-level gate: the sampler is unchanged and linear in the intensity
(G4b/G5), and the two-table wiring is bitwise-gated above.

**Measurements before G15 (heavy, each priced and asked first):** attainability at
grid_scale 1 and 2 for every bin with the linear-kmax10 and halofit tables (galaxy and
matter `xi_min`, sigma^2, clipped P_G modes and power fraction; the M4 f_NL set), and
one realization per bin on `configs/v28_halofit.yaml`'s grids (wall, peak RSS,
`psi_max` against the 150 Mpc/h buffer). G15's 128^3, L 625 box is clip-free: galaxy
halofit sigma^2 11.89 (production bin 5 at 1024^3: 11.89), matter 2.02.
GPU f64 at 1024^3 (~96 GiB allocator peak) does not fit a GH200; f32 (~48) would, but
f32 is not a `RunConfig` field. Reported, not decided here.

**Tables and pair check, measured (2026-10-01, laptop, one thread).** P1: the seven v28
linear tables regenerate **byte-identical** (26 s). New tables, sha256 (first 12):

| z_eff | `lin_kmax10` | `halofit_kmax10` |
|---|---|---|
| 0.1 | `a08066113c62` | `687d36c376a2` |
| 0.3 | `dd20362d66a8` | `9ea62c1e185c` |
| 0.5 | `3e2322a0608e` | `edf1498ee704` |
| 0.7 | `f53b20da5460` | `7a373f25ba5e` |
| 0.9 | `0b9487771d36` | `278319d961be` |
| 1.3 | `f6f3d72f4f9b` | `93e4dce6e15b` |
| 1.9 | `1e3cde868f1a` | `ae46a116a81c` |

- `P_halofit / P_lin - 1` at k0 = 1e-4 is **exactly 0** on all seven pairs; a pair one
  bin apart in z is off by -0.17 to -0.35. The bound `k0^2 sigma_v^2` is 6.4e-8 (z
  1.9, sigma_v 2.52 Mpc/h) to 3.1e-7 (z 0.1, 5.54), so a mismatch fails by ~10^6x.
- The kh-10 linear tables have the v28 nodes exactly (|dk/k| = 0) and differ from the
  v28 tables by up to 1.2e-5 in P (7.5e-6 below k = 0.1); the two CAMB calls differ
  only in `kmax` (2 vs 20 /Mpc) and the output range. That is above the pair bound, so
  a halofit table pairs with the kh-10 linear one only.
- Halofit / linear at the 2x Nyquist: 4.16 (bin 1, k 0.80), 2.82, 2.75, 2.26, 2.13
  (bin 5, k 0.64), 1.45, 1.24 (bin 7, k 0.40). Below k = 0.005 (CAMB's
  `Min_kh_nonlinear`) the ratio still departs from 1, by at most 5.9e-4 (z 0.1);
  not investigated.

**Attainability at grid_scale 1 and 2, measured (2026-10-01, laptop,
`scripts/m4_fnl.py attain`, kh-10 tables; `runs/m4/attain_gs{1,2}*_{lin,halofit}.json`;
peak RSS 10.3 GB at 512^3, 28.8 GB at 896^3, 43.4 GB at 1024^3, ~4.4 min for bins 5-7
per table).** Grid sigma^2 (matter; galaxy linear; galaxy halofit) and the Gaussian
galaxy P_G clipping, f_NL = 0:

| bin | grid | dx | sigma^2 m | sigma^2 g lin | clipped lin | sigma^2 g halofit | clipped halofit (power frac) | xi_min halofit |
|---|---|---|---|---|---|---|---|---|
| 1 | 192 | 7.8 | 2.82 | 2.99 | 0 | 5.23 | 0 | -0.062 |
| 1 | 384 | 3.9 | 5.52 | 5.85 | 0 | 20.09 | **10,222,648 (1.7e-1)** | -0.781 |
| 2 | 256 | 9.8 | 1.78 | 3.10 | 0 | 4.39 | 0 | -0.021 |
| 2 | 512 | 4.9 | 3.65 | 6.36 | 0 | 15.52 | **2,429,104 (2.2e-3)** | -0.322 |
| 3 | 384 | 9.1 | 1.57 | 3.29 | 0 | 4.61 | 0 | -0.014 |
| 3 | 768 | 4.6 | 3.16 | 6.65 | 0 | 15.80 | **1,122,793 (1.1e-4)** | -0.182 |
| 4 | 448 | 10.0 | 1.15 | 2.84 | 0 | 3.64 | 0 | -0.010 |
| 4 | 896 | 5.0 | 2.37 | 5.85 | 0 | 11.71 | 0 | -0.011 |
| 5 | 512 | 9.8 | 0.99 | 3.06 | 0 | 3.84 | 0 | -0.008 |
| 5 | 1024 | 4.9 | 2.02 | 6.26 | 0 | 11.89 | 0 | -0.008 |
| 6 | 512 | 13.7 | 0.47 | 2.31 | 0 | 2.54 | 0 | -0.007 |
| 6 | 1024 | 6.8 | 1.04 | 5.11 | 0 | 7.03 | 0 | -0.006 |
| 7 | 512 | 15.6 | 0.26 | 2.77 | 5 | 2.92 | 5 | -0.012 |
| 7 | 1024 | 7.8 | 0.58 | 6.32 | 0 | 7.61 | 0 | -0.006 |

- The kh-10 linear tables at grid_scale 1 reproduce the M4 attainability table's clip
  counts in every row. The matter targets clip nowhere.
- **At 2x the linear galaxy targets clip nowhere (Gaussian), and bin 7's fundamental
  clips no longer. The halofit galaxy targets clip nowhere in bins 4-7 and massively in
  bins 1-3.** f_NL != 0 adds the same few low-k modes as with the linear target (the
  M4 picture) on top; bins 1-3 at 2x clip at every f_NL.
- Exact ensemble power of the clipped halofit field against its target, per kf-shell
  (`grid_pk_from_xi(expm1(grid_xi(P_G clipped)))`, no realizations;
  `runs/m4/clip_realized_halofit_gs2.json`): **bin 1, 1.04-1.16x below k = 0.1, up to
  2.2x near the Nyquist (clipped modes at |k| 0.29-0.81)**; bin 2, <= 0.5% below
  k = 0.3 and up to +1.15% in the clipped band (|k| 0.38-0.58); bin 3, <= 0.07%
  everywhere (clipped |k| 0.46-0.58). The excess spreads to every k, not only the
  clipped shells.

**One realization per bin on `configs/v28_halofit.yaml`'s grids (2026-10-01, laptop,
realization 0, fiducial mask): every bin stops at the field-stage buffer guard.**
`f max|Psi|` (Mpc/h) against the 150 buffer, and the field stage's peak RSS:

| bin | N | f max\|Psi\| | buffer needed | peak RSS |
|---|---|---|---|---|
| 1 | 270 | 163.4 | 168.2 | 2.8 GB |
| 2 | 450 | 187.2 | 192.0 | 9.4 GB |
| 3 | 720 | 329.9 | 334.1 | 27.6 GB |
| 4 | 896 | 157.9 | 162.2 | 53.7 GB |

Bins 5-7 (1024^3) not run. The matter field is the linear one, so this is the finer
grid, not halofit. Bin 3, realization 0's ic seed, |Psi| over all cells at N 384 vs 720:
median 6.6 / 6.5, p99 18.5 / 18.4, p99.99 46.1 / 49.2, p99.9999 109 / 136, max 140 /
437 Mpc/h; 0 vs 45 cells with `f |Psi| > 150`, the top five 436.6, 436.5, 432.0, 430.4,
427.6 (one cluster). The bulk of the displacement field is grid-independent; the
excess is a few cells around one extreme matter peak. Raising the buffer does not fit
the boxes (`rmax + buffer <= L/2`: bin 1 allows 181, bin 2 172, bin 3 219). Open: JC's
call on the velocity construction at these grids.

**Production grids, as run (2026-10-01, laptop).** G15 (`tests/test_gates_m4.py`, 0
clipped modes, galaxy sigma^2 5.45): 60 bands to k 0.396, SE 0.06-0.53%, floor met in
every band; galaxy (halofit) max |z| 1.62, matter (linear) 1.97. Seven-bin realization 0
of `configs/v28_halofit.yaml` (fiducial mask): 651,451,370 galaxies, 15.04 GiB, layout
ok, realized/target 0.9987-1.0078, 55 s wall (laptop not checked idle), peak RSS
14.0 GB. The first attempt, with bin 7 on the halofit table, raised in bin 7's field
stage (5 clipped modes) as the rule requires.
Full suite 124 passed (8:53), `check-format` + `lint` clean, `m1_reproducibility --n
64` IDENTICAL.

Open, not scheduled: the velocity construction on finer cells (above: one extreme
matter peak sets `max|Psi|`); a test would be Psi from the matter field band-limited
per axis to the production Nyquist, which removes no mode on the production grids.

## M5 -- Ensemble production

Replaces prod_v2, which is retired (JC, 2026-10-01). Scope (JC): 100 realizations to
start, f_NL = 0, the halofit galaxy target of `configs/v28_halofit.yaml` (bins 1-6;
bin 7 keeps the linear target, M4), on deneb.

### Construction (planned 2026-10-01)

**Growth rate.** Drop-in fidelity to prod_v2 no longer constrains `Bin.f`. It is now
the exact linear growth rate `f = dln D / dln a` of the flat-LCDM background the
package already uses for distances (`OMEGA_M_DISTANCE` = 0.3153) and for f_NL's
`g0` (`fnl.growth_md`): `f = -3/2 Om(a) + 1 / (a^2 E(a)^3 I(a))`, with
`I(a) = int_0^a da' / (a' E(a'))^3` the Heath integral. `fnl.growth_rate_md`
computes it; `suite.py` stores the literals (dependency-free) and a test recomputes
them. The CAMB tables' cosmology (Planck 2018, `make_pk_tables.PLANCK18`) has
Om0 = 0.3152 (cdm + baryon + nu), the same background. Measured against
alternatives at the v28 z_eff (2026-10-01):

| z | prod_v2 (astropy Planck18 Om^0.55) | Om(z)^0.55, Om0 0.3153 | exact, Om0 0.3153 | CAMB f sigma8 / sigma8 |
|---|---|---|---|---|
| 0.1 | 0.58193 | 0.58733 | 0.58572 | 0.58654 |
| 0.3 | 0.67977 | 0.68520 | 0.68490 | 0.68587 |
| 0.5 | 0.75571 | 0.76092 | 0.76123 | 0.76231 |
| 0.7 | 0.81277 | 0.81765 | 0.81820 | 0.81936 |
| 0.9 | 0.85505 | 0.85961 | 0.86020 | 0.86141 |
| 1.3 | 0.90961 | 0.91364 | 0.91414 | 0.91538 |
| 1.9 | 0.95058 | 0.95417 | 0.95449 | 0.95572 |

prod_v2 sits 0.5-0.9% below CAMB. The exact flat-LCDM rate sits 0.13-0.14% below it
at every z, a constant offset from what the flat-LCDM integral leaves out (radiation;
neutrinos counted as clustering matter). Every catalog and the v28 config hash change.

**Layout on deneb.** One Slurm job, `--cpus-per-task=64 --exclusive` (only deneb has
64 CPUs), no GPU (the 3050 holds 3 of 7 bins). Two streams, `taskset -c 0-15` and
`-c 16-31` (the 32 physical cores; SMT siblings idle), each running realizations as
one process apiece: `logunusual run` -> `logunusual check` -> sha256 into a manifest.
`sample.default_workers` and the XLA-CPU pool both follow the affinity mask. Existing
catalogs are skipped (atomic rename: a file with the final name is complete), a failed
realization is logged and the loop moves on, and three failures in a row stop the
stream, so the same job resubmitted fills gaps. Output on `/work` (~1.6 TB at
15.04 GiB per realization). Bitwise reproduction of a catalog needs the same 16-core
affinity: XLA-CPU's reduction order follows the thread pool.

### Checks and gates

| check | statistic | where |
|---|---|---|
| f | `suite` literals equal `fnl.growth_rate_md(z, OMEGA_M_DISTANCE)` to round-off; `growth_rate_md` matches a centred finite difference of `ln growth_md` in `ln a` within the difference's own truncation bound | fast, `tests/test_suite.py` |
| pre-push | full suite (the slow RSD gates G6/G10 re-run with the new f), format, lint, `m1_reproducibility --n 64` IDENTICAL | laptop |
| pilot | realizations 0 and 50 together: `check` ok, realized/target inside realization 0's previous range (0.9987-1.0078; same seeds, f moved < 1%), peak RSS < 20 GB per process; per-realization wall and write MB/s recorded, which price the full run | deneb |
| E1 | 100/100 catalogs pass `check`; one config hash; 700 distinct seeds; 100 manifest entries | `scripts/m5_ensemble.py` |
| E2 | per bin, the ensemble mean of realized/target nbar is 1 within 3 SE (SE from the scatter over realizations): the lognormal is normalised to its box mean and stationary, so the real-space density's expectation is exactly the target at every point (by symmetry on the periodic box); the radial RSD shift across a curved shell adds `3 sigma_s^2 (rmax - rmin) / (rmax^3 - rmin^3)` (sigma_s the rms radial `f Psi`), <= 1e-4 in every bin and reported; a wrong fsky, a mask-edge bias or a buffer leak shows here | `scripts/m5_ensemble.py` |

### Pilot and the displacement tail (2026-10-01, deneb jobs 2077 / 2078, commits `48963a8` / `a2a3b8b`)

Pilot (realizations 0 and 50 together, 2x16 cores): realization 0 passes (79 s,
peak RSS 11.9 GB, sha256 7 s, layout ok, realized/target 0.9987-1.0078, 15.04 GiB;
the array took 145-190 MB/s from one writer). **Realization 50 fails the buffer guard
in bin 1**: f max|Psi| = 183.5 Mpc/h, so the shell needs 190.3 against the 150
buffer, and bin 1's box allows at most 181.8 (`L/2 - rmax`; bin 2: 172.2).

Mechanism (`scripts/m5_psi_tail.py`, bin 1, the production field bitwise; f|Psi| in
Mpc/h, maxima over the radial window):

| | r0 | r50 |
|---|---|---|
| densest matter cell: delta_m (Gaussian field there) | 203 (5.17 sigma) | 289 (5.47 sigma) |
| max f|Psi|; its cell's offset from the densest cell | 106.7; far (34, 73, 34) | 183.5; adjacent (0, 1, 0) |
| point-mass f|Psi| from the densest cell alone at the max cell | 0.0 | 105.1 |
| max f|Psi| with delta_m -> 0 in the top 1 / 8 / 64 cells | 106.7 / 86.7 / 60.7 | 93.3 / 88.1 / 62.5 |
| |Psi| per-component rms; p99.999 | 5.65; 120.9 | 5.68; 131.5 |
| window cells (expected galaxies) with f|Psi| > 50 / 100 / 150 | 147 (9,335) / 5 (158) / 0 | 180 (11,756) / 6 (233) / 6 (233) |
| Psi from a Gaussian field, same white noise and target `P / sinc^2`: max f|Psi|; rms; cellwise corr | 19.4; 5.64; 0.87 | 18.7; 5.66; 0.87 |
| lognormal matter target without `/ sinc^2`: max f|Psi|; delta_m max; rms | 64.9; 98; 5.32 | 64.4; 91; 5.33 |

Linear continuity applied to the lognormal matter field turns each of its few densest
cells (delta_m ~ 200-300 at 5-5.5 sigma of the Gaussian field) into a point source of
displacement: f|Psi| ~ 100-180 Mpc/h in the neighbouring cells, 20-30x the rms. In
r50 a single cell sets the guard's requirement (removing it: 183.5 -> 93.3). The tail
reaches ~1e-4 of a bin's galaxies above 50 Mpc/h and ~2e-6 above 100. A Gaussian
field with the same target has the same displacement rms (two-point identical by
construction) and a maximum of ~19. The jitter deconvolution of the matter target
doubles the densest cells and the tail. The guard takes the GLOBAL max over the
window, so the most extreme of ~2.4M cells sets every realization's requirement.

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
  carrying 1e-8 of the power. Those 5 modes are the fundamental shell, and there the
  realized ensemble power is **1.109x the target** (exact, from the clipped P_G,
  2026-10-01); every other shell is exact. Grid sigma^2 of the galaxy field 2.3-3.3,
  matter 0.26-2.8.
- The input P(k) is linear and truncated at kh = 1; the honest reach of the mocks is set
  by the grid (ell ~150-300 sample-weighted at 512^3), see chimera memory
  `project_lognormal_vs_disco_resolution`.
