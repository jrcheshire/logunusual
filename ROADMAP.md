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

- **Nonlinear input P(k):** halofit TSVs (a flag in `make_matter_power.py`) together
  with the 1024^3 grid the finer scales need; at 512^3 halofit alone changes little
  because the grid Nyquist binds first.
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
