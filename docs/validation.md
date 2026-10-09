# Validation

What is checked, against what bar, and what the checks show. Every bar below is the
threshold the test code enforces; where a number is quoted as a result it is a
measurement of the current code. The checks are identities, limits or Poisson
statements: none compares to another mock code.

Contents: [how the checks are built](#how-the-checks-are-built) ·
[running them](#running-the-checks) · one section per check, from
[lognormal grid identity](#lognormal-grid-identity) to
[ensemble integrity and density](#ensemble-integrity-and-density) ·
[fast unit tests](#fast-unit-tests).

## How the checks are built

**Per realization, then averaged.** Each statistic (a band power ratio, a count ratio)
is formed in every realization and the ratios are averaged. The standard error is the
scatter across realizations, `std(ddof=1) / sqrt(n_real)`. It is never a Gaussian
formula: the high-k power of a lognormal field is dominated by rare peaks, so its
realization scatter is far above Gaussian and correlated across k.

**Tolerance and bar.** `gates.TOL = 2%`. A *floored* check requires, in every band,

- `SE <= TOL / 3 = 0.667%`, so that a 2% bias would show at 3 sigma, and
- `|z| < 4` with `z = (mean - target) / SE` (`gates.Z_MAX`; probability 6e-5 per band
  under the null).

A *consistency* check keeps `|z| < 4` and drops the SE floor. Each section says which
kind it is and why.

**Bands.** k-shells of width `k_f` are merged from low k upward until a band holds at
least `n_min` independent modes (`gates.band_edges`), with

    n_min = ceil(9 / (TOL^2 n_real))        (gates.n_min_indep)

so that a Gaussian-scatter SE, `1 / sqrt(n_indep n_real)`, meets `TOL / 3`:

| realizations the bands are built for | 16 | 32 | 64 | 192 |
|---|---|---|---|---|
| `n_min` (independent modes per band) | 1407 | 704 | 352 | 118 |

**Independent modes.** A shell holds `nmodes / 2` independent modes (`n_indep` in
`validate.power_multipoles`), with `nmodes` the Hermitian-weighted full-grid count.
Every case gives the same variance per unit weight: an interior half-grid cell stands
for itself and its conjugate; on the kz = 0 and Nyquist planes both members of a pair
are half-grid cells with the same `|delta_k|^2`; a self-conjugate mode is real, so its
`|delta_k|^2` has twice the variance. Hence `validate.gaussian_se = sqrt(2 sum w P^2) /
sum w`, and bands are sized on `nmodes / 2`. Counting half-grid cells instead would
overcount the low shells (by half again at the fundamental) and understate their SE;
counting the full grid would understate every SE by sqrt 2.
`tests/test_validate.py::test_gaussian_se_is_the_realization_scatter_at_low_k` checks
the formula against the scatter of 4000 Gaussian realizations shell by shell. The
checks' pass/fail SEs are the measured scatter across realizations in any case; the
formula sizes the bands.

**Bands vs seeds.** `n_min x n_real` is fixed, so re-deriving the bands for more seeds
leaves the worst band's Gaussian SE pinned near the floor: for the shot-noise check,
0.591% at 32 seeds, 0.651% at 64 and 0.635% at 128, against 0.667% (the headroom is
how far merged shells overshoot `n_min`). The lognormal and Poisson scatters are
super-Gaussian and can spend that headroom. Checks that need margin therefore run more
seeds through bands built for fewer: `band_seeds=32` with 64 seeds.

**Own seed range.** Every check draws its realizations from its own seed range:

| check | seeds | used as |
|---|---|---|
| lognormal grid identity | 3000-3191 | field seed |
| uniform-field shot noise | 4000-4063 | draw seed |
| fixed-field sampler | field 4100, draws 4200-4215 | field / draw seed |
| catalog power, RSD, density | 5000-5031 | split into (field, draw) |
| Kaiser linear limit | 6000-6031 | split |
| shell density and profile | 7000-7015 | split |
| estimator response, generator mesh | 8000-8063 | split |
| grid identity, f_NL = +100 / -100 | 8000-8191 / 8200-8391 | field seed |
| matched-seed f_NL ratio, +100 / -20 | 8400-8431 / 8500-8531 | split |
| grid identity, halofit target | 8600-8791 | field seed |

"Split" means `sample.split_seed(s) -> (ic, draw)` through `SeedSequence.spawn`; the
spawned field seed of `s` is not `s`, so the two checks that share the integers
8000-8063 use different realizations.

**Estimator.** Catalogs are painted with CIC on an estimator mesh twice the generator
mesh (`box_est`), deconvolved, and the shot noise is subtracted with the Jing (2005)
alias sum (`validate.shot_noise_k`). A catalog drawn from a grid field is
lattice-periodic, so its alias images carry the same Fourier coefficient up to a sign
`(-1)^(r n)` (`r` the mesh ratio, from the half-cell offset of the cell centres) and
add *coherently*. `validate.estimator_response` is that exact, separable correction.
On a 2x mesh every sign is positive and the deconvolved power reads 0.2% low at half
the generator Nyquist and 3.5% low at the Nyquist (shell averages); on the generator's
own mesh the dominant image subtracts and it reads 3.5% low at half the Nyquist (shell
average; 6% along an axis). Every catalog check either divides by the response or
carries it in its prediction.

**Box geometry.** Unless a section says otherwise the periodic-box checks run on a
128^3 grid with L = 1000 Mpc/h (dx = 7.8 Mpc/h, a production-like cell), the bin-5
bias b = 1.76 and growth rate f from `suite.py`, nbar = 3e-3 (h/Mpc)^3, and the z = 0.9
linear CAMB table in `tests/data/`.

## Running the checks

```sh
pixi run test                                   # full suite: fast + slow (~9 min)
pixi run python -m pytest -m "not slow"         # fast unit tests only
pixi run python -m pytest tests/test_gates_box.py -s     # one file, per-band tables
pixi run python -m pytest tests/test_gates_box.py -s -k grid_identity
```

`slow` is opt-out: the full suite is the merge gate and runs every statistical check.
The slow checks live in `tests/test_gates_box.py` (periodic box),
`tests/test_gates_shell.py` (shell product) and `tests/test_gates_targets.py` (f_NL
and halofit targets); with `-s` each prints its per-band mean, SE and z.

The same functions (`logunusual/gates.py`) back two scripts that run larger
configurations and write JSON (gitignored) under `runs/`:

```sh
pixi run python scripts/gates_box.py   [--n 128] [--L 1000] [--seeds 32] \
    [--seeds-shot 64] [--band-seeds 32] [--seeds-field 192] [--nbar 3e-3] \
    [--bin 5] [--pk PATH]
pixi run python scripts/gates_shell.py [--n 128] [--L 1000] [--seeds 16] \
    [--seeds-response 64] [--band-seeds 32] [--shell RMIN RMAX BUFFER] [--pk PATH]
```

The scripts' `--pk` defaults to `data/<bin's table>`; on a checkout without `data/`,
pass `--pk tests/data/matterpower_camb_zeff=0.9.tsv`. Their seed defaults are not
the tests' (`gates_box.py --seeds` sets the shot-noise, catalog and Kaiser seed counts
together, and only the shot-noise check takes `--band-seeds`), so a script run is a
larger or different configuration, not a replay of the test.

## Lognormal grid identity

**Tests.** The field stage's central identity. For a periodic Gaussian field with grid
spectrum `P_G`, `exp(G) / <exp G> - 1` has grid two-point function exactly
`exp(xi_G) - 1 = xi` in the ensemble (Coles & Jones 1991), so a field built from
`P_G = FFT[log(1 + xi)]` has power `P_grid` on every mode up to the Nyquist, as long
as no `P_G` mode is clipped. `P_grid` is the deconvolved target
`b^2 P / prod sinc^2` (galaxy) or `P / prod sinc^2` (matter).

**Statistic and bar.** `<P(delta)> / P_grid` per band, measured on the generator grid
(no sampling), galaxy and matter fields; 192 realizations, bands to the Nyquist.
Floored.

**Runs in.** `tests/test_gates_box.py::test_lognormal_grid_identity`
(`gates.gate_field_identity`); `scripts/gates_box.py`.

**Result.** 60 bands to the Nyquist; galaxy max |z| 2.0, matter 2.2; SE 0.06-0.56%;
no clipped `P_G` modes. Galaxy grid variance sigma^2 = 3.9.

## Uniform-field shot noise

**Tests.** The sampler and estimator alone: a constant intensity `nbar V_cell`, drawn
on the per-slab streams and placed uniformly in each cell exactly as a real catalog,
must have, after CIC deconvolution on the 2x mesh, the Jing (2005) shot spectrum.

**Statistic and bar.** Measured monopole over `validate.shot_noise_k` per band, to the
generator Nyquist; 64 realizations on bands built for 32. Floored.

**Runs in.** `tests/test_gates_box.py::test_uniform_field_shot_noise`
(`gates.gate_uniform_shot`); `scripts/gates_box.py`.

**Result.** Lowest band (k = 0.035 h/Mpc, the band with least SE headroom):
1.0004 +- 0.0051, SE 0.51% against the 0.667% floor.

## Fixed-field sampler

**Tests.** Placement, Poisson sampling and the estimator, with cosmic variance
removed: one fixed field, many draws. The prediction
(`gates.catalog_power_prediction`) is the exact per-mode expectation of the
deconvolved, shot-free mesh power for that field: its grid power mapped onto the
estimator mesh's alias images, times `validate.effective_window` (coherent sum of CIC
window x placement window) over the CIC deconvolution.

**Statistic and bar.** Draw-averaged catalog monopole over the prediction, one field
(seed 4100), 16 draws. Floored in the first Brillouin zone (to the generator Nyquist);
reported to the estimator Nyquist (the alias images) with `|z| < 4` and no floor.

**Runs in.** `tests/test_gates_box.py::test_fixed_field_sampler`
(`gates.gate_fixed_field_sampler`); `scripts/gates_box.py`. Per-band values are printed
with `-s`.

**Result.** First zone: 51 bands, max |z| 2.0, SE 0.11-0.26%, floor met in every band.
To the estimator Nyquist: 115 bands, max |z| 2.0.

## Catalog monopole

**Tests.** The whole real-space chain, field + sampler + estimator: the catalog
carries `b^2 P_in` in the first zone.

**Statistic and bar.** Per realization, shot-subtracted deconvolved monopole over two
references, each floored, k < k_Nyq / 2, 32 realizations:

- `b^2 P_in x estimator_response` (ensemble statement);
- the fixed-field prediction from the realization's own field (removes cosmic
  variance, so it is the tighter of the two).

**Runs in.** `tests/test_gates_box.py::test_catalog_power_rsd_and_density`
(`gates.gate_catalog`, keys `monopole`, `monopole_fixed_field`); `scripts/gates_box.py`.
Per-band values are printed with `-s`.

**Result.** 23 bands to k = 0.195 h/Mpc. Against `b^2 P_in x estimator_response`: max
|z| 1.8, SE 0.35-0.61%. Against the fixed-field prediction: max |z| 2.7, SE
0.09-0.17%. Floor met in every band of both.

## Redshift-space distortions at production amplitude

**Tests.** The same 32 realizations in redshift space, plane-parallel: each galaxy
moves by `f Psi_z` of its own cell. Linear RSD (Kaiser) is a k -> 0 limit. At
production amplitude `f Psi_rms` is about 3.2 Mpc/h, so `k f Psi_rms` is ~0.6 at
k = 0.19, and the lognormal galaxy and matter fields are not perfectly correlated at
finite k: on the generator grid `P_gm / (b P_mm)` falls from 0.993 at k = 0.035 to
0.965 at k = 0.19.

**Statistic and bar.** Asserted: `(P_2 / P_0) / Kaiser(f / b)` in the lowest band,
`|z| < 4`, no floor. Measured and printed, not asserted: the same ratio in every band;
`P_2` and `P_0` over the linear prediction built from the realization's own grid
spectra (`(4f/3) P_gm + (4f^2/7) P_mm`, `P_gg + (2f/3) P_gm + (f^2/5) P_mm`); and
`P_gm / (b P_mm)`. The quadrupole rises above Kaiser with k; the printed tables show by
how much.

**Runs in.** `tests/test_gates_box.py::test_catalog_power_rsd_and_density` (keys
`kaiser_lowest_band`, `kaiser_all_bands`, `quadrupole_generalised`,
`monopole_generalised`, `premise`); `scripts/gates_box.py`.

**Result.** `f Psi_rms = 3.25` Mpc/h. Lowest band (k = 0.035): `(P_2 / P_0) /
Kaiser` = 1.002 +- 0.031. At k = 0.195 the same ratio is 1.24 +- 0.01, `P_gm /
(b P_mm)` is 0.965, and the redshift-space monopole sits 4.8% above its linear
prediction.

## Kaiser linear limit

**Tests.** Kaiser as an exact statement. With the input spectrum scaled by
`eps = 1e-2` the lognormal fields are Gaussian to O(eps), the galaxy-matter
correlation is 1 to O(eps), and the redshift-space mapping is linear to
O((k f Psi_rms)^2), so `P_2 / P_0 = Kaiser(f / b)` in every band.

**Statistic and bar.** `(P_2 / P_0) / Kaiser(f / b)` per band, k < k_Nyq / 2, on a
64^3, L = 500 generator with a 128^3 estimator, nbar = 0.1 (dense, so shot noise stays
below the small signal), 32 realizations. Consistency: `|z| < 4`, no floor; the
quadrupole ratio's SE is a few percent per band here. The nonlinear budget
`(k_max f Psi_rms)^2` is about 4e-3 at the band top and is printed with the result.
The deterministic exactness of the displacement is carried by fast tests (see
[fast unit tests](#fast-unit-tests): own-cell displacement, constant displacement,
plane-wave displacement).

**Runs in.** `tests/test_gates_box.py::test_kaiser_linear_limit`
(`gates.gate_kaiser_linear_limit`); `scripts/gates_box.py`.

**Result.** 7 bands to k = 0.19, max |z| 2.0, SE 1.6-3.7%; `f Psi_rms = 0.32` Mpc/h,
nonlinear budget 4.1e-3 and shot noise 0.41 of the signal at the band top.

## Density

**Tests.** The catalog's mean density and Poisson statistics, in the same 32
realizations as the catalog monopole.

**Statistic and bar.** Per realization `z = (N - sum lambda) / sqrt(sum lambda)` and
`sum lambda / (nbar V)`. Pass: `|mean z| < 4 / sqrt(32) = 0.71`, sample std of z in
(0.5, 1.6), and `|sum lambda / (nbar V) - 1| < 1e-10` in every realization (the
lognormal is normalised to its box mean, so the expected count is exact).

**Runs in.** `tests/test_gates_box.py::test_catalog_power_rsd_and_density` (key
`density`).

**Result.** Mean z -0.14, std 0.80; `sum lambda / (nbar V)` within 1.3e-15 of 1.

## Reproducibility

**Same seed, same bytes, across processes.**
`tests/test_run.py::test_two_processes_write_the_same_bytes` runs the CLI in two fresh
processes, one with 1 sampler thread and one with 4: on Linux the two catalog files are
byte-identical; on macOS the test checks the same galaxy count and positions to 1e-9
Mpc/h. `scripts/reproducibility.py [--n 64] [--L 500]` runs the field stage and a
catalog in two subprocesses and compares `delta_g`, `psi_z`, positions and cell
indices; on Linux it exits 1 unless identical, elsewhere it reports the differing
cells. On a 16-core Apple M4 Max the 64^3 run reports IDENTICAL.

**Thread count.** Each x-slab draws on its own Philox stream, keyed by
`draw_seed | slab << 64` (`sample.slab_rng`), and slabs are yielded in order, so a
catalog depends on `(draw_seed, field)` alone and is the same for any thread count by
construction. Asserted bit for bit by
`tests/test_shell.py::test_sample_shell_is_the_same_for_any_thread_count` (3, 8 and
all cores against 1).
`tests/test_sample.py::test_slab_streams_are_deterministic_distinct_and_local`
checks that neighbouring slabs' and seeds' streams share no values and that one
slab's draws do not move when another's counts change.

**Platform.** XLA on macOS-arm64 CPU can differ in the last bit between two runs of
the same program, so in-process and cross-process repeats are asserted bitwise on
Linux and characterised on macOS (`max relative difference < 1e-12`,
`tests/test_field.py::test_in_process_reproducibility_is_bitwise_or_characterised`).
The ensemble script pins each process to a fixed set of 16 cores, because XLA-CPU's
reduction order can follow its thread pool; reproduce a catalog bit for bit with the
same affinity. CPU and CUDA field stages are not bitwise equal (they differ at the
1e-13 level; see `docs/performance.md`).

**jit.** `generate_fields(..., jit=True)` is not bit-preserving: XLA fuses and
reassociates, moving a third to most of the cells by round-off.
`tests/test_field.py::test_jit_is_not_bit_preserving_but_is_round_off` bounds the move
at 20 float64 eps of the field's own maximum (measured 0.25-4) and fails if jit
happens to move nothing. `jit` is off by default and is not a `RunConfig` field.

## Shell density and radial profile

**Tests.** The shell product end to end at production amplitude: observer-centred
radial window with buffer, radial RSD, angular mask, inclusive shell cut. A missing
buffer or a wrong edge crossing would show as a deficit in the edge sub-shells.

**Setup.** 128^3, L = 1000; shell [300, 371.1] Mpc/h (bin 5's thickness scaled into
the box) with a 90 Mpc/h buffer; an equatorial NSIDE-32 band mask, `|cos theta| < 0.7`
(fsky 0.70); nbar = 3e-3; 16 realizations; 8 radial sub-shells.

**Statistic and bar.**

- `N_kept / (nbar fsky V_shell)`, and `n(r) / nbar` in each sub-shell including both
  edges: consistency, `|z| < 4`, no floor. The SE is the field's own sample variance
  on the shell scale (a few percent per realization), not Poisson, so a 2%/3 floor
  does not apply.
- Draws over the buffered window against `sum lambda`: per-realization Poisson z with
  `|mean| < 4 / sqrt(16) = 1` and sample std in (0.5, 1.6).
- The number of galaxies that left the box is reported (such a galaxy is beyond
  `rmax + buffer` and is never kept).

**Runs in.** `tests/test_gates_shell.py::test_shell_density_and_profile`
(`gates.gate_shell_density`); `scripts/gates_shell.py`. Per-sub-shell values are
printed with `-s`.

**Result.** `N_kept / (nbar fsky V_shell)` = 1.009 +- 0.006 (z 1.4); sub-shells
0.997-1.024, max |z| 2.3, both edge sub-shells within 1.5 sigma; draws mean z -0.03,
std 1.20; no galaxy left the box.

## Estimator response on the generator's own mesh

**Tests.** The coherent-alias response where it is largest: with the estimator on the
generator's own mesh the dominant alias image subtracts and the deconvolved power reads
3.5% low at half the Nyquist (shell average). The 2x-mesh checks cannot see the image
signs (all positive there); this check can.

**Statistic and bar.** The catalog monopole check on a 1x mesh: monopole over
`b^2 P_in x estimator_response`, and over the fixed-field prediction, k < k_Nyq / 2,
64 realizations on bands built for 32. Both floored.

**Runs in.** `tests/test_gates_shell.py::test_one_x_mesh_response` (`gates.gate_catalog`
with `box_est = box`); `scripts/gates_shell.py`. The sign itself is pinned against a
hand sum (relative 1e-12) by
`tests/test_validate.py::test_effective_window_carries_the_half_cell_image_sign`.

**Result.** 23 bands to k = 0.195; the response itself departs from 1 by up to 3.5% in
these bands. Against `b^2 P_in x estimator_response`: max |z| 1.5, SE 0.20-0.47%.
Against the fixed-field prediction: max |z| 2.0. Floor met in every band of both.

## f_NL: M(k) normalisation

**Tests.** The Poisson factor `M(k) = sqrt(P / P_Phi) / g0` built from the bin's own
table (`fnl.poisson_M`, LSS convention) against its large-scale limit
`(2/3) (c/H0)^2 k^2 D / Omega_m` (T = 1) at the table's lowest node.

**Bar (derived).** `|1 - M / M_{T=1}|` must be within
`(1 + z) a_eq + 2.2 q_min`: radiation, which the growth integral omits, shifts D by
order `(1 + z) a_eq`, and T(k) departs from 1 by at most ~2.2 q at low k
(`q = k / (Omega_m h)`). The reference takes its growth from the integral directly,
so a wrong normalisation inside `fnl` cannot cancel against it.

**Mutation tests.** Each of four wrong normalisations must fail the same bound:

| mutation | residual |
|---|---|
| none (z = 0.9 table) | 7.7e-4 against a bound of 1.59e-3 |
| pivot 0.05 read as h/Mpc instead of 1/Mpc | 6.2e-3 |
| A_s off by 2% | 1.1e-2 |
| CMB normalisation where LSS is meant (g0 = 1) | 0.21 |
| missing 9/25 in `P_Phi` | 0.40 |

On all seven v28 tables the residual is 4.5e-4 (z = 0.1) to 1.25e-3 (z = 1.9), each
1.5-3x inside its own bound.

**Runs in.** `tests/test_fnl.py::test_m_matches_the_large_scale_limit`,
`test_m_check_fails_on_a_wrong_normalisation`; with `test_growth_integral` (EdS gives
g0 = 1; D -> a at high z) and `test_bias_sign_scaling_and_null_cases` (sign,
`Delta b ~ k^-2`, linear in f_NL, `b = p` gives no response).

## Grid identity with f_NL bias

**Tests.** The lognormal grid identity with the scale-dependent galaxy target
`b(k)^2 P`, `b(k) = b + 2 (b - p) f_NL delta_c / M(k)`. At f_NL = +100 / -100 on this
box `b(k_f) / b` = 1.35 / 0.65, and neither clips a `P_G` mode.

**Statistic and bar.** As the [lognormal grid identity](#lognormal-grid-identity):
galaxy and matter, 192 realizations, every band to the Nyquist, floored; and the
largest clipped power fraction must be exactly 0.

**Runs in.** `tests/test_gates_targets.py::test_grid_identity_with_fnl_bias`
(parametrised over f_NL = +100, -100).

**Result.** 60 bands to the Nyquist, SE 0.05-0.6%, floor met in every band.

| f_NL | `b(k_f) / b` | galaxy max \|z\| | matter max \|z\| |
|---|---|---|---|
| +100 | 1.35 | 3.25 | 2.73 |
| -100 | 0.65 | 2.19 | 2.54 |

At f_NL = +100 the galaxy ratio sits coherently low above k = 0.2 (mean -0.26% over 32
bands; the largest |z|, 3.25, is at k = 0.35). Nothing is clipped, so the identity is
exact there; the lognormal's high-k scatter is correlated across k, which makes a
coherent excursion of this size consistent with the bar.

## Matched-seed f_NL catalog ratio

**Tests.** That the run path hands the catalog the injected b(k): same field seed and
draw seed with and without f_NL, so most of the cosmic variance cancels in the ratio.

**Statistic and bar.** Per realization, real-space shot-subtracted monopole with f_NL
over without, divided by the predicted `<b(k)^2 P R> / <b^2 P R>` (R the estimator
response, band-averaged with the measurement's mode weights); one band per k_f shell
to k = 0.05; 128^3, L = 2000 (k_f = 3.1e-3 h/Mpc), 256^3 estimator, nbar = 1e-3, 32
realizations. Consistency: `|z| < 4`, no floor. The 2%/3 floor is sized to the catalog
power itself; here the lowest shells stay noisy under matched seeds (per-realization
scatter ~26% at k_f, ~10% at 2 k_f), because part of the low-k lognormal power is a
white term built from the high-k modes that is nearly the same in both arms but does
not scale with the shell's own amplitude. Meeting the floor per shell would take ~1500
realizations at k_f. The exact, floored statement is the grid identity above; the
sampler is linear in the intensity and is checked on its own above.

**Runs in.** `tests/test_gates_targets.py::test_fnl_matched_seed_catalog_ratio`
(`gates.gate_fnl_ratio`; f_NL = +100 and -20, the values this box can carry without
clipping).

**Result.**

| f_NL | predicted ratio at k_f | measured / predicted at k_f | max \|z\| |
|---|---|---|---|
| +100 | 3.48 | 1.074 +- 0.040 | 1.97 |
| -20 | 0.69 | 0.987 +- 0.024 | 2.16 |

## Grid identity with a halofit galaxy target

**Tests.** The grid identity with a second, nonlinear table for the galaxy target
only: galaxy `b^2 P_halofit`, matter `P_lin`, at z = 0.9 from the kh-10 tables in
`tests/data/`.

**Statistic and bar.** As the [lognormal grid identity](#lognormal-grid-identity),
192 realizations, floored; also the pair check must return exactly 0, nothing may be
clipped, and the galaxy grid variance must exceed 5 (the halofit target, not the
linear one).

**Runs in.**
`tests/test_gates_targets.py::test_grid_identity_with_halofit_galaxy_target`.

**Result.** 60 bands to k = 0.396, SE 0.06-0.53%, floor met in every band; galaxy
(halofit) max |z| 1.62, matter (linear) max |z| 1.97; galaxy grid sigma^2 = 5.45, no
clipped modes.

## Table-pair check

**Tests.** That a galaxy table and its linear table describe the same redshift
(`pk.check_table_pair`, called when a run loads its tables). They must share their
lowest k node, and there `|P_gal / P_lin - 1|` must be within `k0^2 sigma_v^2`, the
leading low-k nonlinear correction (`sigma_v^2 = int P dk / 6 pi^2` from the linear
table); otherwise it raises.

**Result.** Bound 6.4e-8 (z = 1.9) to 3.1e-7 (z = 0.1) over the seven v28 pairs;
measured exactly 0 on every pair. A pair one bin apart in z is off by 0.17-0.35, about
10^6 times the bound. The kh-10 linear tables differ from the kh-1 v28 tables by up to
1.2e-5 in P, more than the bound, so a halofit table pairs only with the kh-10 linear
table.

**Runs in.** `tests/test_pk.py::test_table_pair_check` (bound value, an exact pair, a
damped pair inside the bound, offsets of +-2x the bound and 20% raise, mismatched k
raises).

## Table regeneration

**Tests.** That the input linear tables can be rebuilt exactly.
`pixi run -e tables pk-tables check [--pk-dir data]` regenerates the seven v28 linear
tables (kh 1e-4 to 1, 10001 nodes) with the pinned CAMB build and compares them byte
for byte with the files the default bin table names (`Bin.pk_file`,
`matterpower_camb_zeff=<z>.tsv`), as written by `pk-tables make-default`; on a mismatch
it prints the maximum relative difference and exits 1.

**Result.** Byte-identical (about 26 s with `OMP_NUM_THREADS=1`). The `tables`
environment is osx-arm64 only.

## Buffer guard and angular pre-cut

The shell product draws galaxies only in a buffered radial window and, with a mask,
only in cells that can feed a set pixel. Both cuts must be exact supersets of the
cells whose galaxies can land in the shell. These are fast, deterministic tests in
`tests/test_shell.py`:

| test | asserts |
|---|---|
| `test_required_buffer_is_the_exact_feeding_condition` | `Shell.required_buffer` equals an independent full-grid computation of the line-of-sight interval test (rel 1e-13); without RSD it is at most the half cell diagonal; `sample_shell` records `psi_max` and `required_buffer` |
| `test_no_undrawn_cell_reaches_the_shell_at_the_required_buffer` | radial displacements planted on both sides; at buffer = requirement, no corner or interior point of any undrawn cell lands in the shell; the guard passes at the requirement and raises at `requirement x (1 - 1e-9)` |
| `test_line_of_sight_guard_is_complete_for_any_direction` | the same completeness with random-direction displacements; the requirement never exceeds the `f |Psi_c|` magnitude bound |
| `test_line_of_sight_guard_counts_a_tangential_displacement_through_the_chord` | a tangential displacement that reaches the shell only through the chord term is counted, at the exact requirement |
| `test_an_outward_displacement_beyond_the_window_does_not_count` | a large outward displacement beyond the window leaves the requirement unchanged; the same displacement inward raises |
| `test_a_large_displacement_counts_only_where_it_can_feed_the_shell` | a large displacement in a drawn cell inside the shell does not raise; in an undrawn cell next to the window it does |
| `test_angular_precut_is_an_exact_superset_of_the_feeding_cells` | band and sparse masks: every kept galaxy, drawn from the full radial window with RSD, comes from a cell the pre-cut keeps; the pre-cut is a strict subset of the radial window |
| `test_angular_precut_leaves_uncut_slabs_bitwise` | a mask excluding a cap around +x cuts only the high-x slabs; every other slab (holding >1000 kept galaxies) yields identical galaxies with and without the pre-cut, and at least one cut slab differs |
| `test_mask_dilation_is_the_centre_distance_set` | the mask dilation the pre-cut uses equals the set of pixels within the radius, by brute force |
| `test_sample_shell_uniform_field_is_poisson_in_the_shell` | `P_in x 1e-4`: `N_kept ~ Poisson(nbar fsky V_shell)` and draws `~ Poisson(sum lambda)`, each `|z| < 4`; flat radial profile; no kept galaxy outside the box |

`tests/test_run.py::test_uniform_input_gives_poisson_counts_through_the_driver` repeats
the uniform-field Poisson statement through the whole driver (window, RSD, mask,
writer), `|z| < 4` per bin.

## Ensemble integrity and density

**Tests.** A produced ensemble, from its `summary.json` files (the catalogs need not be
present):

```sh
pixi run python scripts/ensemble_check.py RUN_DIR [--manifest PATH] [--n 100] \
    [--out runs/ensemble/ensemble_check.json]
```

- **Integrity.** Realizations `0..n-1` all present and no others; one config hash;
  every per-bin seed, field seed and draw seed distinct; with `--manifest`, the
  job's sha256 manifest (one line per catalog that passed `logunusual check`) lists
  exactly those realizations.
- **Density.** Per bin, the ensemble mean of `realized_over_target` (kept count over
  `nbar fsky V_shell`) against `1 + edge`, with SE from the scatter over
  realizations; pass `|z| <= 3`. The lognormal is normalised to its box mean and
  stationary, so the real-space density's expectation is the target everywhere; the
  radial RSD shift across a curved shell adds a second-order excess

      edge = 3 sigma_s^2 (rmax - rmin) / (rmax^3 - rmin^3)

  with `sigma_s` = f times the rms of one displacement component. A wrong fsky, a
  mask-edge bias or a buffer leak would show here.

Exit code 0 only if both pass.

**Result: 100 realizations of `configs/v28_halofit.yaml`** (f_NL = 0, radial buffer
160 Mpc/h, survey mask). Integrity passes: one config hash, 700 distinct seeds, the
manifest is exactly realizations 0-99.

| bin | galaxies (mean) | realized / target | SE | edge | z | max required / buffer |
|---|---|---|---|---|---|---|
| 1 | 40,214,427 | 0.99939 | 0.00070 | 1.0e-4 | -1.02 | 0.943 |
| 2 | 132,865,592 | 0.99982 | 0.00034 | 1.8e-5 | -0.59 | 0.771 |
| 3 | 108,458,840 | 1.00002 | 0.00022 | 7.2e-6 | 0.08 | 0.705 |
| 4 | 123,317,319 | 1.00014 | 0.00017 | 3.9e-6 | 0.82 | 0.653 |
| 5 | 120,463,443 | 0.99997 | 0.00015 | 2.4e-6 | -0.20 | 0.560 |
| 6 | 102,408,693 | 1.00004 | 0.00007 | 1.2e-6 | 0.60 | 0.315 |
| 7 | 23,369,009 | 1.00003 | 0.00007 | 5.1e-7 | 0.37 | 0.205 |

The worst bin is bin 1 at z = -1.02. The largest line-of-sight buffer requirement over
all 700 bin-realizations is 150.9 Mpc/h (realization 2, bin 1), 0.943 of the 160
Mpc/h buffer; the next largest is 135.2 (realization 6, bin 1). The requirement has a
tail set by the lognormal matter field's densest cells: a realization beyond these can
need more, up to bin 1's box room of 181.8 Mpc/h, and the run raises if it does. Bin 7
keeps the linear galaxy target and reports a clipped power fraction of 1e-8 (its 5
fundamental modes); every other bin clips nothing.

## Fast unit tests

`pixi run python -m pytest -m "not slow"`. What each file covers:

| file | covers |
|---|---|
| `tests/test_grid.py` | box geometry, k-grid against FFT conventions, Hermitian weights count the full grid, sinc windows, CIC shot-noise alias factor limits |
| `tests/test_pk.py` | TSV loader and spline nodes, power-law tails; `pk_on_grid` per-radius table vs direct evaluation (index exact; bound `slope_max |dk/k| + 4 ulp(max |ln P|)`, ~37 eps at 32^3, measured max 9 eps); `grid_xi` vs a closed-form Gaussian pair; `grid_pkG` second-order expansion; rejects `xi <= -1` and reports clipping; JAX and numpy paths agree; table-pair check |
| `tests/test_field.py` | deconvolved target; Gaussian colouring reproduces `P_G` in the ensemble (48 seeds, all `|z| < 4.5`, mean z^2 in (0.4, 1.8)); lognormal has zero mean and `1 + delta > 0`; plane-wave displacement is analytic; divergence identity of the three displacement components; f_NL = 0 is the scalar-bias stage bitwise; unattainable f_NL and clipping galaxy tables raise; a galaxy table moves the galaxy field only; `dtype` (f32 stays f32, agrees with f64 within 100 float32 eps of the field maximum, measured 2-22) and `jit` (round-off only, within 20 float64 eps) |
| `tests/test_sample.py` | intensity sums to `nbar V`; placement inverts to counts and stays in the box; slab streams deterministic, distinct and local; slab-wise draw equals Poisson-then-place; Poisson total; constant displacement translates z; each galaxy moves by its own cell's displacement; `split_seed` |
| `tests/test_shell.py` | shell validation and `check_box`; radial window vs brute force and the continuum shell volume; mask from h5 in both orderings; selection with inclusive edges against an independent `ang2pix`; radial RSD identities and its far-observer plane-parallel limit; galaxies cross both shell edges; thread-count independence; buffer guard and angular pre-cut (table above); radial histogram including edges; real and redshift space share the draw |
| `tests/test_validate.py` | multipoles vs brute force; cross power and shot subtraction are linear; CIC conserves mass; uniform Poisson catalog has Jing shot noise; `gaussian_se` equals the scatter of 4000 Gaussian realizations shell by shell; bands hold `nmodes / 2` modes and close at the first shell reaching `n_min`; Kaiser formulae; rebinning; effective-window limits and the half-cell image sign |
| `tests/test_fnl.py` | M(k) normalisation and its mutations; growth integral; bias sign, `k^-2` scaling and null cases; f_NL = 0 bitwise; DC handling; zero-crossing diagnostics; M taken from the linear table when a galaxy table is given |
| `tests/test_io.py` | row-group layout and `check_layout`; bytes independent of how slabs arrive; metadata round trip and empty bins; `None` written as `none`; layout faults detected; descending bins rejected, temporary file cleaned up on error |
| `tests/test_run.py` | config YAML round trip and hash scope (f_NL and galaxy-table keys enter only when set); bin scaling; end-to-end realizations (Gaussian, f_NL, galaxy table); uniform input is Poisson through the driver; two processes write the same bytes; CLI commands |
| `tests/test_suite.py` | v28 bin table: indices, shells tile z and r without gaps, boxes hold the buffered shells, cell sizes; growth-rate literals equal `fnl.growth_rate_md` and match a finite difference of `ln D`; seed schedule is collision-free; table file names |
