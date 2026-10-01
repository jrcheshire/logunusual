# CLAUDE.md

Guidance for working in this repo. User-level preferences (`~/.claude/CLAUDE.md`) apply on
top of this and win on conflict. `README.md` is the collaborator-facing how-to; this file
is the agent-facing map: what the package is, the product contract it must honour, the
conventions that are easy to get wrong, and where the project stands.

## What this is

`logunusual` is a **lognormal galaxy mock generator: JAX for the field stage (CPU or
CUDA), numpy for per-cell Poisson sampling and placement, streamed parquet output.**
A run is a list of bins, each a periodic box centred on the observer holding one
radial shell; galaxies are drawn in the buffered shell, displaced radially by their
cell's velocity, cut to the shell and an optional angular HEALPix mask, and streamed
to one parquet per realization. The input P(k) interface is a plain TSV per bin (a
nonlinear spectrum is a different table, M4); local f_NL enters as a scale-dependent
galaxy bias built from that table (`fnl.py`, M4).

The **default inputs** are a survey forecast: the seven-bin v28 table in `suite.py`
(shells partitioning z = 0-2.2, densities, biases, growth rates) and, when given, a
survey mask. Nothing else in the code knows about a survey; any bin table with the
same fields can be given in a run config. No other repository is a dependency, a
consumer named in the code, or a gate.

Reference material (conventions only; no gate compares to either):
- `~/spherex/LogNormalGalaxies` + `~/spherex/LogNormalSimulations` (Julia, Henry's):
  the construction this package was measured against (see Construction).
- `~/spherex/disco-mocks` (`discomocks/catalog.py`, `xcheck.py`): the streamed
  sampling / parquet / CIC-estimator patterns were copied from here (small, numpy,
  stable), not path-depended on.

Remote: `github.com/jrcheshire/logunusual`.

## Build / test / run

Env tool is **pixi**; platforms `osx-arm64` (laptop, JAX CPU fp64), `linux-64`
(deneb, RTX 3050 **6 GB** -- `nvidia-smi` reports 6144 MiB, 2026-09-15; earlier
notes here said 8 GB), `linux-aarch64` (TACC Vista).

```sh
pixi install                 # conda env; `default` and `gpu` share one solve-group
pixi run setup               # ONE-TIME (CPU): pinned jax/jaxlib 0.10.1 + editable install
pixi run -e gpu setup-gpu    # ONE-TIME (CUDA): jax[cuda12]; the -e gpu is REQUIRED, the
                             #   task refuses without it
pixi run test                # pytest -q; `pytest -m "not slow"` is the dev loop
pixi run check-format        # black --check; `pixi run format` applies
pixi run lint                # flake8, max-line 88, ignore E203
pixi run -e tables pk-tables check   # P1: regenerate the v28 linear tables byte for byte
pixi run -e tables pk-tables make --model halofit   # kh 1e-4..10 tables into data/
```

- The `tables` env (osx-arm64 only, no JAX, no default feature) pins conda-forge
  `camb 1.5.9` build `py312h904ef0c_0`, the build that made the v28 tables; P1 is
  byte-identical with it (2026-10-01). Run CAMB with `OMP_NUM_THREADS=1` on a shared
  laptop (26 s for the seven tables).

- **`JAX_ENABLE_X64=1` is mandatory** for the field stage (it is what makes float64
  the default; `dtype="f32"` is an explicit opt-in, not a consequence of dropping it). The pixi tasks set it; scripts
  and tests `setdefault` it and call `jax.config.update("jax_enable_x64", True)` before any
  array work. float32 is allowed for velocity grids and positions in flight; positions are
  written as float64 per the contract.
- On a CUDA machine every `pixi run` needs `-e gpu`; a bare `pixi run python` silently runs
  on CPU. GPU-less nodes of a CUDA-locked platform need `CONDA_OVERRIDE_CUDA=12.0` AND
  `JAX_PLATFORMS=cpu` (umbrella memory `reference_pixi_gotchas`).
- **`pixi.lock` co-commit invariant:** a dependency-spec change to `pyproject.toml` needs
  `pixi install` + the regenerated lock in the same commit; metadata/version-only edits
  leave it byte-identical (`pixi install --locked` is the check). `.github/workflows/ci.yml`
  exists, but GitHub **Actions is disabled on the repository**, so no run has ever fired.
- `data/`, `outputs/`, `runs/` and all `*.parq*`/`*.npz`/`*.png` are gitignored. Input
  P(k) TSVs and the mask h5 live on the LogNormalSimulations
  `jc/spherex-broad-bin-inputs` branch and in
  `~/spherex/myscripts/lognormal_mocks_diagnostics/data/`; copy into `data/` locally.
  The kh-10 linear and halofit tables (`matterpower_camb_{lin,halofit}_kmax10_zeff=*`)
  are made by `scripts/make_pk_tables.py`; sha256s in ROADMAP M4.

## Code layout (`logunusual/`)

- `suite.py` **[M0]** -- the default bin table `BIN_SUITE_V28` (frozen `Bin`s, now
  with a `pk_file` name and an optional `pk_galaxy_file`, M4), `seed_for`, the seed base and stride, `RADIAL_BUFFER`, mask
  constants, distance cosmology. Dependency-free.
- `grid.py` **[M1]** -- `Box`, k-grids (rfft on z), Hermitian weights, the separable
  `sinc` windows, the CIC shot-noise alias factor (Jing 2005).
- `pk.py` **[M1, M3]** -- TSV loader + `PowerSpectrum` (log-log cubic spline, power-law
  tails, `file_hash`), and the **grid-native** `grid_pkG`: `P -> xi -> log1p -> P_G`
  with two FFTs on the simulation grid. Reports `xi_min`, `sigma2`, the clipped
  negative-`P_G` fraction; raises if `xi <= -1`. `pk_on_grid` evaluates the spectrum
  once per integer radius `q = i^2 + j^2 + l^2` and gathers by `radius_index` (M3);
  ~1e-15 relative from evaluating on `k_grid`'s `|k|`, not bitwise.
  `check_table_pair` (M4): a galaxy table must match the linear one at their common
  lowest node to `k0^2 sigma_v^2` (the leading low-k nonlinear correction).
- `fnl.py` **[M4]** -- local f_NL, LSS convention: `LocalPNG` (f_nl, p, delta_c, and
  the tables' A_s / n_s / k_pivot / omega_m), `poisson_M` (`sqrt(P / P_Phi) / g0`
  from the bin's own table), `delta_b`, `galaxy_spectrum` (exactly `b * b * P_gal`
  at f_NL = 0; `galaxy_table=` is `P_gal`, M stays on the linear table), `growth_md`, `diagnostics` (`b(k_f)/b`, the k where b(k) changes sign).
- `field.py` **[M1, M2, M3, M4]** -- JAX (eager, x64): white noise (numpy PCG64) ->
  Gaussian -> lognormal galaxy and matter fields -> displacement components
  (`psi_axes`, any subset of "xyz"; `Fields.psi` dict, `psi_flat`). `generate_fields`
  is the entry point; `trace=` hook for memory instrumentation. Two M3 knobs, both
  defaulting to the M1 behaviour and neither reachable from a `RunConfig`:
  `dtype="f32"` halves the device arrays (`resolve_dtype`), and `jit=True` compiles
  `coloured_lognormal` / `displacement`. **`jit` is not bit-preserving**; `dtype`
  changes every catalog. See ROADMAP M3 for both. `fnl=` (M4) makes the galaxy target
  `b(k)^2 P`; with f_NL != 0 any clipped galaxy P_G mode raises. `galaxy_table=` (M4)
  replaces P in the galaxy target only; matter field and displacements are the
  linear run's bitwise, and any clipped galaxy P_G mode raises.
- `sample.py` **[M1, M3]** -- numpy: intensity, per-slab RNG streams (`slab_rng`,
  `draw_slab`: Poisson then uniform-in-cell placement on one x-slab's own stream),
  own-cell plane-parallel RSD, `split_seed`, `default_workers`.
- `shell.py` **[M2, M3]** -- `Shell` (rmin, rmax, buffer; `check_box`,
  `required_buffer`), observer-centred `cell_window` / `slab_window` (radial window
  and the angular pre-cut per slab), `AngularMask` (HEALPix h5, NESTED or RING, any
  NSIDE; `distance_to_set`), `select`, `rsd_radial`, `radial_histogram`, and the
  threaded `sample_shell` generator with `ShellStats`.
- `io.py` **[M2]** -- the catalog format: `CatalogWriter` (pinned row groups, bins
  ascending, atomic rename), `read_metadata`, `bin_metadata`, `check_layout`,
  `read_bin`.
- `config.py` **[M2]** -- `RunConfig` (YAML <-> dataclass; bins default to the suite;
  `nbar_scale`/`grid_scale` for smoke runs; `config_hash` over the mock definition,
  not paths; `f_nl` / `fnl_p` / `primordial` enter the hash only when f_NL != 0, and
  `png` is None at f_NL = 0; an empty `pk_galaxy_file` is neither serialised nor
  hashed, `_bin_dict`), `default_config`, `pk_galaxy_path`.
- `run.py` **[M2]** -- `generate_realization`: bins in order, field stage -> streamed
  shell draw -> writer; per-bin metadata and `summary.json`; `plan_realization`.
- `cli.py` **[M2]** -- `logunusual run | check | default-config` (console script).
- `validate.py` **[M1]** -- CIC painter, deconvolution, Jing shot noise, multipoles
  (Hermitian-weighted, `n_indep` for SEs), `gaussian_se`, Kaiser boosts, and the
  **coherent-alias estimator response** (`effective_window`, `estimator_response`).
- `gates.py` **[M1, M2, M4]** -- the statistical gates as functions (slow tests and
  `scripts/m{1,2}_gates.py`); per-realization ratios, scatter SEs, derived bands;
  `gate_shell_density` (M2); `gate_field_identity(..., fnl=)` (G13) and the
  matched-seed `gate_fnl_ratio` (G14, consistency only) (M4).
- `scripts/` -- `m1_gates.py`, `m1_reproducibility.py`, `m1_memory.py`,
  `m2_gates.py`, `m3_device.py`, `m4_fnl.py` (attainability sweep; `--grid-scale`,
  `--pk-template`, `--pk-galaxy-template`, matter rows), `make_pk_tables.py` (CAMB
  tables, `tables` env). `configs/v28_default.yaml` is the worked run config;
  `configs/v28_halofit.yaml` the halofit galaxy target on finer grids.

## Construction (M1, decided 2026-09-04 from measurements; first principles, not a
## port of the Julia code)

Grid `N^3`, box `L`, `dx = L/N`, `V_cell = dx^3`; cell centres at `(i + 0.5) dx`.

1. **Target grid spectrum** `P_grid(k) = P(|k|) / prod_i sinc(k_i dx / 2 pi)^2`, DC 0,
   with `P = b^2 P_in` (galaxies) or `P_in` (matter); `b` enters BEFORE the transform
   (lognormal bias). The `1/sinc^2` undoes the uniform-in-cell placement window
   (step 5) so the **catalog** has continuum power `P` in the first Brillouin zone.
2. **Grid-native P_G**: `xi = irfftn(P_grid)/V_cell`, `xi_G = log1p(xi)`,
   `P_G = rfftn(xi_G) V_cell`, DC 0, negatives clipped and REPORTED. For a periodic
   Gaussian field with grid spectrum P_G, `exp(G)/<exp G> - 1` has grid two-point
   function exactly `exp(xi_G) - 1 = xi` in the ensemble (Coles & Jones 1991), so its
   power is `P_grid` on EVERY mode up to the Nyquist. That identity is gate G3.
3. **Gaussian field** `G_k = rfftn(w) sqrt(P_G / V_cell)`, `w` unit white noise from
   `numpy.random.default_rng(ic_seed)`; the SAME `w` colours galaxy and matter fields.
   Estimator normalisation `V/N^6 |delta_k|^2` (disco-mocks convention).
4. **Lognormal** `1 + delta = exp(G) / mean_box(exp G)`, both fields; `1 + delta >= 0`
   by construction, no clipping anywhere downstream.
5. **Sampling** `lambda = nbar V_cell (1 + delta_g)`, per x-slab on the slab's own
   stream `Philox(key = draw_seed | slab << 64)`: `rng.poisson` over the slab's cells,
   then positions **uniform within the cell** (`jitter_p = 1`) galaxy-major. A catalog
   depends on `(draw_seed, field)` alone, so slabs run on a thread pool and the result
   is bitwise the same for any thread count. Seeds: `split_seed(seed) -> (ic, draw)`
   via `SeedSequence.spawn`.
6. **Velocities** `Psi_k = i k / k^2 delta_m,k` (linearised continuity on the MATTER
   lognormal field, Agrawal et al. 2017), DC 0, Mpc/h; each galaxy gets its OWN cell's
   `Psi`, times `f` at sampling; plane-parallel `s_z = z + f Psi_z` in M1, radial in
   M2.

7. **Local f_NL (M4)**: the galaxy target becomes `b(k)^2 P` with
   `b(k) = b + 2 (b - p) f_NL delta_c / M(k)`, LSS convention (`D(0) = 1`). M is not a
   transfer-function input: `P = M_CMB^2 P_Phi` with `P_Phi = (9/25) 2 pi^2 A_s k^-3
   (k/k_pivot)^(n_s - 1)` (Phi = 3/5 zeta in matter domination), so
   `M = sqrt(P / P_Phi) / g0`, `g0 = D_md(0)` from the flat-LCDM growth integral; the
   table's A_s / n_s / pivot must be the ones it was made with (G12). Matter field and
   velocities unchanged. **Where b(k) falls toward zero at low k (f_NL < 0) no
   lognormal reaches the target**: the higher orders of `log(1 + xi)` floor the
   realizable low-k power at these sigma^2, P_G clips on the lowest shells, and the
   clipped field's power there would be 1.4x-306x the target. An f_NL run therefore
   raises on any clipped galaxy mode (attainability table in ROADMAP M4).

8. **Nonlinear galaxy target (M4)**: a bin's `pk_galaxy_file` (Takahashi halofit,
   CAMB with the Bird et al. neutrino terms) replaces P in the galaxy target,
   `b(k)^2 P_gal`; `pk_file` stays linear and drives the matter field, the velocities
   (continuity is a linear relation) and M(k) (`P = M^2 P_Phi` is linear theory).
   Halofit pays only with finer cells: at the 2x grids' Nyquist it is 1.24x (bin 7)
   to 4.2x (bin 1) the linear power, and there a lognormal cannot reach it in bins
   1-3 (grid sigma^2 15-20), so `configs/v28_halofit.yaml` gives them the finest
   clip-free grids; a galaxy-table run raises on any clipped P_G mode.

**Why not the Julia construction.** Henry's 0.11.0 applies a `sinc^-p` deconvolution
to the lognormal field AFTER exponentiation. Measured here at bin-5 settings (128^3,
dx 7.8, b 1.76): that leaves `1 + delta < 0` in 33% of the cells and 27% of the
expected galaxies (`min(1+delta) = -45`); a sampler that zeroes those cells inflates
the density by ~27%, which matches prod_v2's 1.28x over-density (consistent with, not
proven). Deconvolving the TARGET instead is exact by the grid identity, but with the
triangular (p = 2) jitter it needs `P/sinc^4`, whose `xi` reaches -1.22 at the
neighbour lag: no lognormal exists. With uniform (p = 1) placement `P/sinc^2` is
attainable with zero clipped modes for v28 bins 1-6; bin 7 clips its 5 fundamental
modes, where the realized power is 1.109x the target (see ROADMAP). Own-cell velocity
assignment (not the 3x3x3 stencil) keeps the RSD prediction window-free.

**Estimator response (validate.py).** A catalog drawn from a grid field is
lattice-periodic, so on a finite estimator mesh the alias images carry the SAME
Fourier coefficient up to a sign and add coherently:
`P_mesh = |sum_n s_n W_cic(k_n) T(k_n)|^2 P_grid` (separable), with `s_n = (-1)^(r n)`
per axis from the half-cell offset of the cell centres (`r` = mesh ratio). The
standard incoherent alias sum applies to the shot noise only (Jing 2005). On a 2x
mesh every sign is + and the deconvolved power is biased low by 0.3% at half the
generator Nyquist and 3.5% at the Nyquist. On the generator's OWN mesh the dominant
image subtracts: the deconvolved power reads 6% low at half the Nyquist (shell
average; 11% along an axis), measured by G11 (2026-09-04; the sign was missing in M1's
formula, which the 2x gates could not see). `estimator_response` is the exact
correction and every catalog gate uses it.

**Shell product (M2).** Each bin is its own periodic box centred on the observer
(positions shifted by `-L/2`; cell centres at `(i + 0.5) dx - L/2`). The intensity is
multiplied at cell level by the full-sky window `r_lo <= |x_centre| <= r_hi` with
`r_lo = max(0, rmin - buffer)`, `r_hi = rmax + buffer`; the lognormal field itself is
periodic and unwindowed, so the grid identity (G3) is untouched. Each galaxy is
displaced radially by its own cell's displacement, `s = x + f (Psi . x / r^2) x`, then
kept iff `rmin <= |s| <= rmax` (inclusive) and, with a mask, the HEALPix pixel of `s`
is set (`hp.vec2pix`). No periodic wrap after the shift (a galaxy can only leave the box
from a buffer cell touching a face and is outside the shell either way; the count is
reported). The buffer must exceed a cell diagonal and `rmax + buffer <= L/2`
(`Shell.check_box`, config time) AND, for the realized field, `sqrt(3)/2 dx +
f max|Psi|` over the drawn cells (`Shell.required_buffer`; `sample_shell` raises
below it and records `psi_max` / `required_buffer` per bin). The production 150 Mpc/h
covers bin 2's ~113-128 (the lognormal displacement tail: `max|Psi|` 150-175 Mpc/h
at 256^3 against `psi_rms` 5); test fixtures with 20 Mpc/h were short and were
resized (2026-09-20). With a mask, the **angular pre-cut** (`slab_window`) drops the
cells whose galaxies cannot land in a set pixel: radial RSD keeps direction, so the
test is `AngularMask.distance_to_set` at the cell centre's pixel against
`cell_angular_radius(r_c) + 2 max_pixrad` -- an exact superset of the feeding cells
(gate in `tests/test_shell.py`), 0.73-0.75 of the radial window at fsky 0.713.
Consequences: `N_kept ~ Poisson(nbar fsky V_shell)` exactly for a uniform field (the
fast gate), and the radial profile is flat through both edges because galaxies cross
them in both directions (G10).

## Catalog format (the package's own spec; `io.py`)

- Path: `<output_dir>/<run_name>/realization_{i:05d}/catalog.parq` (+ `summary.json`).
- Columns, in this order: `x, y, z: float64` (comoving Mpc/h, observer at the origin,
  redshift space), `bin: int8` (the bin's `index`, always written). Nothing else: no
  velocities, no redshift, no randoms.
- Rows bin-contiguous, bins ascending; row groups of exactly `row_group_rows`
  (default 2^20) except each bin's last, so a row group never mixes bins; column
  statistics written; consecutive row groups byte-contiguous (one bin = one byte
  range). `io.check_layout` verifies all of this without reading the data; the CLI
  `check` runs it.
- Seeds: `seed_base + realization * 1000 + bin.index` -> `split_seed` -> (ic, draw).
- Density: each bin targets its configured `nbar` (times `nbar_scale`); there is no
  clip, so the file stamps `nbar_overdensity_factor = 1.0`.
- Metadata (file-level, string-valued): global keys (`generator`, `generator_version`,
  `created`, `host`, `machine`, `jax_backend`, `config_hash`, `seed_base`, scales,
  `radial_buffer`, `jitter_p`, `rsd`, `rng_scheme`, `n_workers`, `angular_precut`,
  mask name/sha256/nside/ordering/fsky or `mask = none`, `bins`) and per-bin keys
  `bin{index:02d}.*` (shell, box, grid, `b`, `f`, `nbar_target`, `realized_nbar` =
  `n_kept / (fsky V_shell)`, `n_galaxies`, `n_drawn`, `n_left_box`, `n_window_cells`
  and `_radial`, `lam_window`, `psi_max`, `required_buffer`, seeds, `pk_file`,
  `pk_sha256`, `pk_galaxy_file`, `pk_galaxy_sha256` (the linear table's when the bin
  names none), `sigma2_*`, `xi_min_galaxy`, `clipped_power_fraction`, `psi_rms`,
  timings, peak live JAX bytes). `f_nl` is always written; with f_NL != 0 also
  `fnl_convention` (LSS), `fnl_p`, `fnl_delta_c`, `fnl_A_s`, `fnl_n_s`, `fnl_k_pivot`,
  `fnl_omega_m`, `fnl_g0`, and per bin `fnl_delta_b_kf`, `fnl_b_kf_over_b`,
  `fnl_k_zero`.
- Written as `catalog.parq.tmp` and renamed on close: a file with the final name is
  complete.

## Conventions & gotchas

- Units: lengths Mpc/h, k in h/Mpc, P(k) in (Mpc/h)^3. Periodic-box code (M1) has
  cell centres at `(i + 0.5) * dx` and positions in `[0, L)`; the shell product shifts
  both by `-L/2` (observer at the origin). Shell edges are inclusive at both ends.
- Mask lookups use `hp.vec2pix(nside, x, y, z, nest=...)` on the redshift-space
  position; the ordering comes from the file's `ORDERING` attribute.
- `np.sinc(x) = sin(pi x)/(pi x)`; the windows take cycles per cell `f = k dx / 2 pi`.
- Standard errors: a shell's INDEPENDENT mode count is the rfft half-grid count
  (`n_indep`), not the Hermitian-weighted full-grid count (`nmodes`); using the latter
  understates the SE by sqrt 2 (caught by gate G2). Lognormal power at high k is
  dominated by rare peaks: its realization scatter is far above Gaussian and correlated
  across k, so SEs come from the scatter across realizations, never from a formula.
- Ratios are formed per realization, then averaged; each gate has its own seed range.
- Bands: kf-shells merged from low k until each holds `n_min_indep(n_real)` independent
  modes (so a Gaussian-scatter SE resolves 2% at 3 sigma); `gates.band_edges`.
- Growth rate: `Bin.f` are the prod_v2 literals, generated with astropy Planck18
  (Om0 = 0.30966, 0.06 eV neutrino), NOT the distance cosmology's Om0 = 0.3153; they
  differ by 0.4-0.9%. Pinned by `tests/test_suite.py`; a deliberate change is an M4
  decision, not a cleanup.
- The default input P(k) is **linear CAMB truncated at kh = 1.0**; the grid Nyquist
  (0.20-0.40 h/Mpc) binds first. Halofit only pays with a finer grid (M4); the 2x
  grids' corners reach k = 1.39, so they take the kh-10 tables. The kh-10 linear
  tables share the v28 nodes exactly but differ from them by up to 1.2e-5 in P (CAMB's
  `kmax`), more than the pair bound: pair a halofit table with the kh-10 linear one,
  never with the v28 one. The TSV spline is log-log cubic with power-law tails from the
  end-node secants.
- Linear-theory RSD is a k -> 0 limit. At production amplitude (bin 5, f Psi_rms =
  3.2 Mpc/h, sigma_g^2 = 3.9 on the grid) the galaxy-matter correlation is 0.99 at
  k = 0.035 and 0.965 at k = 0.19, and P2/P0 exceeds Kaiser by 25% at k = 0.19. The
  Kaiser gate therefore lives in a scaled-amplitude control arm (`P_in x 1e-2`) and the
  production-amplitude curves are MEASUREMENTS reported by the gate script.
- Reproducibility: numpy RNG + numpy sampler are deterministic and, with one Philox
  stream per x-slab, independent of the thread count BY CONSTRUCTION (a test asserts
  it); the JAX field stage is eager by default (`jit=True` is opt-in and moves
  37-88% of cells by 0.25-4 float64 eps of the field maximum). Bitwise across
  processes is asserted on
  Linux (CI) and characterised on macOS (umbrella memory: XLA CPU on macOS-arm64 can
  wobble in the last bit). Measured 2026-09-04 on the M4 laptop: identical bytes at
  32^3 and 64^3; 2026-09-20 `scripts/m1_reproducibility.py --n 64` IDENTICAL.
- **Philox `counter` is a position, not a stream id.** Streams that differ only in
  `counter` emit the same numbers shifted by a block; the slab index goes into the
  128-bit `key`. The counter version passed every fast test and failed G10's
  draw-count Poisson check at 9 sigma (2026-09-20). A distinctness test must check
  that two streams share NO values.
- Memory: JAX arrays are invisible to tracemalloc and `memory_stats()` is None on CPU;
  `scripts/m1_memory.py` polls `jax.live_arrays()` (misses XLA scratch, says so). On a
  device that HAS an allocator, the scratch it misses is 50-60% on top: measured
  2026-09-15/16 on deneb's RTX 3050 and a Vista GH200, the field stage's allocator peak
  is **12.0 x N^3 float64 at every production grid** (192^3 to 512^3; 12.03 GiB at
  512^3), against 7.5x live. **The allocator peak decides whether a grid fits; the live
  count does not.** `scripts/m3_device.py` reports both. On the GH200 f32 halves the
  allocator peak (6.02x) and jit takes 2.0x off f64 (10.02x); f32 + jit is 5.07x.
  The poller is also a sampling instrument with ~10% run-to-run spread, so it cannot
  settle a percent-level question on its own.
- Where the time went in the Julia reference (bin 5, 512^3, laptop): field 8 s, draw
  104 s single-threaded, constraint randoms 64 s, estimator 35 s; peak 75 GB. Here
  (2026-09-04, laptop CPU, full-density seven-bin realization): 187 s wall for all
  seven bins, 19.3 GB host RSS, 7.0 x N^3 float64 live JAX bytes per bin; bin 5 is
  8 s field + 31 s sample. The sample stage is 80% of the wall (M3's target), and the
  full-sky buffered window draws 2.4x the kept galaxies. The same realization on a
  Vista GH200 node (2026-09-17, 72 cores): 259 s, sample ~214 s, under one core busy
  on average -- the sample stage WAS serial numpy, and a GPU node lost to the laptop
  on it; the GPU field stage gains only ~1.2x at 512^3 (`ROADMAP.md` M3). Rebuilt
  2026-09-20: per-slab streams on a thread pool + angular pre-cut take bin 2's sample
  stage from 22.7 s to 2.1 s on the laptop's 16 cores (19.8 s on one thread; draws
  311M -> 230M). With that and the spline table (2026-09-30) the full seven-bin
  realization takes **82 s on the laptop** (was 187 s), peak RSS 13.8 GB; field 31 s,
  sample + write 50 s. The same realization on a Vista GH200 (2026-09-30, job 1039040)
  also takes 82 s, with identical counts per bin: field 28 s, sample + write 50 s on 72
  threads -- the sample stage does not scale past the laptop's 16 cores (unmeasured
  why), and the GPU pays a first-use cost per grid size (~20 s of the 82, estimated).

## Working rules (project)

- **Validation rigor over feature speed.** Every milestone's gates are written in
  `ROADMAP.md` before the code; gate thresholds are derived (self-convergence, Poisson
  SE, identity checks), not picked; **tolerances are not relaxed without explicit
  discussion + approval.** A gate that cannot fail is not a gate.
- **Author on shippable artifacts = "James Cheshire"** (never "Jamie").
- **git / PRs:** work on `jc/*` branches, one PR per session per repo, squash-merge onto
  `main`. Pushing, `gh repo create`, `gh pr create`, merging: **only at explicit
  direction**, never inferred from an adjacent action. Before a push, run the local gate
  (`pixi run test`, `check-format`, `lint`).
- **Do not calibrate mock inputs to SPHEREx data** (it puts the contamination into the
  null); the input ladder is linear -> halofit -> literature b(k)/HOD -> external survey.
- Keep SPHEREx-specific survey config (mask, forecast densities) out of any public
  surface; the v28 numbers in `suite.py` are forecast products, not data.
- Heavy local runs: price first (the laptop is shared); the Julia bin-5 baseline is 75 GB.

## Status & next step

- **M0 bootstrap DONE (2026-09-04).**
- **M1 periodic-box core DONE (2026-09-04, merged):** grid-native P_G, uniform
  placement with the target deconvolved before the transform, own-cell velocities,
  gates G1-G9 in `ROADMAP.md`. The Julia code was read in full and used for
  conventions only; no gate compares to it.
- **M2 shell product DONE (2026-09-04)** on branch `jc/m2-shell-product`: three
  displacement components, `shell.py`, `io.py`, `config.py`, `run.py`, `cli.py`,
  `configs/v28_default.yaml`; fast tests + slow gates G10/G11 (`ROADMAP.md` M2).
  Direction recorded 2026-09-04: this is a lognormal mock code, not a survey code; the
  v28 table and a mask are default inputs; no other repository is touched or used as
  a gate.
- **M3 performance and memory, measurement pass DONE (2026-09-15 to 09-17):** deneb
  RTX 3050 (6 GB, holds 3 of 7 bins; correctness box only), Vista GH200 (all bins fit,
  12.0 x N^3 f64 allocator peak, field stage ~1.2x the laptop, full realization 259 s
  vs 187 s), CPU-vs-CUDA ULPs ~1e-13 relative, not bitwise. Tables in `ROADMAP.md` M3.
- **M3 sample stage DONE (2026-09-20):** per-slab Philox streams (catalog independent
  of thread count by construction; every seed's catalog changed vs M2), threaded
  `sample_shell`, field-time buffer guard, angular pre-cut. Bin 2 sample stage 22.7 s
  -> 2.1 s on 16 cores. G3-G11 re-run under the new scheme; G4a's lowest band sat over
  the SE floor and was fixed 2026-09-21 by giving `gate_uniform_shot` the `band_seeds`
  split (64 seeds on the 32-seed bands, SE 0.69% -> 0.51%). **Re-deriving bands from a
  larger seed count never clears that floor -- the seed count cancels** (`ROADMAP.md`
  M3).
- **M3 field stage DONE (2026-09-21):** the P -> P_G step is split and it is
  **host-bound** -- 65% of the conversion is host numpy/scipy and 59% is the spectrum
  spline, which puts 45% of the whole field stage where no accelerator reaches it and
  explains the GH200's 1.2x. A real `dtype` knob replaced the f32 probe (live arrays
  exactly halve); `jit` is in and is not bitwise, and buys nothing on the laptop.
- **M3 allocator peaks DONE (2026-09-22, Vista job 1014508):** at 512^3, x N^3 f64:
  f64 eager 12.03, f64 jit 10.02, f32 eager 6.02 (exactly half), f32 jit 5.07. On the
  node the P -> P_G conversion is 98% host and the spline alone is 4.5 of the 6.27 s
  stage; jit's 512^3 wall is 0.93x eager. Fits on deneb are predictions from the
  GH200 table, not measured (`ROADMAP.md` M3).
- **M3 spline table ADOPTED (2026-09-30):** `pk_on_grid` gathers a per-radius table
  (max 9 eps from direct evaluation, gate derived in `tests/test_pk.py`); every
  catalog changes at the last bit. Laptop 512^3: each conversion 2.82 -> 1.20 s,
  field stage 8.31 -> 5.16 s blocked (8.09 -> 4.62 unblocked). GH200 (job 1039040):
  each conversion 2.57 -> 0.44 s, unblocked stage 6.27 -> 2.01 s, 2.3x the laptop.
- **Seven-bin re-measurement DONE (2026-09-30):** laptop 82 s vs 187 s, 651.4M
  galaxies, 15.04 GiB, layout ok; Vista GH200 82 s vs 259 s (job 1039040), identical
  counts per bin (`ROADMAP.md` M3).
- **M3 DONE (2026-10-01).** Open, not M3 deliverables: the sample stage's flat scaling
  from 16 to 72 cores and the GPU's per-grid-size first-use cost, both observed only.
- **M4 local f_NL DONE (2026-10-01)** on branch `jc/m4-fnl-bias`: `fnl.py`, LSS
  convention, |f_NL| <= 100 measured. G12 (M normalisation, derived bound, mutations
  fail), G13 (grid identity with b(k), f_NL = +-100, floor met), G14 (matched-seed
  catalog ratio, consistency gate). f_NL < 0 is unattainable on the lowest shells of
  bins 2-7 and raises; positive f_NL runs on every bin.
- **M4 halofit on finer grids, in progress (2026-10-01)**, branch `jc/m4-halofit`:
  table maker + P1 (byte-identical), two-table plumbing, pair check, attainability at
  1x / 2x (halofit at 2x clips in bins 1-3 -> grids 270 / 450 / 720 there, 2x in 4-7;
  a galaxy-table run raises on any clip). Next: one realization per bin on those
  grids, then G15/G16 (`ROADMAP.md` M4).
- Open, not blocking: the 1.28x closure arm (the post-transform deconvolution's
  clipped mass; see Construction) and whether to report it -- JC's call.
