# CLAUDE.md

Guidance for working in this repo. User-level preferences (`~/.claude/CLAUDE.md`) apply on
top of this and win on conflict. `README.md` is the collaborator-facing how-to; this file
is the agent-facing map: what the package is, the product contract it must honour, the
conventions that are easy to get wrong, and where the project stands.

## What this is

`logunusual` is a **fast lognormal galaxy mock generator: JAX for the field stage (CPU or
CUDA), numpy for per-cell Poisson sampling and placement, streamed parquet output.** Its
first job is to be a drop-in producer of the SPHEREx per-bin lognormal mock suite
(`prod_v2`: 7 shells partitioning z = 0-2.2, observer at the origin, RSD applied, 100
realizations) at a fraction of the wall time and memory of the Julia path, with an input
P(k) interface that can later take a nonlinear spectrum and f_NL scale-dependent bias.

Siblings and their roles:
- `~/spherex/LogNormalGalaxies` + `~/spherex/LogNormalSimulations` (Julia, Wide-Angle-Team,
  Henry's): the **reference implementation** and parity target. Read the code, not the
  docstrings: `LogNormalGalaxies.jl` (`generate_field`, `draw_galaxies_with_velocities`,
  `pixel_window!`), `pk_to_pkG.jl`, `LogNormalSimulations/src/Run_Sims.jl` (per-option
  stage order), `catalog_generation.jl` (`apply_rsd!`, `select_galaxies`),
  `winlib.jl` (`calc_win`).
- `~/spherex/disco-mocks` (`discomocks/catalog.py`, `icfield.py`, `xcheck.py`): the
  **sampling / parquet / validation layer is copied from here** (small, numpy, stable),
  not path-depended on -- a path dep would drag DISCO-DJ and its env.
- `~/spherex/SPHEREx-L4-Cosmology-Pipeline` (`chimera`): the **consumers**.
  `systematics/lognormal_mocks/{run_bin_suite.py, generate_config.py, read_outputs.py,
  contaminate_catalog.py}` and `systematics/null_test/run_field_level.py` define the
  contract below. `myscripts/lognormal_mocks_diagnostics/pk_bin_patches.py` and
  `diagnose_realization.py` are the shell-level validation instruments.

Remote: `github.com/jrcheshire/logunusual` (pushed 2026-09-04).

## Build / test / run

Env tool is **pixi**; platforms `osx-arm64` (laptop, JAX CPU fp64), `linux-64`
(deneb, RTX 3050 8 GB), `linux-aarch64` (TACC Vista).

```sh
pixi install                 # conda env; `default` and `gpu` share one solve-group
pixi run setup               # ONE-TIME (CPU): pinned jax/jaxlib 0.10.1 + editable install
pixi run -e gpu setup-gpu    # ONE-TIME (CUDA): jax[cuda12]; the -e gpu is REQUIRED, the
                             #   task refuses without it
pixi run test                # pytest -q; `pytest -m "not slow"` is the dev loop
pixi run check-format        # black --check; `pixi run format` applies
pixi run lint                # flake8, max-line 88, ignore E203
```

- **`JAX_ENABLE_X64=1` is mandatory** for the field stage. The pixi tasks set it; scripts
  and tests `setdefault` it and call `jax.config.update("jax_enable_x64", True)` before any
  array work. float32 is allowed for velocity grids and positions in flight; positions are
  written as float64 per the contract.
- On a CUDA machine every `pixi run` needs `-e gpu`; a bare `pixi run python` silently runs
  on CPU. GPU-less nodes of a CUDA-locked platform need `CONDA_OVERRIDE_CUDA=12.0` AND
  `JAX_PLATFORMS=cpu` (umbrella memory `reference_pixi_gotchas`).
- **`pixi.lock` co-commit invariant:** a dependency-spec change to `pyproject.toml` needs
  `pixi install` + the regenerated lock in the same commit; metadata/version-only edits
  leave it byte-identical (`pixi install --locked` is the check). No CI yet (added in M1).
- `data/`, `outputs/`, `runs/` and all `*.parq*`/`*.npz`/`*.png` are gitignored. Input
  P(k) TSVs and the mask h5 live on the LogNormalSimulations
  `jc/spherex-broad-bin-inputs` branch and in
  `~/spherex/myscripts/lognormal_mocks_diagnostics/data/`; copy into `data/` locally.

## Code layout (`logunusual/`)

- `suite.py` **[M0]** -- `BIN_SUITE_V28` (7 frozen `Bin`s), `seed_for`, the seed base
  and stride, `RADIAL_BUFFER`, mask constants, distance cosmology. **The one source of
  the bin table**; chimera/myscripts carry three drifting copies until M2.
- `grid.py` **[M1]** -- `Box`, k-grids (rfft on z, the plane-parallel line of sight),
  Hermitian weights, the separable `sinc` windows, the CIC shot-noise alias factor
  (Jing 2005).
- `pk.py` **[M1]** -- TSV loader + `PowerSpectrum` (log-log cubic spline, power-law
  tails), and the **grid-native** `grid_pkG`: `P -> xi -> log1p -> P_G` done with two
  FFTs on the simulation grid (see Construction). Reports `xi_min`, `sigma2`, and the
  clipped negative-`P_G` fraction; raises if `xi <= -1` (no lognormal exists).
- `field.py` **[M1]** -- JAX (eager, x64): white noise (numpy PCG64) -> Gaussian ->
  lognormal galaxy and matter fields -> z displacement. `generate_fields` is the entry
  point; `trace=` hook for memory instrumentation.
- `sample.py` **[M1]** -- numpy: intensity, Poisson counts, streamed uniform-in-cell
  placement (chunking is bit-invariant), own-cell plane-parallel RSD, `split_seed`.
- `validate.py` **[M1]** -- CIC painter, deconvolution, Jing shot noise, multipoles
  (Hermitian-weighted, `n_indep` for SEs), `gaussian_se`, Kaiser boosts, and the
  **coherent-alias estimator response** (`effective_window`, `estimator_response`).
- `gates.py` **[M1]** -- the statistical gates as functions (used by the slow tests
  and `scripts/m1_gates.py`); per-realization ratios, scatter SEs, derived bands.
- `shell.py` [M2] -- buffered shell + HEALPix mask window at cell level, observer at
  origin, radial RSD (needs all three displacement components).
- `io.py` [M2] -- streamed parquet writer with pinned row groups + provenance metadata.
- `cli.py` [M2] -- `logunusual run --config ... --realizations a:b`.
- `scripts/` -- `m1_gates.py` (full gate tables -> `runs/m1_gates/*.json`),
  `m1_reproducibility.py` (two processes, byte compare), `m1_memory.py` (live JAX
  bytes, polled).

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
5. **Sampling** `lambda = nbar V_cell (1 + delta_g)`, `rng.poisson`, positions
   **uniform within the cell** (`jitter_p = 1`), streamed over slabs in galaxy-major
   draw order (any chunking is bit-identical). Seeds: `split_seed(seed) -> (ic, draw)`
   via `SeedSequence.spawn`.
6. **Velocities** `Psi_k = i k / k^2 delta_m,k` (linearised continuity on the MATTER
   lognormal field, Agrawal et al. 2017), DC 0, Mpc/h; each galaxy gets its OWN cell's
   `Psi`, times `f` at sampling; plane-parallel `s_z = z + f Psi_z` in M1, radial in
   M2.

**Why not the Julia construction.** Henry's 0.11.0 applies a `sinc^-p` deconvolution
to the lognormal field AFTER exponentiation. Measured here at bin-5 settings (128^3,
dx 7.8, b 1.76): that leaves `1 + delta < 0` in 33% of the cells and 27% of the
expected galaxies (`min(1+delta) = -45`); a sampler that zeroes those cells inflates
the density by ~27%, which matches prod_v2's 1.28x over-density (consistent with, not
proven). Deconvolving the TARGET instead is exact by the grid identity, but with the
triangular (p = 2) jitter it needs `P/sinc^4`, whose `xi` reaches -1.22 at the
neighbour lag: no lognormal exists. With uniform (p = 1) placement `P/sinc^2` is
attainable with zero clipped modes for every v28 bin (see ROADMAP). Own-cell velocity
assignment (not the 3x3x3 stencil) keeps the RSD prediction window-free.

**Estimator response (validate.py).** A catalog drawn from a grid field is
lattice-periodic, so on a finite estimator mesh the alias images carry the SAME
Fourier coefficient and add coherently: `P_mesh = |sum_n W_cic(k_n) T(k_n)|^2 P_grid`
(separable). The standard incoherent alias sum applies to the shot noise only (Jing
2005). On a 2x mesh the deconvolved power is biased low by 0.3% at half the generator
Nyquist and 3.5% at the Nyquist; `estimator_response` is the exact correction and
every catalog gate uses it. A consumer measuring these mocks on a mesh comparable to
the generator's sees the same effect.

## Product contract (M2 target; the layout is the interface)

Copied from the chimera readers, 2026-09-04. Cross-check against
`run_bin_suite.py:172-188` and `pk_bin_patches.py:114-186` before changing anything.

- Path: `<out>/<run_name>/realization_{i:05d}/catalog.parq`.
- Columns: `x, y, z: double` (comoving Mpc/h, distance cosmology H0 = 67.36,
  Om0 = 0.3153), `bin: int8` = 1..7 (`Bin.index`). Nothing else: no velocities, no
  redshift, no randoms.
- Frame: **observer at the origin**; each bin is its own box but all share the origin,
  so the file is one nested-shell cloud. **RSD already applied** (radial).
- Ordering: rows **bin-contiguous, bins ascending**; row groups **2^20 rows**,
  byte-contiguous per bin (`pk_bin_patches.py` hard-exits otherwise; pyarrow's default
  gives 2^20 implicitly today -- pin it explicitly). The same reader selects row groups
  by the `bin` column's min/max **statistics**, so statistics must be written, and it
  takes the slab start from column 0, so `x` stays the first column.
- Seeds: `SEED_BASE + realization * 1000 + bin_index`, base 137_000_000.
- Mask: HEALPix NSIDE 128 NESTED int8 dataset `MASK` (h5), fsky 0.7127; galaxies are
  kept where the mask is 1 and `rmin <= r <= rmax`. Shells are constant-phi_r
  (`radial_selection form=1`); the drawn region is the shell padded by 150 Mpc/h.
- Density: the **nominal v28 nbar** (`Bin.nbar`). prod_v2 is 1.24-1.28x over-dense and
  `contaminate_catalog.py --nbar-overdensity-factor` defaults to 1.28 because of it;
  M2 makes that metadata-driven. Do not reproduce the over-density.
- Metadata (file-level, string-valued, disco-mocks style): `box_size_x/y/z` per bin,
  `generator = logunusual`, package version, host, config hash, input P(k) file hash per
  bin, `nbar_target` and `realized_nbar` per bin, `ic_seed`/`draw_seed` per bin.

## Conventions & gotchas

- Units: lengths Mpc/h, k in h/Mpc, P(k) in (Mpc/h)^3. Cell centres at
  `(i + 0.5) * dx`; positions in `[0, L)`.
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
- Input P(k) is **linear CAMB truncated at kh = 1.0**; the grid Nyquist (0.20-0.40 h/Mpc)
  binds first. Halofit only pays with a finer grid (M4). The TSV spline is log-log
  cubic with power-law tails from the end-node secants.
- Linear-theory RSD is a k -> 0 limit. At production amplitude (bin 5, f Psi_rms =
  3.2 Mpc/h, sigma_g^2 = 3.9 on the grid) the galaxy-matter correlation is 0.99 at
  k = 0.035 and 0.965 at k = 0.19, and P2/P0 exceeds Kaiser by 25% at k = 0.19. The
  Kaiser gate therefore lives in a scaled-amplitude control arm (`P_in x 1e-2`) and the
  production-amplitude curves are MEASUREMENTS reported by the gate script.
- Reproducibility: numpy RNG + numpy sampler are deterministic; the JAX field stage is
  eager (no jit). Bitwise across processes is asserted on Linux (CI) and characterised
  on macOS (umbrella memory: XLA CPU on macOS-arm64 can wobble in the last bit).
  Measured 2026-09-04 on the M4 laptop: identical bytes at 32^3 and 64^3.
- Memory: JAX arrays are invisible to tracemalloc and `memory_stats()` is None on CPU;
  `scripts/m1_memory.py` polls `jax.live_arrays()` (misses XLA scratch, says so).
- Where the time went in the Julia reference (bin 5, 512^3, laptop): field 8 s, draw
  104 s single-threaded, constraint randoms 64 s, estimator 35 s; peak 75 GB. Here at
  128^3: fields 0.7 s, 3e6-galaxy catalog 0.13 s (M3 measures 512^3).

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
- **M1 periodic-box core DONE (2026-09-04)** on branch `jc/m1-periodic-box`: modules
  above, 44 fast tests + 5 slow gates, CI workflow, three scripts. Gate results and the
  two gate re-derivations (estimator response; Kaiser as a linear-limit arm) are in
  `ROADMAP.md` M1. The Julia code was read in full and used for conventions only; no
  gate compares to it (JC's direction, 2026-09-04).
- **Next: M2 shell product** (shell + mask window, radial RSD with all three
  displacement components, streamed parquet, 7-bin driver, cli). Open a fresh plan-mode
  session against `ROADMAP.md` M2.
- Open, not blocking: the 1.28x closure arm (now with a measured mechanism: the
  post-transform deconvolution's clipped mass; see Construction) and whether to report
  it to Henry -- JC's call.
