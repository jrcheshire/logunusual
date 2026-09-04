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

Remote: none yet (local repo). Intended home `github.com/jrcheshire/logunusual`.

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

Planned modules with status tags; only `suite.py` exists at M0.

- `suite.py` **[M0, done]** -- `BIN_SUITE_V28` (7 frozen `Bin`s), `seed_for`, the
  seed base and stride, `RADIAL_BUFFER`, mask constants, distance cosmology. **The one
  source of the bin table**; chimera/myscripts carry three drifting copies until M2.
- `pk.py` [M1] -- TSV loader (`# kh\tPk`, 10001 rows, kh 1e-4..1.0), power-law
  extrapolation, P(k) -> xi(r) -> xi_G = log1p(xi) -> P_G(k) (FFTLog), plus the 3D-FFT
  route Henry uses for gridded inputs (`scale_by_pk!` array method).
- `field.py` [M1] -- coloured Gaussian field, lognormal transform, voxel-window
  correction, displacement field (JAX).
- `sample.py` [M1] -- per-cell intensity, Poisson counts, jittered placement, per-cell
  velocity stencil, RSD (plane-parallel in M1, radial in M2).
- `shell.py` [M2] -- buffered shell + HEALPix mask window at cell level, observer at
  origin.
- `io.py` [M2] -- streamed parquet writer with pinned row groups + provenance metadata.
- `validate.py` [M1] -- periodic CIC P(k) with shot subtraction (from disco-mocks
  `xcheck.py`), Poisson self-check.
- `cli.py` [M2] -- `logunusual run --config ... --realizations a:b`.

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
  `(i + 0.5) * dx`.
- Field colouring: `delta_k = rfftn(w) * sqrt(P_G(k) / V_cell)` with `w` unit white
  noise (numpy `default_rng(seed).standard_normal`), DC mode zeroed. Same convention as
  disco-mocks/mbody; Henry's `draw_phases` + `multiply_by_pkG!` is the same thing with
  the `1/sqrt(N^3)` and `sqrt(P_G V)` factors split differently -- match in P(k), not
  in intermediate normalisation.
- Lognormal transform: `delta = exp(G) / <exp(G)> - 1` with the mean taken over the
  box (Henry's `generate_field`), for BOTH the biased galaxy field (P_G from
  `b^2 P`) and the unbiased matter field (P_G from `P`).
- Velocities: displacement of the **matter** lognormal field, `v_k = i k / k^2 delta_m,k`
  after the voxel-window correction, times `f` at sampling
  (`velocities_aH .*= f` in `Run_Sims.jl`). Each galaxy gets its **cell's** velocity:
  Julia `velocity_assignment = 6` averages a 3x3x3 neighbourhood with overlap weights
  evaluated at the cell centre, so it is a fixed stencil on the grid, not per-galaxy work.
- Positions: cell centre + `(u1 + u2 - 1) * dx` per axis (Julia `voxel_window_power = 2`,
  a triangular kernel). Counts: Poisson per cell with mean
  `nbar * V_cell * (1 + delta_g) * W`, `W` the shell window.
- Voxel-window correction: `delta_g,k *= sinc(k dx / 2pi)^-p`, p = 2, applied to the
  galaxy field before sampling and to the matter field before the velocity divide.
  LogNormalGalaxies >= 0.10 turns this on by default; 0.9.4 (the Manifest pin in the
  local LogNormalSimulations checkout) has it off. The 0.9.4 vs 0.11.0 count ratio
  (1.263 and 1.281 on bins 1 and 5) matches prod_v2's 1.28x over-density; consistent
  with the flip being the cause, not proven (umbrella memory
  `project_lognormal_generator_stage_split`).
- RSD: radial, `x += (v . x / r^2) x` on the observer-centred positions
  (`catalog_generation.jl apply_rsd!`), applied before periodic wrap and before the mask.
- Growth rate: `Bin.f` are the prod_v2 literals, generated with astropy Planck18
  (Om0 = 0.30966, 0.06 eV neutrino), NOT the distance cosmology's Om0 = 0.3153; they
  differ by 0.4-0.9%. Pinned by `tests/test_suite.py`; a deliberate change is an M4
  decision, not a cleanup.
- Input P(k) is **linear CAMB truncated at kh = 1.0**; the grid Nyquist (0.20-0.40 h/Mpc)
  binds first. Halofit only pays with a finer grid (M4).
- P_G inversion can go negative at high k and is stressed by an f_NL 1/k^2 low-k boost;
  Henry clips `P_G >= 0` and truncates the rising tail (`pk_to_pkG.jl`). Reproduce the
  behaviour, then measure whether it matters (M1 gate), do not silently improve it.
- Where the time goes in the Julia reference (bin 5, 512^3, laptop): field 8 s, draw
  104 s single-threaded, constraint randoms 64 s, estimator 35 s; peak 75 GB of which
  ~50 GB is randoms + estimator. The design targets the draw and drops the other two.

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

- **M0 bootstrap DONE (2026-09-04):** skeleton, pixi env (default + gpu), `suite.py` +
  7 tests green, `docs/landscape.md`, `ROADMAP.md`.
- **Next: M1 periodic-box core.** Open a fresh plan-mode session against `ROADMAP.md`
  M1; first design questions there are the FFTLog implementation choice (mcfit vs an
  in-repo FFTLog) and dumping Henry's TwoFAST P_G(k) for the bin-5 TSV as the parity
  curve.
- Open, not blocking: the 1.28x closure arm (LogNormalGalaxies 0.11.0 with the voxel
  correction off) and whether to report it to Henry -- JC's call.
