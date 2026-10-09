# CLAUDE.md

Agent-facing map of this repo: what the package is, how to build and test it, where
things live, and the conventions that are easy to get wrong. `README.md` is the user
how-to; the method, file format, checks and costs are in `docs/`.

## What this is

`logunusual` is a **lognormal galaxy mock generator: JAX for the field stage (CPU or
CUDA), numpy for per-cell Poisson sampling and placement, streamed parquet output.**
A run is a list of bins, each a periodic box centred on the observer holding one
radial shell; galaxies are drawn in the buffered shell, displaced radially by their
cell's velocity, cut to the shell and an optional angular HEALPix mask, and streamed
to one parquet per realization. Inputs are one P(k) TSV per bin (optionally a second,
nonlinear table for the galaxy target only); local f_NL enters as a scale-dependent
galaxy bias built from the bin's linear table.

The **default inputs** are a survey forecast: the seven-bin v28 table in `suite.py`
(shells partitioning z = 0-2.2, densities, biases, growth rates) and, when given, a
survey mask. Nothing else in the code knows about a survey; any bin table with the
same fields can be given in a run config. The package has no dependency on any other
repository.

Docs:
- `docs/construction.md` -- the algorithm, its equations and why each step is built
  the way it is; attainability limits.
- `docs/catalog.md` -- the catalog file format and metadata keys.
- `docs/validation.md` -- every statistical check: what it tests, its bar, how to run.
- `docs/performance.md` -- cost and memory per stage, CPU vs GPU, sizing.
- `docs/references.md` -- literature.

## Build / test / run

Env tool is **pixi**; platforms `osx-arm64` (JAX CPU), `linux-64` (CPU or CUDA),
`linux-aarch64` (CPU or CUDA, e.g. GH200 nodes).

```sh
pixi install                 # conda env; `default` and `gpu` share one solve-group
pixi run setup               # ONE-TIME (CPU): pinned jax/jaxlib 0.10.1 + editable install
pixi run -e gpu setup-gpu    # ONE-TIME (CUDA): jax[cuda12]; the -e gpu is REQUIRED, the
                             #   task refuses without it
pixi run test                # pytest -q, slow statistical checks included (~9 min)
pixi run python -m pytest -m "not slow"   # the fast dev loop
pixi run check-format        # black --check; `pixi run format` applies
pixi run lint                # flake8, max-line 88, ignore E203
pixi run -e tables pk-tables make-default   # the default bin table's tables into data/
pixi run -e tables pk-tables check   # regenerate the v28 linear tables byte for byte
pixi run -e tables pk-tables make --model halofit   # kh 1e-4..10 tables into data/
```

- Before a push, run all three of `pixi run test`, `check-format`, `lint`. The CI
  workflow in `.github/workflows/ci.yml` runs the same three.
- The `tables` env (osx-arm64 only, no JAX, no default feature) pins conda-forge
  `camb 1.5.9` build `py312h904ef0c_0`, the build that made the v28 tables;
  `pk-tables check` is byte-identical with it. Run CAMB with `OMP_NUM_THREADS=1` on a
  shared machine (~26 s for the seven tables).
- **`JAX_ENABLE_X64=1` is mandatory** for the field stage (it makes float64 the
  default; `dtype="f32"` is an explicit opt-in). The pixi tasks set it; scripts and
  tests `setdefault` it and call `jax.config.update("jax_enable_x64", True)` before any
  array work. Positions are written as float64.
- On a CUDA machine every `pixi run` needs `-e gpu`; a bare `pixi run python` silently
  runs on CPU. GPU-less nodes of a CUDA-locked platform need `CONDA_OVERRIDE_CUDA=12.0`
  AND `JAX_PLATFORMS=cpu`.
- **`pixi.lock` co-commit invariant:** a dependency-spec change to `pyproject.toml`
  needs `pixi install` and the regenerated lock in the same commit; metadata-only edits
  leave it byte-identical (`pixi install --locked` is the check).
- `data/`, `outputs/`, `runs/` and all `*.parq*`/`*.npz`/`*.png` are gitignored. Input
  tables are made into `data/` by `pk-tables make-default` / `make`; a mask is supplied
  by the user (`data/mask.h5` in the example configs). `tests/data/` holds the one
  table the tests use.

## Code layout

- `logunusual/suite.py` -- the default bin table `BIN_SUITE_V28` (frozen `Bin`s with
  `pk_file` and optional `pk_galaxy_file`; `f` from `fnl.growth_rate_md`), `seed_for`,
  `RADIAL_BUFFER`, mask constants, distance cosmology. Dependency-free.
- `grid.py` -- `Box`, k-grids (rfft on z), Hermitian weights, separable `sinc`
  windows, the CIC shot-noise alias factor.
- `pk.py` -- TSV loader + `PowerSpectrum` (log-log cubic spline, power-law tails,
  `file_hash`); the grid-native `grid_pkG` (`P -> xi -> log1p -> P_G`, clipping
  reported); `pk_on_grid` (per-integer-radius table, ~1e-15 relative from direct
  evaluation, not bitwise); `check_table_pair`.
- `fnl.py` -- local f_NL, LSS convention: `LocalPNG`, `poisson_M`, `delta_b`,
  `galaxy_spectrum`, `growth_md` / `growth_rate_md`, `diagnostics`.
- `field.py` -- JAX field stage: `generate_fields` (white noise -> Gaussian ->
  lognormal galaxy and matter fields -> displacement components). Knobs not reachable
  from a `RunConfig`: `dtype="f32"` and `jit=True` (**not bit-preserving**); `fnl=`
  and `galaxy_table=` change the galaxy target only and raise on any clipped galaxy
  P_G mode.
- `sample.py` -- numpy sampling: per-slab Philox streams (`slab_rng`, `draw_slab`),
  uniform-in-cell placement, plane-parallel RSD, `split_seed`, `default_workers`.
- `shell.py` -- `Shell` (`check_box`, `required_buffer`), observer-centred
  `cell_window` / `slab_window` (radial window and angular pre-cut), `AngularMask`,
  `select`, `rsd_radial`, `radial_histogram`, the threaded `sample_shell`.
- `io.py` -- the catalog format: `CatalogWriter`, `read_metadata`, `bin_metadata`,
  `check_layout`, `read_bin`.
- `config.py` -- `RunConfig` (YAML <-> dataclass; `config_hash` over the mock
  definition, not paths; f_NL fields enter the hash only when f_NL != 0; an empty
  `pk_galaxy_file` is neither serialised nor hashed), `default_config`.
- `run.py` -- `generate_realization` (bins in order: field stage -> streamed shell
  draw -> writer; per-bin metadata and `summary.json`), `plan_realization`.
- `cli.py` -- `logunusual run | check | default-config`.
- `validate.py` -- CIC painter, deconvolution, Jing shot noise, multipoles, Gaussian
  SEs, Kaiser boosts, and the coherent-alias `estimator_response`.
- `gates.py` -- the statistical checks as functions returning dicts, shared by the
  slow tests and `scripts/gates_{box,shell}.py`.
- `scripts/` -- `gates_box.py`, `gates_shell.py`, `reproducibility.py`, `device.py`,
  `attainability.py`, `ensemble_check.py`, `ensemble.sbatch`, `make_pk_tables.py`
  (see `README.md`).
- `configs/v28_default.yaml` (worked example), `configs/v28_halofit.yaml` (halofit
  galaxy target, bins 1-6).

## Conventions & gotchas

- Units: lengths Mpc/h, k in h/Mpc, P(k) in (Mpc/h)^3. Periodic-box code has cell
  centres at `(i + 0.5) * dx` and positions in `[0, L)`; the shell product shifts both
  by `-L/2` (observer at the origin). Shell edges are inclusive at both ends.
- `np.sinc(x) = sin(pi x)/(pi x)`; the windows take cycles per cell `f = k dx / 2 pi`.
- Mask lookups use `hp.vec2pix(nside, x, y, z, nest=...)` on the redshift-space
  position; the ordering comes from the file's `ORDERING` attribute.
- **The target is deconvolved before the lognormal transform** (`P / sinc^2`, uniform
  placement). Deconvolving the field after exponentiation, or using p = 2 jitter,
  breaks attainability (`docs/construction.md`).
- Mode counts: a shell holds `nmodes / 2` independent modes (`nmodes` the
  Hermitian-weighted full-grid count). Not the half-grid cell count: the kz = 0 and
  Nyquist planes hold both members of each conjugate pair, and a self-conjugate mode
  is real (twice the variance). `validate.gaussian_se` and `gates.band_edges` use it.
  Lognormal power at high k is dominated by rare peaks: realization scatter is far
  above Gaussian and correlated across k, so SEs come from the scatter across
  realizations, never from a formula.
- Ratios are formed per realization, then averaged; each check has its own seed range.
  Bands merge kf-shells from low k until each holds `n_min_indep(n_real)` independent
  modes (`gates.band_edges`).
- On the generator's own mesh the dominant alias image SUBTRACTS (sign `(-1)^(r n)`);
  every catalog check corrects with `validate.estimator_response`.
- **Philox `counter` is a position, not a stream id.** Streams that differ only in
  `counter` emit the same numbers shifted by a block; the slab index goes into the
  128-bit `key`. A distinctness test must check that two streams share NO values.
- Reproducibility: catalogs are bitwise independent of the thread count by
  construction (a test asserts it). The JAX field stage is eager by default; across
  processes it is bitwise on Linux and can differ in the last bit on macOS-arm64.
- Pair a halofit table with the **kh-10 linear** table, never the v28 one: the two
  linear sets share nodes but differ by up to 1.2e-5 in P, more than the pair bound.
- Linear-theory RSD is a k -> 0 limit; at production amplitude the Kaiser ratio is a
  measurement, and the Kaiser check runs on a scaled-amplitude arm (`P_in x 1e-2`).
- Memory: JAX arrays are invisible to tracemalloc and `memory_stats()` is None on CPU.
  **The device allocator peak (12.0 x N^3 float64 for the field stage) decides whether
  a grid fits, not the live-array count**; `scripts/device.py ladder` reports both.

## Working rules

- **Validation rigor over feature speed.** New behaviour gets its checks written down
  (in `docs/validation.md`) before the code; bars are derived (self-convergence,
  Poisson SE, identity checks), not picked; **tolerances are not relaxed without an
  explicit, recorded decision.** A check that cannot fail is not a check.
- **git:** feature branches, squash-merge onto `main`. Push, open or merge a PR only on
  explicit instruction; run the local gate first.
- **Do not calibrate mock inputs to SPHEREx data** (it puts the contamination into the
  null); the input ladder is linear -> halofit -> literature b(k)/HOD -> external
  survey.
- Heavy local runs: price them first (a full seven-bin realization peaks at ~14 GB RSS;
  the field stage's device peak is 12 x N^3 float64).
