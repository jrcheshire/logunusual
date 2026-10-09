# logunusual

Lognormal galaxy mock generator. A run is a list of redshift bins; each bin is a
periodic box centred on the observer that holds one radial shell. Per bin, a JAX field
stage (CPU or CUDA) builds a lognormal galaxy field and a lognormal matter field from
one white noise; a threaded numpy sampler draws Poisson galaxies cell by cell, one RNG
stream per grid slab (so the catalog does not depend on the thread count), places them
uniformly within their cells, shifts them radially by their cell's linear-continuity
displacement, and keeps those inside the shell and an optional HEALPix mask. Each
realization is streamed to one parquet file. Options: a local-f_NL scale-dependent
galaxy bias, and a nonlinear (e.g. halofit) galaxy target. The default bin table is a
seven-shell survey forecast (`logunusual/suite.py`); any table with the same fields
can be given in the run config.

## Install

[pixi](https://pixi.sh) manages the environments.

```sh
pixi install                 # conda env
pixi run setup               # once: pinned CPU JAX + the `logunusual` command
pixi run test                # full test suite incl. slow statistical checks (minutes)
```

On a CUDA 12 machine (linux-64 or linux-aarch64), install CUDA JAX into the separate
`gpu` env and pass `-e gpu` to **every** command; a plain `pixi run` uses the CPU env:

```sh
pixi run -e gpu setup-gpu
pixi run -e gpu logunusual run ...
```

The `tables` env (osx-arm64 only) holds the pinned CAMB build that makes the input
P(k) tables; it has no JAX and does not install the package.

## Inputs

**P(k) tables.** One TSV per bin in the config's `pk_dir`: a `# kh\tPk` header line,
then two tab-separated columns, k in h/Mpc (positive, strictly increasing) and P in
(Mpc/h)^3 (positive). The table is interpolated as a cubic spline in log-log with
power-law tails beyond its end nodes; it should cover `sqrt(3) k_Nyquist` of the
bin's grid. The CAMB tables for the default bins (Planck 2018) come from:

```sh
pixi run -e tables pk-tables make --model linear
pixi run -e tables pk-tables make --model halofit
```

These write `data/matterpower_camb_{lin,halofit}_kmax10_zeff=<z>.tsv` (kh 1e-4 to 10)
for the seven default `z_eff`.

`configs/v28_halofit.yaml` uses these two sets. The default bin table (and
`configs/v28_default.yaml`) names `matterpower_camb_zeff=<z>.tsv` instead: linear, kh
1e-4 to 1, 10001 nodes. `make` writes those bytes under its own naming, so rename:

```sh
pixi run -e tables pk-tables make --model linear --kh-max 1 --npoints 10001
for z in 0.1 0.3 0.5 0.7 0.9 1.3 1.9; do
  mv "data/matterpower_camb_lin_kmax1_zeff=$z.tsv" "data/matterpower_camb_zeff=$z.tsv"
done
pixi run -e tables pk-tables check    # regenerates them, compares byte for byte
```

`make` also takes `--z`, `--kh-min` and `--out-dir`.

**Mask (optional).** A HEALPix 0/1 map of kept pixels in HDF5: any NSIDE, an integer
dataset (default name `MASK`), root attribute `ORDERING` = `NESTED` or `RING`, and
`PIXTYPE` = `HEALPIX` if present. Positions are looked up in redshift space.

## Quick start

```sh
pixi run logunusual default-config > my_run.yaml     # the default seven bins as YAML
pixi run logunusual run --config my_run.yaml --realizations 0:4 --dry-run   # plan only
pixi run logunusual run --config my_run.yaml --realizations 0:4
pixi run logunusual check runs/v28/realization_00000/catalog.parq
```

`run` takes `--realizations a:b` (half-open) or one index, `--bins i j ...` for a
subset, `--output-dir` to override the config's, and `--overwrite` (existing catalogs
are skipped otherwise). `check` verifies a catalog's layout and prints the realized
density per bin. `configs/v28_default.yaml` is the annotated example.

**Smoke run.** With the default tables in `data/`, this config runs the seven bins at
a quarter of the grid and 1% of the density:

```yaml
run_name: smoke
output_dir: runs
pk_dir: data
grid_scale: 0.25
nbar_scale: 0.01
```

On a 16-core CPU it takes about 6 s and 1.4 GB peak memory and writes 9.1 million
galaxies (0.21 GiB, full sky); `check` reports realized/target density within 0.5% of
1 in every bin.

## Configuration

A run config is YAML; `config.RunConfig` reads it and rejects unknown keys.

| key | default | meaning |
|---|---|---|
| `run_name` | required | output subdirectory |
| `output_dir` | required | output root |
| `pk_dir` | required | directory holding the P(k) tables |
| `bins` | the default table | list of bin records (below) |
| `mask` | none | `{path, dataset}` or a path (fields `mask_path`, `mask_dataset`; dataset default `MASK`) |
| `seed_base` | 137000000 | base of the seed schedule |
| `nbar_scale` | 1.0 | multiplies every bin's `nbar` |
| `grid_scale` | 1.0 | multiplies every `N_grid` (rounded to even, >= 4) |
| `radial_buffer` | 160.0 | Mpc/h drawn beyond each shell edge |
| `jitter_p` | 1 | in-cell placement order; 1 = uniform in the cell (2 is not attainable at typical cell sizes) |
| `n_workers` | null | sampler threads (null: the cores available); output is identical for any value |
| `angular_precut` | true | with a mask, draw only cells whose galaxies can land in a set pixel |
| `row_group_rows` | 1048576 | parquet row-group size |
| `n_radial_bins` | 8 | sub-shells of the radial-profile diagnostic in `summary.json` |
| `f_nl` | 0.0 | local f_NL, LSS convention; 0 = Gaussian |
| `fnl_p` | 1.0 | `p` in `b(k) = b + 2 (b - p) f_NL delta_c / M(k)` |
| `primordial` | the default tables' | `A_s`, `n_s`, `k_pivot` (h/Mpc), `omega_m` of the P(k) tables |

A bin record has `name`, `index` (1-127, written to the catalog), `z_min`, `z_max`,
`z_eff`, `rmin`, `rmax` (Mpc/h, inclusive), `L_box` (Mpc/h), `N_grid` (even), `nbar`
((Mpc/h)^-3), `b`, `f` (linear growth rate), `pk_file` (default
`matterpower_camb_zeff=<z_eff>.tsv`) and optionally `pk_galaxy_file`. Bin indices must
be ascending and unique.

- **Seeds.** Bin `i` of realization `r` uses `seed_base + 1000 r + i`
  (`0 <= r < 1000`), split into independent field and sampling seeds, so every
  realization is reproducible bin by bin and independent of which bins are run.
- **`radial_buffer`.** Galaxies are drawn in the shell widened by the buffer on both
  sides so that RSD can carry them across either edge. Before sampling, each bin
  computes the buffer its own displacement field needs and raises if the configured
  one is smaller; the requirement is recorded per bin. The buffer must also exceed a
  cell diagonal and fit the box (`rmax + buffer <= L_box / 2`).
- **`f_nl`, `fnl_p`, `primordial`.** f_NL changes only the galaxy target, through
  `b(k)`; the matter field and velocities are unchanged. `M(k)` is recovered from each
  bin's linear table, so `primordial` must match how the tables were made. Positive
  f_NL up to 100 runs on every default bin; negative f_NL raises on bins whose lowest
  modes it makes unattainable by a lognormal (table in
  [docs/construction.md](docs/construction.md#attainability)).
- **`pk_galaxy_file`.** A second table (e.g. halofit) for the galaxy target only;
  `pk_file` stays the linear table behind the matter field, the velocities and `M(k)`.
  The two must agree at their lowest k node (same redshift and cosmology), and the run
  raises if no lognormal reaches the galaxy target on the bin's grid.

The config hash (written to every catalog) covers the bins and every key except
`run_name`, `output_dir`, `pk_dir`, `n_workers` and the mask path (only whether a mask
is used); the f_NL keys enter only when `f_nl != 0`. Input files are hashed separately
into the catalog metadata.

## Output

`<output_dir>/<run_name>/realization_NNNNN/catalog.parq` and `summary.json`. Columns
`x, y, z` (float64, comoving Mpc/h, observer at the origin, redshift space) and `bin`
(int8); bins ascending and contiguous, row groups that never mix bins, provenance and
per-bin diagnostics in the file metadata. Full specification:
[docs/catalog.md](docs/catalog.md).

## Scripts

- `scripts/make_pk_tables.py` -- CAMB linear / halofit tables (`pk-tables` task).
- `scripts/gates_box.py` -- periodic-box statistical gates at full size, JSON report.
- `scripts/gates_shell.py` -- shell-product statistical gates at full size.
- `scripts/reproducibility.py` -- same seed in two fresh processes, compared bytewise.
- `scripts/device.py` -- field-stage device memory, CPU-vs-CUDA agreement, stage timing.
- `scripts/attainability.py` -- per bin, whether a lognormal reaches the galaxy target
  for a set of f_NL, grid scale and tables.
- `scripts/ensemble_check.py` -- ensemble readout over `summary.json` files.
- `scripts/ensemble.sbatch` -- Slurm job for ensemble production on one CPU node.

## Layout

- `logunusual/suite.py` -- default bin table, seed schedule, constants.
- `logunusual/{grid,pk,field,sample}.py` -- periodic-box generator.
- `logunusual/{shell,io,config,run,cli}.py` -- shell product, catalog format, run
  config, driver, command line.
- `logunusual/fnl.py` -- local-f_NL scale-dependent bias.
- `logunusual/{validate,gates}.py` -- estimators and statistical gates.
- `configs/` -- example run configs; `tests/` -- pytest suite (`tests/data/`: fixture
  tables).

## Documentation

- [docs/construction.md](docs/construction.md) -- the algorithm and its limits.
- [docs/catalog.md](docs/catalog.md) -- catalog file format and metadata.
- [docs/validation.md](docs/validation.md) -- what is tested and how.
- [docs/performance.md](docs/performance.md) -- wall time and memory.
- [docs/references.md](docs/references.md) -- literature and related codes.

Author: James Cheshire. License: BSD-3-Clause (`LICENSE`).
