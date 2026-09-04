# logunusual

Lognormal galaxy mock generator: JAX for the field stage (CPU or CUDA), vectorised
per-cell galaxy sampling, streamed parquet output. A run is a list of bins, each a
periodic box centred on the observer holding one radial shell; galaxies are drawn in
the buffered shell, displaced radially by their cell's velocity, cut to the shell and
an optional angular HEALPix mask, and written as one parquet per realization. The
default bin table is a seven-shell survey forecast (`logunusual/suite.py`); any table
with the same fields can be given in the run config.

Status: **M2 shell product** (spectrum -> lognormal fields -> Poisson shell catalog
with radial RSD and mask -> parquet). See `ROADMAP.md` for milestones and gates and
`CLAUDE.md` for the construction and the catalog format.

## Install

```sh
pixi install          # conda env (default + gpu, one solve-group)
pixi run setup        # once: pinned CPU JAX + editable install (gives the `logunusual` command)
pixi run test         # pytest (fast tests + slow statistical gates)
```

On a CUDA machine: `pixi install` then `pixi run -e gpu setup-gpu`, and use
`pixi run -e gpu ...` for everything.

## Run

```sh
pixi run logunusual default-config > my_run.yaml   # the seven-bin default; edit paths
pixi run logunusual run --config my_run.yaml --realizations 0:4 --dry-run
pixi run logunusual run --config my_run.yaml --realizations 0:4
pixi run logunusual check runs/<run_name>/realization_00000/catalog.parq
```

`configs/v28_default.yaml` is the annotated example. Inputs: one two-column P(k) TSV
per bin (`k [h/Mpc]`, `P [(Mpc/h)^3]`) in `pk_dir`, and optionally a HEALPix 0/1 mask
in HDF5 (root attributes `PIXTYPE=HEALPIX`, `ORDERING=NESTED|RING`). `nbar_scale` and
`grid_scale` shrink a run for smoke tests. Seeds follow `seed_base + realization *
1000 + bin index`, so realizations are reproducible bin by bin.

Output: `<output_dir>/<run_name>/realization_NNNNN/catalog.parq` with columns
`x, y, z` (float64, Mpc/h, observer at the origin, redshift space) and `bin` (int8),
bins ascending, row groups pinned at 2^20 rows and never mixing bins, provenance in
the file metadata; plus a `summary.json`. `logunusual check` verifies the layout and
prints the realized density per bin.

## Layout

- `logunusual/suite.py` -- the default bin table, seed schedule, constants.
- `logunusual/{grid,pk,field,sample}.py` -- the periodic-box generator.
- `logunusual/{shell,io,config,run,cli}.py` -- the shell product, catalog format,
  run config, driver, command line.
- `logunusual/{validate,gates}.py` -- estimators and statistical gates.
- `scripts/m1_*.py`, `scripts/m2_gates.py` -- gate tables, reproducibility, memory.
- `docs/landscape.md` -- literature and code landscape.
- `ROADMAP.md` -- master plan with acceptance gates per milestone.

Author: James Cheshire. License: BSD-3.
