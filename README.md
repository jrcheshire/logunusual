# logunusual

Fast lognormal galaxy mock generator for SPHEREx: JAX for the field stage (CPU or
CUDA), vectorised per-cell galaxy sampling, streamed parquet output. A drop-in
producer of the per-bin lognormal mock catalogs used by the SPHEREx L4 systematics
null tests, replacing the LogNormalGalaxies / LogNormalSimulations (Julia) path.

Status: **M1 periodic-box core** (spectrum -> lognormal fields -> Poisson catalog with
plane-parallel RSD -> validated P(k) multipoles; no shells / mask / parquet yet). See
`ROADMAP.md` for milestones and gates and `CLAUDE.md` for the construction and the
product contract.

## Install

```sh
pixi install          # conda env (default + gpu, one solve-group)
pixi run setup        # once: pinned CPU JAX + editable install of this package
pixi run test         # pytest
```

On a CUDA machine: `pixi install` then `pixi run -e gpu setup-gpu`, and use
`pixi run -e gpu ...` for everything.

## Layout

- `logunusual/suite.py` -- the v28 seven-bin table, seed schedule, mask and
  distance-cosmology constants (single source of truth).
- `logunusual/{grid,pk,field,sample,validate,gates}.py` -- the periodic-box generator
  and its validation (see `CLAUDE.md`).
- `scripts/m1_*.py` -- gate tables, cross-process reproducibility, memory profile.
- `docs/landscape.md` -- literature and code landscape.
- `ROADMAP.md` -- master plan with acceptance gates per milestone.

Author: James Cheshire. License: BSD-3.
