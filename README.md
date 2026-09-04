# logunusual

Fast lognormal galaxy mock generator for SPHEREx: JAX for the field stage (CPU or
CUDA), vectorised per-cell galaxy sampling, streamed parquet output. A drop-in
producer of the per-bin lognormal mock catalogs used by the SPHEREx L4 systematics
null tests, replacing the LogNormalGalaxies / LogNormalSimulations (Julia) path.

Status: **M0 bootstrap** (bin table + seed schedule only; no generator yet). See
`ROADMAP.md` for milestones and `CLAUDE.md` for the product contract.

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
- `docs/landscape.md` -- literature and code landscape.
- `ROADMAP.md` -- master plan with acceptance gates per milestone.

Author: James Cheshire. License: BSD-3.
