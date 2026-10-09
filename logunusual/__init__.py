"""logunusual -- fast lognormal galaxy mock generator (JAX + numpy).

Per-bin lognormal mock catalogs (observer-centred shells, radial RSD, one parquet of
x/y/z + bin per realization). The field stage runs on JAX (CPU or CUDA); the galaxy
stage is vectorised per cell. Entry points: `config.RunConfig`,
`run.generate_realization`, the `logunusual` CLI. Method in `docs/construction.md`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("logunusual")
except PackageNotFoundError:  # source checkout without an install
    __version__ = "0.0.0+unknown"

__author__ = "James Cheshire"
