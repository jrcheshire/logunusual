"""logunusual -- fast lognormal galaxy mock generator (JAX + numpy).

A drop-in producer of the SPHEREx per-bin lognormal mock catalogs (the prod_v2
contract: observer-centred shells, RSD applied, parquet with x/y/z + bin), built
so that the field stage runs on JAX (CPU or CUDA) and the galaxy stage is
vectorised per cell instead of looped per galaxy. See ROADMAP.md for milestones
and CLAUDE.md for the product contract and conventions.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("logunusual")
except PackageNotFoundError:  # source checkout without an install
    __version__ = "0.0.0+unknown"

__author__ = "James Cheshire"
