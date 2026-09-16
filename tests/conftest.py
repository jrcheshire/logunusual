"""Shared fixtures. JAX x64 is forced before any array work (field.py also does it)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from logunusual.pk import PowerSpectrum  # noqa: E402

DATA = Path(__file__).parent / "data"
PK_TSV = DATA / "matterpower_camb_zeff=0.9.tsv"  # CAMB linear P(k), z = 0.9


@pytest.fixture(scope="session")
def spectrum():
    return PowerSpectrum.from_tsv(PK_TSV)
