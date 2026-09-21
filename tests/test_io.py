"""Catalog format contracts: layout self-check, row-group pin, chunk invariance of the
bytes, metadata round trip, empty bins, ordering errors, atomic completion."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from logunusual import io

META = {
    "generator": "logunusual",
    "generator_version": "test",
    "config_hash": "abc",
    "bins": "1,2,3",
    "nbar_overdensity_factor": 1.0,
    "rsd": "radial",
    "rng_scheme": "philox-per-x-slab",
    "jitter_p": 1,
    "radial_buffer": 20.0,
}


def _bin_meta(idx, n):
    return {
        io.bin_key(idx, k): v
        for k, v in dict(
            index=idx,
            rmin=1.0,
            rmax=2.0,
            box_size=10.0,
            n_grid=4,
            nbar_target=1.0,
            realized_nbar=1.0,
            n_galaxies=n,
            seed=idx,
        ).items()
    }


def _write(path, slabs, row_group_rows=100, meta=META):
    """`slabs`: list of (bin_index, n_rows) in order; returns per-bin totals."""
    rng = np.random.default_rng(0)
    totals = {}
    with io.CatalogWriter(path, meta, row_group_rows=row_group_rows) as w:
        for b, n in slabs:
            w.write(rng.uniform(-1, 1, (n, 3)), b)
            totals[b] = totals.get(b, 0) + n
        for b in (1, 2, 3):
            totals.setdefault(b, 0)
        w.add_metadata(
            {k: v for b in (1, 2, 3) for k, v in _bin_meta(b, totals[b]).items()}
        )
    return totals


def test_layout_pin_and_check(tmp_path):
    p = tmp_path / "c.parq"
    totals = _write(p, [(1, 250), (1, 7), (2, 100), (2, 1), (3, 99)], 100)
    rep = io.check_layout(p)
    assert rep.ok, rep.problems
    assert rep.bins == [1, 2, 3] and rep.rows_per_bin == totals
    sizes = [
        pq.ParquetFile(p).metadata.row_group(i).num_rows
        for i in range(rep.n_row_groups)
    ]
    assert sizes == [100, 100, 57, 100, 1, 99]
    assert tuple(pq.ParquetFile(p).schema_arrow.names) == io.COLUMNS
    assert not p.with_name("c.parq.tmp").exists()
    d = io.read_bin(p, 2)
    assert d["x"].size == 101


def test_bytes_do_not_depend_on_slab_sizes(tmp_path):
    a = tmp_path / "a.parq"
    b = tmp_path / "b.parq"
    rng = np.random.default_rng(5)
    xyz = rng.uniform(-1, 1, (1000, 3))
    for path, cuts in ((a, [0, 300, 301, 1000]), (b, [0, 1000])):
        with io.CatalogWriter(path, META, row_group_rows=128) as w:
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                w.write(xyz[lo:hi], 1)
            w.add_metadata(_bin_meta(1, 1000) | _bin_meta(2, 0) | _bin_meta(3, 0))
    assert a.read_bytes() == b.read_bytes()
    t = pq.read_table(a)
    assert np.array_equal(np.asarray(t["x"]), xyz[:, 0])
    assert np.array_equal(np.asarray(t["bin"]), np.ones(1000, np.int8))


def test_metadata_round_trip_and_empty_bin(tmp_path):
    p = tmp_path / "c.parq"
    _write(p, [(1, 10), (3, 5)], 100)  # bin 2 declared, no rows
    meta = io.read_metadata(p)
    for k, v in META.items():
        assert meta[k] == str(v)
    assert meta["row_group_rows"] == "100"
    assert io.bin_metadata(meta, 2)["n_galaxies"] == "0"
    rep = io.check_layout(p)
    assert rep.ok, rep.problems
    assert rep.rows_per_bin == {1: 10, 3: 5}
    assert io.read_bin(p, 2)["x"].size == 0


def test_check_layout_flags_problems(tmp_path):
    p = tmp_path / "c.parq"
    _write(p, [(1, 10)], 100, meta={**META, "bins": "1,2,3"})
    # wrong count in metadata
    bad = tmp_path / "bad.parq"
    with io.CatalogWriter(bad, META, row_group_rows=100) as w:
        w.write(np.zeros((10, 3)), 1)
        w.add_metadata(_bin_meta(1, 11) | _bin_meta(2, 0) | _bin_meta(3, 0))
    rep = io.check_layout(bad)
    assert any("n_galaxies 11" in x for x in rep.problems)
    # a plain parquet without the layout
    import pyarrow as pa

    pq.write_table(pa.table({"x": [1.0], "bin": [1]}), tmp_path / "plain.parq")
    rep = io.check_layout(tmp_path / "plain.parq")
    assert not rep.ok and any("columns" in x for x in rep.problems)


def test_writer_rejects_descending_bins_and_cleans_up_on_error(tmp_path):
    p = tmp_path / "c.parq"
    with pytest.raises(ValueError, match="ascending"):
        with io.CatalogWriter(p, META) as w:
            w.write(np.zeros((3, 3)), 2)
            w.write(np.zeros((3, 3)), 1)
    assert not p.exists() and not p.with_name("c.parq.tmp").exists()
    with pytest.raises(ValueError, match="ascending"):
        with io.CatalogWriter(p, META) as w:
            w.write(np.zeros((3, 3)), 1)
            w.write(np.zeros((3, 3)), 2)
            w.write(np.zeros((3, 3)), 1)  # a closed bin cannot be reopened
