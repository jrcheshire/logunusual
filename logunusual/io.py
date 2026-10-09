"""The catalog file format (parquet) and its self-check (`docs/catalog.md`).

- Columns, in order: `x, y, z: float64` (Mpc/h, observer at the origin, redshift
  space) and `bin: int8` (the bin's index).
- Rows bin-contiguous, bins ascending; row groups of exactly `row_group_rows` (default
  2^20) except each bin's last, so none mixes bins; column statistics written (a
  reader picks a bin's row groups by `bin` min/max); one bin is one byte range.
- File-level metadata: flat string keys, global and per-bin `bin{index:02d}.*`.
- Written as `<name>.tmp` and renamed on close: a file with the final name is complete.
Entry points: `CatalogWriter`, `read_metadata`, `check_layout`, `read_bin`.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP_ROWS = 2**20
COLUMNS = ("x", "y", "z", "bin")
SCHEMA = pa.schema(
    [
        pa.field("x", pa.float64()),
        pa.field("y", pa.float64()),
        pa.field("z", pa.float64()),
        pa.field("bin", pa.int8()),
    ]
)
REQUIRED_KEYS = (
    "generator",
    "generator_version",
    "config_hash",
    "row_group_rows",
    "bins",
    "nbar_overdensity_factor",
    "rsd",
    "rng_scheme",
    "jitter_p",
    "radial_buffer",
)
REQUIRED_BIN_KEYS = (
    "index",
    "rmin",
    "rmax",
    "box_size",
    "n_grid",
    "nbar_target",
    "realized_nbar",
    "n_galaxies",
    "seed",
)


def bin_key(index: int, name: str) -> str:
    return f"bin{int(index):02d}.{name}"


def encode_metadata(meta: dict) -> dict:
    """Every value to `str` (parquet key-value metadata is bytes/str only)."""
    return {str(k): str(v) for k, v in meta.items()}


class CatalogWriter:
    """Streams `(xyz, bin_index)` slabs into the layout above.

    `write` may be called any number of times per bin with any slab sizes; bins must
    arrive in ascending order. `metadata` given at construction is written up front;
    `add_metadata` (before `close`) adds keys known only after writing (counts)."""

    def __init__(self, path, metadata=None, row_group_rows: int = ROW_GROUP_ROWS):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.row_group_rows = int(row_group_rows)
        if self.row_group_rows < 1:
            raise ValueError("row_group_rows must be >= 1")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"row_group_rows": self.row_group_rows}
        meta.update(metadata or {})
        self._writer = pq.ParquetWriter(
            self.tmp, SCHEMA.with_metadata(encode_metadata(meta)), write_statistics=True
        )
        self._buf = []
        self._buf_rows = 0
        self._bin = None
        self._last_bin = None
        self.rows_per_bin = {}
        self.row_groups = 0
        self._closed = False

    def write(self, xyz, bin_index: int):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (n, 3), got {xyz.shape}")
        bin_index = int(bin_index)
        if not -128 <= bin_index <= 127:
            raise ValueError("bin index must fit int8")
        if self._bin is not None and bin_index != self._bin:
            self.end_bin()
        if self._last_bin is not None and bin_index < self._last_bin:
            raise ValueError(
                f"bins must be written in ascending order: {bin_index} after "
                f"{self._last_bin}"
            )
        self._bin = bin_index
        self._last_bin = bin_index
        self.rows_per_bin.setdefault(bin_index, 0)
        if xyz.shape[0] == 0:
            return
        self._buf.append(xyz)
        self._buf_rows += xyz.shape[0]
        while self._buf_rows >= self.row_group_rows:
            self._flush(self.row_group_rows)

    def _flush(self, n_rows: int):
        if n_rows == 0:
            return
        data = self._buf[0] if len(self._buf) == 1 else np.concatenate(self._buf)
        chunk, rest = data[:n_rows], data[n_rows:]
        self._buf = [rest] if rest.shape[0] else []
        self._buf_rows = int(rest.shape[0])
        table = pa.table(
            {
                "x": np.ascontiguousarray(chunk[:, 0]),
                "y": np.ascontiguousarray(chunk[:, 1]),
                "z": np.ascontiguousarray(chunk[:, 2]),
                "bin": np.full(chunk.shape[0], self._bin, dtype=np.int8),
            },
            schema=SCHEMA,
        )
        self._writer.write_table(table, row_group_size=n_rows)
        self.rows_per_bin[self._bin] += int(n_rows)
        self.row_groups += 1

    def end_bin(self):
        """Flush the partial row group of the current bin."""
        if self._bin is None:
            return
        self._flush(self._buf_rows)
        self._bin = None

    def add_metadata(self, metadata: dict):
        self._writer.add_key_value_metadata(encode_metadata(metadata))

    def close(self):
        if self._closed:
            return
        self.end_bin()
        self._writer.close()
        os.replace(self.tmp, self.path)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            try:
                self._writer.close()
            finally:
                if self.tmp.exists():
                    self.tmp.unlink()


def read_metadata(path) -> dict:
    """File-level key-value metadata as `{str: str}` (pyarrow's own keys dropped).
    Read from the footer (`FileMetaData.metadata`), which holds both the keys written
    with the schema and those added at close."""
    md = pq.ParquetFile(path).metadata.metadata or {}
    out = {}
    for k, v in md.items():
        k = k.decode() if isinstance(k, bytes) else k
        if k.startswith("ARROW:") or k == "pandas":
            continue
        out[k] = v.decode() if isinstance(v, bytes) else v
    return out


def bin_metadata(meta: dict, index: int) -> dict:
    """The `bin{index:02d}.*` keys of `meta`, prefix stripped."""
    pre = f"bin{int(index):02d}."
    return {k[len(pre) :]: v for k, v in meta.items() if k.startswith(pre)}


@dataclass
class LayoutReport:
    path: str
    n_rows: int
    n_row_groups: int
    row_group_rows: int | None
    bins: list  # ascending bin indices present in the data
    rows_per_bin: dict
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _rg_start(rg):
    c = rg.column(0)
    return c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset


def _rg_size(rg):
    return sum(rg.column(j).total_compressed_size for j in range(rg.num_columns))


def check_layout(path) -> LayoutReport:
    """Verify the layout without reading the data: schema and column order, metadata
    keys, per-row-group `bin` statistics (min == max, ascending across groups),
    the row-group pin (every group full except each bin's last), byte contiguity of
    consecutive row groups, and the per-bin row counts against the metadata."""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    meta = read_metadata(path)
    problems = []
    names = tuple(pf.schema_arrow.names)
    if names != COLUMNS:
        problems.append(f"columns are {names}, expected {COLUMNS}")
    else:
        for name, want in zip(COLUMNS, SCHEMA.types):
            got = pf.schema_arrow.field(name).type
            if got != want:
                problems.append(f"column {name} is {got}, expected {want}")
    for k in REQUIRED_KEYS:
        if k not in meta:
            problems.append(f"missing metadata key {k!r}")
    if meta.get("generator") != "logunusual":
        problems.append(f"generator is {meta.get('generator')!r}")
    R = int(meta["row_group_rows"]) if "row_group_rows" in meta else None

    bi = pf.schema_arrow.get_field_index("bin") if "bin" in names else None
    seq = []  # (bin, num_rows) per row group
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        if bi is None:
            break
        st = rg.column(bi).statistics
        if st is None or not st.has_min_max:
            problems.append(f"row group {i}: no bin statistics")
            continue
        if st.min != st.max:
            problems.append(f"row group {i}: mixes bins {st.min}..{st.max}")
        seq.append((int(st.min), rg.num_rows))
        if i + 1 < md.num_row_groups:
            a, b = rg, md.row_group(i + 1)
            if _rg_start(b) != _rg_start(a) + _rg_size(a):
                problems.append(f"byte gap between row groups {i} and {i + 1}")
    bins_seen = [b for b, _ in seq]
    if bins_seen != sorted(bins_seen):
        problems.append(f"bins are not ascending across row groups: {bins_seen}")
    rows_per_bin = {}
    for j, (b, n) in enumerate(seq):
        rows_per_bin[b] = rows_per_bin.get(b, 0) + n
        last_of_bin = j + 1 == len(seq) or seq[j + 1][0] != b
        if R is not None and not last_of_bin and n != R:
            problems.append(f"row group {j} (bin {b}) has {n} rows, expected {R}")
        if R is not None and n > R:
            problems.append(f"row group {j} (bin {b}) has {n} rows > {R}")
    declared = meta.get("bins", "")
    declared = [int(x) for x in declared.split(",") if x.strip()] if declared else []
    for b in rows_per_bin:
        if declared and b not in declared:
            problems.append(f"bin {b} present in data but not declared: {declared}")
    for b in declared:
        n_meta = meta.get(bin_key(b, "n_galaxies"))
        if n_meta is None:
            problems.append(f"missing metadata key {bin_key(b, 'n_galaxies')!r}")
        elif int(n_meta) != rows_per_bin.get(b, 0):
            problems.append(
                f"bin {b}: metadata n_galaxies {n_meta} != rows "
                f"{rows_per_bin.get(b, 0)}"
            )
        for k in REQUIRED_BIN_KEYS:
            if bin_key(b, k) not in meta:
                problems.append(f"missing metadata key {bin_key(b, k)!r}")
    if md.num_rows != sum(rows_per_bin.values()):
        problems.append("row-group rows do not add up to the file's row count")
    return LayoutReport(
        path=str(path),
        n_rows=md.num_rows,
        n_row_groups=md.num_row_groups,
        row_group_rows=R,
        bins=sorted(rows_per_bin),
        rows_per_bin=rows_per_bin,
        problems=problems,
    )


def read_bin(path, index: int, columns=("x", "y", "z")):
    """Read one bin's rows by row-group statistics (no full scan). Returns a dict of
    numpy arrays."""
    pf = pq.ParquetFile(path)
    bi = pf.schema_arrow.get_field_index("bin")
    picks = []
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(bi).statistics
        if st is not None and st.min <= index <= st.max:
            picks.append(i)
    if not picks:
        return {c: np.empty(0, dtype=np.float64) for c in columns}
    tab = pf.read_row_groups(picks, columns=list(columns) + ["bin"])
    sel = np.asarray(tab["bin"]) == index
    return {c: np.asarray(tab[c])[sel] for c in columns}
