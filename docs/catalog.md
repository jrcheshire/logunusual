# Catalog format

The package's own output specification, implemented in `logunusual/io.py` (writer,
reader, self-check) and `logunusual/run.py` (what is written).

## Paths

```
<output_dir>/<run_name>/realization_NNNNN/catalog.parq     # one file per realization
<output_dir>/<run_name>/realization_NNNNN/summary.json     # run summary, JSON
```

`NNNNN` is the realization index, zero-padded to five digits. A realization whose
`catalog.parq` exists is skipped unless `run --overwrite` is given.

## Parquet layout

| property | specification |
|---|---|
| columns | exactly `x, y, z` (float64) then `bin` (int8), in this order |
| `x, y, z` | comoving Mpc/h, observer at the origin, redshift space (after radial RSD) |
| `bin` | the bin's `index` from the run config (1-127); always present |
| row order | bin-contiguous, bins ascending; within a bin, by x-slab of the bin's grid |
| row groups | exactly `row_group_rows` rows (default 2^20), except the last row group of each bin; a row group never mixes bins |
| statistics | column min/max written for every row group, so a bin's row groups are found from the `bin` statistics without reading data |
| byte layout | consecutive row groups are byte-contiguous, so one bin is one byte range of the file |
| atomicity | written as `catalog.parq.tmp` and renamed on close: a file with the final name is complete; on an error the temporary file is removed |

Nothing else is stored per galaxy: no velocities, redshifts, weights, or randoms.
Each bin's galaxies come from that bin's own periodic box; all bins share the
observer origin and axis orientation. With a mask, only galaxies whose
redshift-space direction falls in a set pixel are written; without one, the shells
are full sky.

## Reading

```python
from logunusual import io

meta = io.read_metadata(path)        # {str: str}, all file-level keys
b5 = io.bin_metadata(meta, 5)        # the bin05.* keys, prefix stripped
cols = io.read_bin(path, 5)          # {"x", "y", "z"}: numpy arrays of bin 5 only
report = io.check_layout(path)       # LayoutReport; report.ok, report.problems
```

`read_bin` selects row groups by their `bin` statistics, so it reads only that bin.
Any parquet reader works; `pyarrow.parquet.ParquetFile(path).metadata.metadata`
holds the key-value metadata.

## Layout check

`io.check_layout` (run by `logunusual check`) verifies without reading the data:

- column names, order and types;
- presence of the required global keys (`generator`, `generator_version`,
  `config_hash`, `row_group_rows`, `bins`, `nbar_overdensity_factor`, `rsd`,
  `rng_scheme`, `jitter_p`, `radial_buffer`) and `generator == "logunusual"`;
- per row group: `bin` statistics present and min == max; bins ascending across row
  groups;
- the row-group pin: every row group has exactly `row_group_rows` rows except each
  bin's last, and none has more;
- byte contiguity of consecutive row groups;
- every bin in the data is listed in `bins`; for every listed bin the required
  per-bin keys (`index`, `rmin`, `rmax`, `box_size`, `n_grid`, `nbar_target`,
  `realized_nbar`, `n_galaxies`, `seed`) are present and `n_galaxies` equals the bin's
  row count; the row groups add up to the file's row count.

`logunusual check` prints the rows, target and realized density per bin, and exits 1
if any problem is found.

## Metadata

File-level key-value metadata; every value is a string (numbers via `str()`, so
`None` appears as `None` and booleans as `True` / `False`). Global keys are plain;
per-bin keys are prefixed `bin{index:02d}.` (e.g. `bin05.realized_nbar`).

### Global keys

| key | value |
|---|---|
| `generator`, `generator_version` | `logunusual` and the installed package version |
| `created` | UTC timestamp, ISO 8601, seconds |
| `host`, `machine`, `system` | `platform.node()`, `platform.machine()`, `platform.system()` |
| `python_version`, `numpy_version`, `jax_version` | library versions |
| `jax_backend` | `cpu` or `gpu` |
| `run_name`, `config_hash` | from the run config; the hash is SHA-256 over the mock definition |
| `seed_base`, `nbar_scale`, `grid_scale`, `radial_buffer`, `jitter_p`, `angular_precut` | run-config values |
| `n_workers` | sampler threads actually used (does not affect the catalog) |
| `row_group_rows` | the row-group pin |
| `bins` | comma-separated bin indices in the file, e.g. `1,2,3,4,5,6,7` |
| `rsd` | `radial` |
| `rng_scheme` | `philox-per-x-slab` |
| `velocity_assignment` | `own-cell` |
| `frame` | `observer at origin; each bin its own periodic box centred there` |
| `shell_edges` | `inclusive` |
| `nbar_overdensity_factor` | `1.0`: each bin targets its configured `nbar` (times `nbar_scale`) |
| `f_nl` | local f_NL (LSS convention); always written |
| `mask` | mask file name, or `none` |
| `mask_fsky` | fraction of set pixels; `1.0` without a mask |
| `mask_dataset`, `mask_sha256`, `mask_nside`, `mask_ordering` | with a mask only |
| `fnl_convention` | `LSS`; with `f_nl != 0` only |
| `fnl_p`, `fnl_delta_c`, `fnl_A_s`, `fnl_n_s`, `fnl_k_pivot`, `fnl_omega_m`, `fnl_g0` | f_NL settings and `g0`; with `f_nl != 0` only |

### Per-bin keys (`binNN.*`)

| key | value |
|---|---|
| `index`, `name` | bin index and name |
| `z_min`, `z_max`, `z_eff` | from the bin record |
| `rmin`, `rmax` | shell edges, Mpc/h (inclusive) |
| `box_size`, `n_grid`, `cell` | box side, grid points per side (after `grid_scale`), cell size |
| `b`, `f` | bias and growth rate |
| `nbar_nominal`, `nbar_scale`, `nbar_target` | configured `nbar`, scale, and their product |
| `realized_nbar` | `n_galaxies / (mask_fsky * 4/3 pi (rmax^3 - rmin^3))` |
| `realized_over_target` | `realized_nbar / nbar_target` |
| `n_galaxies` | galaxies written for this bin |
| `n_drawn` | galaxies drawn in the buffered window before the shell and mask cut |
| `n_left_box` | drawn galaxies shifted outside the box by RSD (never kept) |
| `lam_window` | sum of the Poisson means over the drawn cells (expected `n_drawn`) |
| `n_window_cells`, `n_window_cells_radial` | cells drawn; cells of the radial window before the angular pre-cut |
| `radial_buffer` | buffer used, Mpc/h |
| `required_buffer` | smallest buffer this realization's displacement field needs, Mpc/h |
| `psi_max` | largest `|Psi|` over the radial window, Mpc/h (growth rate not applied) |
| `psi_rms` | JSON object `{"x": ..., "y": ..., "z": ...}`: rms of each displacement component over the box, Mpc/h |
| `seed`, `ic_seed`, `draw_seed` | scheduled seed and the field and sampling seeds split from it |
| `pk_file`, `pk_sha256` | linear table name and SHA-256 |
| `pk_galaxy_file`, `pk_galaxy_sha256` | galaxy-target table and SHA-256 (the linear table's when the bin names none) |
| `sigma2_galaxy`, `sigma2_matter` | cell variance of the galaxy and matter targets on the grid |
| `xi_min_galaxy` | minimum of the galaxy target's grid correlation function |
| `clipped_power_fraction` | galaxy `P_G` power set to zero over the power kept; 0 when the target is attainable |
| `t_field_s`, `t_sample_s` | wall time of the field stage and of sampling + writing, s |
| `peak_live_jax_bytes` | peak bytes of live JAX arrays during the field stage |
| `fnl_delta_b_kf`, `fnl_b_kf_over_b` | `b(k) - b` and `b(k) / b` at the fundamental; with `f_nl != 0` only |
| `fnl_k_zero` | lowest grid `|k|` at which `b(k)` changes sign, or `None`; with `f_nl != 0` only |

## summary.json

Written after the catalog, next to it:

| key | value |
|---|---|
| `realization`, `run_name`, `config_hash`, `path` | identification |
| `bins` | list, one object per bin (below) |
| `n_galaxies` | total over bins |
| `wall_s` | wall time of the realization, s |
| `peak_live_jax_bytes` | largest per-bin value |
| `file_bytes` | size of `catalog.parq` |

Each entry of `bins` holds the per-bin keys above as native JSON types (`psi_rms` an
object, `fnl_k_zero` `null` when `b(k)` keeps its sign), plus the radial profile:
`r_edges`, `n_radial_bins + 1` uniform radii from `rmin` to `rmax`, and `r_hist`, the
galaxies written per radial sub-shell.

`scripts/ensemble_check.py` reads these files to check an ensemble (completeness, one
config hash, distinct seeds, mean realized density per bin).
