# Performance

What one realization costs, where the time and memory go, how the CPU and GPU compare,
and how to size a run. Every number is marked **measured** (on the hardware named) or
**predicted** (scaled from a measurement on other hardware). GiB = 2^30 bytes,
GB = 10^9 bytes. "N^3 f64" is the size of one float64 array on the N^3 grid (1 GiB at
512^3).

Contents: [hardware](#hardware) · [a full realization](#a-full-realization) ·
[sample stage](#sample-stage) · [field stage](#field-stage) ·
[CPU vs GPU agreement](#cpu-vs-gpu-agreement) · [memory](#memory) ·
[dtype and jit](#dtype-and-jit) ·
[ensemble production](#ensemble-production-on-one-cpu-node) ·
[measuring on your own machine](#measuring-on-your-own-machine).

## Hardware

| name used below | what it is |
|---|---|
| Apple M4 Max | 16-core Apple M4 Max CPU, JAX CPU backend (osx-arm64) |
| GH200 node | NVIDIA GH200 120 GB node (72 Grace cores); JAX allocator limit 71.25 GiB |
| RTX 3050 | NVIDIA RTX 3050 (6 GB); JAX allocator limit 4.25 GiB by default, 5.38 GiB with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` |
| CPU node | 32-core x86-64 Linux node, JAX CPU backend (linux-64) |

## A full realization

**Measured** on the CPU node, `configs/v28_halofit.yaml` (seven v28 bins at full
density, radial buffer 160 Mpc/h, survey mask fsky 0.713), 16 sampler threads, with a
second identical process running on the node's other 16 cores. Medians over 100
realizations, `[min, max]` where it matters:

| bin | N | L (Mpc/h) | field (s) | sample + write (s) | drawn / kept | kept |
|---|---|---|---|---|---|---|
| 1 | 192 | 1500 | 1.2 | 3.0 | 2.19 | 40M |
| 2 | 256 | 2500 | 1.8 | 9.1 | 1.77 | 133M |
| 3 | 384 | 3500 | 4.9 | 9.0 | 1.79 | 108M |
| 4 | 448 | 4500 | 7.8 | 11.5 | 1.87 | 123M |
| 5 | 512 | 5000 | 11.5 | 13.8 | 1.96 | 120M |
| 6 | 512 | 7000 | 10.0 | 10.7 | 1.42 | 102M |
| 7 | 512 | 8000 | 8.5 | 5.0 | 1.56 | 23M |

| per realization | value |
|---|---|
| wall, generator | 111 s median, [99, 137] s |
| of which field / sample + write | ~46 s / ~62 s (sum of the bin medians) |
| peak host RSS | 11.2-12.3 GB |
| peak live JAX arrays | 7.0 x N^3 f64 in every bin (7.0 GiB at 512^3) |
| galaxies kept / drawn | 6.5e8 / ~1.2e9 |
| catalog size | 15.04 GiB |

The field-stage column for the same seven grids on the other hardware is under
[field stage](#field-stage). Sample-stage times scale with the density and the drawn
volume, so a run at `nbar_scale` < 1 or without a mask is cheaper in that column.

## Sample stage

**Why it dominates.** The field stage is a fixed number of FFTs on the grid; the sample
stage is per-galaxy work: a Poisson draw per window cell, uniform-in-cell placement,
radial RSD from the galaxy's own cell, the HEALPix lookup and shell cut, and the
parquet write. It draws more than it keeps because the window extends a full buffer
beyond both shell edges; with a mask, the angular pre-cut drops the cells whose
galaxies cannot land in a set pixel (0.73-0.75 of the radial window's cells survive at
fsky 0.713), which leaves 1.4-2.2 drawn per kept galaxy.

**How it threads.** One x-slab is one task on a thread pool. Each slab draws on its
own Philox stream and builds its own window, and numpy and healpy release the GIL in
the draw, placement, RSD and pixel lookup, so slabs run in parallel; slabs are yielded
in order, so the catalog does not depend on the thread count. The thread count is
`n_workers` in the run config (default: the cores in the process's affinity mask,
`sample.default_workers`). Some of the stage runs on one thread: the intensity, the
whole-box buffer-requirement pass (`Shell.required_buffer`, a loop over slabs) and the
in-order parquet write.

**Measured** on the Apple M4 Max, bin 2 (256^3, survey mask fsky 0.71, 150 Mpc/h
buffer), one realization, no parquet write:

| threads | angular pre-cut | sample stage | galaxies drawn |
|---|---|---|---|
| 1 | off | 19.8 s | 311M |
| 8 | off | 3.2 s | 311M |
| 16 | off | 2.4 s | 311M |
| 16 | on | 2.1 s | 230M |

**Observed, cause not measured:** the stage stops scaling past ~16 threads. Over a
full realization, 72 sampler threads on the GH200 node gave the same sample-stage time
as 16 on the Apple M4 Max (per bin, faster on some bins and slower on others), with
about 2 cores busy on average over the whole process. The ensemble script therefore
runs several 16-thread processes per node rather than one process on all its cores
(see [ensemble production](#ensemble-production-on-one-cpu-node)).

## Field stage

White noise (numpy, host) -> galaxy and matter `P -> P_G` conversions -> two lognormal
fields -> three displacement components (JAX, CPU or CUDA).

**Per bin, in a full realization (measured).** Seconds; the Apple M4 Max and GH200
node ran the v28 default configuration with the same seeds, the CPU node the
[full realization](#a-full-realization) above (median, with a second process on the
node):

| bin | N | Apple M4 Max | GH200 node | CPU node |
|---|---|---|---|---|
| 1 | 192 | 0.8 | 2.2 | 1.2 |
| 2 | 256 | 1.1 | 2.3 | 1.8 |
| 3 | 384 | 2.4 | 6.7 | 4.9 |
| 4 | 448 | 5.0 | 5.9 | 7.8 |
| 5 | 512 | 7.7 | 6.8 | 11.5 |
| 6 | 512 | 7.1 | 1.9 | 10.0 |
| 7 | 512 | 7.3 | 1.9 | 8.5 |
| total | | 31.5 | 27.8 | ~46 |

**Observed, cause not measured: a first-use cost per grid size on the GPU.** On the
GH200 the second and third 512^3 bins take 1.9 s, matching the standalone 2.0 s below,
while the first 512^3 bin (bin 5) takes 6.8 s and the only 384^3 bin 6.7 s. Scaling
the 512^3 time by `N^3 log N`, bins 1-5 would need ~4.4 s against the 24.1 s they take,
so ~20 s of the GPU's field time per realization is first-use cost (an estimate). The
pattern fits a per-shape compile or FFT-plan cost. The CPU shows no such step.

**One 512^3 bin, standalone (measured,** `scripts/device.py wall --bin bin05`**).**
"Blocked" waits on every array at each step and is an upper bound; "unblocked" is the
end-to-end time:

| | Apple M4 Max | GH200 node |
|---|---|---|
| field stage, blocked | 5.16 s | 3.55 s |
| field stage, unblocked | 4.62 s | 2.01 s |

Apple M4 Max, blocked, per step: white noise 0.52, white-noise FFT 0.07, galaxy
`P -> P_G` 1.32, galaxy field 0.33, matter `P -> P_G` 1.24, matter field 0.28, matter
FFT 0.07, displacements 0.40 / 0.42 / 0.45 s (x / y / z).

**The `P -> P_G` conversion, split (measured,** `scripts/device.py pkg --bin bin05`**,
median of 5).** Each piece is the library call `generate_fields` makes, timed in
place; the assembled result is checked bitwise against the shipped call in the same
process.

| piece | kind | Apple M4 Max | GH200 node |
|---|---|---|---|
| radius index `i^2 + j^2 + l^2` | host | 0.023 s | 0.026 s |
| spectrum on each integer radius | host | 0.003 s | 0.004 s |
| gather onto the grid | host | 0.117 s | 0.231 s |
| placement window + divide | host | 0.088 s | 0.128 s |
| upload | upload | 0.007 s | 0.035 s |
| `irfftn` -> `xi`, `log1p`, `rfftn` -> `P_G`, clip | device | 0.875 s | 0.014 s |
| value syncs (`xi_min`, `sigma2`, clipping) | sync | 0.089 s | 0.003 s |
| **one conversion** | | **1.20 s (host 19%)** | **0.44 s (host 88%)** |

The spectrum is evaluated once per attainable integer radius `q <= 3 (N/2)^2` and
gathered by index (`pk.pk_on_grid`), not on every grid point; this agrees with direct
evaluation to ~1e-15 relative (bitwise it does not). On the CPU the conversion is
mostly FFTs (`irfftn` is the one bimodal piece, 0.16-0.68 s); on the GH200 the device
work is 3% of it and the host gather alone is half. White noise is also host work
(0.5-0.6 s at 512^3 on both machines). Summed, the host pieces (~0.6 s of white noise
and ~0.8 s for the two conversions) are most of the GH200's 2.0 s stage, which is
~2.3x faster than the Apple M4 Max's 4.6 s.

## CPU vs GPU agreement

**Measured** with `scripts/device.py ulp`, which runs the field stage on both backends
in one process (one jaxlib build on both sides), on the RTX 3050 and the GH200 node:
median 6-20 ulp per cell (120 on the matter field at 64^3 on the GH200), largest
absolute difference 4e-15 to 1.6e-13 on fields of rms 0.18-3.8. **Not bitwise.** The
largest per-cell ulp count is not a meaningful figure: it blows up where a field
crosses zero.

In a full seven-bin realization run on both the GH200 node (GPU field stage) and the
Apple M4 Max (CPU) with the same seeds, the kept, drawn and left-the-box galaxy counts
agree exactly in every bin; the window's `psi_max` agrees to 3.4e-15 relative and the
galaxy field's sigma^2 to 3e-16.

## Memory

### Device: live arrays vs allocator peak

Two instruments, and they differ:

- **Live arrays** (`jax.live_arrays()`, polled at each step; recorded per bin in the
  catalog metadata as `peak_live_jax_bytes`): 7.0 x N^3 f64 in every bin on the CPU
  (7.01-7.02 over the 100-realization ensemble), 7.5-8.5 x N^3 on a GPU. The poller
  samples, with ~10% run-to-run spread, so it cannot settle a percent-level question.
- **Device allocator peak** (`peak_bytes_in_use`, GPU only; the CPU backend has no
  allocator to report): also counts XLA's intra-op scratch, 50-60% on top of the live
  arrays. **This is the number that decides whether a grid fits.**

**Measured** on the GH200 node, allocator peak in units of N^3 f64:

| N | f64 eager (default) | f64 jit | f32 eager | f32 jit |
|---|---|---|---|---|
| 192 | 12.09-12.59 | 10.05 | 6.30 | 5.03 |
| 256 | 12.07-12.57 | 10.04 | 6.29 | 5.02 |
| 384 | 12.04 | 10.03 | 6.28 | 5.02 |
| 448 | 12.04 | 10.02 | 6.02 | 5.08 |
| 512 | 12.03 (12.03 GiB) | 10.02 (10.02 GiB) | 6.02 (6.02 GiB) | 5.07 (5.07 GiB) |

The default field stage needs **12.0 x N^3 f64 at every production grid**: 0.64,
1.51, 5.08, 8.06 and 12.03 GiB at 192^3 to 512^3. float32 halves it, scratch included;
jit takes 2.0 x N^3 off the f64 peak and ~1 x N^3 off the f32 one. The small-grid f64
eager range is two runs of the same measurement.

### Host

**Measured** peak RSS of one seven-bin realization: 11.2-12.3 GB on the CPU node
(`configs/v28_halofit.yaml`, read exactly from `getrusage` of the generator process).
Beyond the field arrays the sampler holds the intensity and a float64 copy of the three
displacement components (3 x N^3 f64) while it draws. The ensemble script reserves
40 GB for two concurrent processes.

### What fits on which card

| card (allocator limit) | f64 eager | f32 eager | f64 jit | f32 jit |
|---|---|---|---|---|
| GH200 (71.25 GiB) | **measured:** every v28 grid, >5x headroom at 512^3 | measured: all | measured: all | measured: all |
| RTX 3050 (4.25 GiB) | **measured:** 192^3, 256^3 fit; 384^3 (4.75 GiB) does not | *predicted:* up to 448^3 (4.03 GiB, 5% margin) | *predicted:* up to 256^3; 384^3 (4.23 GiB) too close to call | *predicted:* up to 448^3 (3.40 GiB) |
| RTX 3050 (5.38 GiB at fraction 0.95) | **measured:** up to 384^3; 448^3 and 512^3 run out of memory | *predicted:* up to 448^3 | *predicted:* up to 384^3 | *predicted:* all, 512^3 at 5.07 GiB (6% margin) |
| GH200 at 1024^3 | *predicted:* ~96 GiB, does not fit | *predicted:* ~48 GiB, fits | | |

Predictions use the GH200 factors above. The scratch term is card-dependent (the RTX
3050 measured 11.27 x N^3 f64 at 384^3 where the GH200 measures 12.04), so a
prediction within ~10% of a limit needs a `ladder` run on the card. `dtype` and `jit`
are not `RunConfig` fields, so the shipped driver always runs f64 eager; on a card
smaller than the 512^3 bins need, run the field stage on the CPU.

## dtype and jit

Both are arguments of `field.generate_fields` and of `scripts/device.py`; neither is
reachable from a run config.

**`dtype="f32"`** sets the precision of the device arrays (`field.resolve_dtype`).

- The white noise is still drawn in float64 and cast, so the f32 field is the f64
  field at float32 precision, cell by cell, not a different realization: they agree to
  2-22 float32 eps of the field's maximum (bar 100 eps in
  `tests/test_field.py::test_f32_computes_the_same_field_as_f64`).
- `Fields.psi_flat` still returns float64, so the sampler is unchanged.
- Live arrays halve exactly (3.51 vs 7.02 x N^3 f64); the allocator peak halves too
  (table above).
- Every catalog changes at float32 round-off, so f32 and f64 catalogs from the same
  seed are not interchangeable bit for bit.

**`jit=True`** compiles the stage's two pure device functions, `coloured_lognormal`
and `displacement`. The host pieces (white noise, the spectrum gather, the value
syncs `grid_pkG` needs for its diagnostics and its `xi <= -1` check) cannot go inside
a jit, so jit reaches only the device part of the stage.

- Not bit-preserving: it moves a third to most of the cells by 0.25-4 float64 eps of
  the field's maximum (bar 20 eps,
  `tests/test_field.py::test_jit_is_not_bit_preserving_but_is_round_off`).
- It does not change the live-array peak; it lowers the allocator peak (10.02 vs
  12.03 x N^3 f64 at 512^3).
- The k grid is not baked into the executable as a constant (at 512^3 the compiled
  displacement reports no generated-code memory against 1.0 GiB of arguments).

## Ensemble production on one CPU node

`scripts/ensemble.sbatch` runs two concurrent streams of realizations on one node,
each process pinned to 16 cores (`taskset -c 0-15` and `-c 16-31`):

```sh
ENSEMBLE_OUT=/path/to/out sbatch scripts/ensemble.sbatch 0:1 50:51     # pilot
ENSEMBLE_OUT=/path/to/out sbatch scripts/ensemble.sbatch 0:50 50:100   # 100
```

Set the partition and account lines for your cluster; `ENSEMBLE_CONFIG` overrides the
default `configs/v28_halofit.yaml`. Each realization is its own process:
`logunusual run` -> `logunusual check` -> sha256 appended to
`$ENSEMBLE_OUT/sha256_manifest.txt`. An existing catalog is skipped (the file is
written under a temporary name and renamed when complete), a failed realization is
logged under `runs/ensemble/` and the stream moves on, and three failures in a row
stop a stream, so resubmitting the same command fills gaps. Peak RSS of each
generator process is read exactly with `getrusage` and logged.
`scripts/ensemble_check.py` then reads the ensemble out (`docs/validation.md`).

**Measured,** 100 realizations of `configs/v28_halofit.yaml` on the CPU node:

| | value |
|---|---|
| per realization, process wall | 101-138 s per stream, two streams at once |
| per realization, `logunusual check` | ~1 s |
| per realization, sha256 read-back | 30-104 s (storage-dependent) |
| peak RSS per process | 11.2-12.3 GB |
| output | 15.04 GiB per realization (~16.2 GB); 1.5 TB for 100 |
| 100 realizations | 2 h 20 min |

To size a different run: ~110 s and ~12 GB per realization per 16-core process at the
v28 densities, and ~15 GiB of output per realization; count storage write and
read-back bandwidth, since two streams write ~15 GiB every ~2 minutes and the sha256
pass reads every catalog back.

## Measuring on your own machine

`scripts/device.py` measures the field stage at the v28 production grids. On a CUDA
machine run it as `pixi run -e gpu python scripts/device.py ...`; plain
`pixi run python` runs on the CPU.

```sh
pixi run -e gpu python scripts/device.py ladder [--bins bin01 ... bin05] \
    [--dtype f64 f32] [--jit] [--out FILE]
pixi run -e gpu python scripts/device.py ulp [--n 64 128 256] [--L 5000]
pixi run -e gpu python scripts/device.py wall [--bin bin05] [--jit]
pixi run -e gpu python scripts/device.py pkg  [--bin bin05] [--repeat 3]
```

| mode | reports | notes |
|---|---|---|
| `ladder` | live-array peak, device allocator peak, host `ru_maxrss`, per bin | one subprocess per point (JAX keeps no resettable peak); an out-of-memory point is reported as a result. bin05 stands for bins 6-7 (same grid). The allocator column exists only on a GPU |
| `ulp` | CPU vs CUDA per-cell ulp distance and max difference, all five fields | needs a CUDA device; one process, one jaxlib |
| `wall` | per-step time (blocked) and end-to-end time (unblocked) for one bin | blocking at each step serialises JAX's async dispatch, so the per-step numbers are upper bounds; a 32^3 warm-up runs first |
| `pkg` | the `P -> P_G` conversion split into host, upload, device and sync pieces, median of `--repeat` runs | checks the assembled result bitwise against the shipped call before reporting |

The scripts take the bin's table from `data/` when present and otherwise fall back to
the z = 0.9 table in `tests/data/` (a footprint depends on the grid, not the spectrum).
For a whole realization, `logunusual run` prints field and sample time per bin and
writes them (`t_field_s`, `t_sample_s`) with `wall_s` to `summary.json`; measure peak
RSS with `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux). On a shared or busy
machine, timings are not comparable run to run.
