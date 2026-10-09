# Construction

How one bin of one realization is built, step by step, with the equations and the
properties each step guarantees. The code is in `logunusual/`: `grid.py`, `pk.py`,
`field.py`, `sample.py`, `shell.py`, `fnl.py`, `run.py`. The checks that hold each step
to these statements are in [validation.md](validation.md).

## Contents

- [Notation](#notation)
- [Pipeline](#pipeline)
- [Target grid spectrum](#1-target-grid-spectrum)
- [Grid-native lognormal transform](#2-grid-native-lognormal-transform)
- [Gaussian fields](#3-gaussian-fields)
- [Lognormal fields](#4-lognormal-fields)
- [Displacements](#5-displacements)
- [Sampling](#6-sampling)
- [Seeds](#seeds)
- [Shell product](#shell-product)
- [Local f_NL](#local-f_nl)
- [Nonlinear galaxy target](#nonlinear-galaxy-target)
- [Attainability](#attainability)
- [Limits of linear-theory RSD](#limits-of-linear-theory-rsd)
- [Estimator response](#measuring-the-catalogs-estimator-response)
- [Other properties](#other-properties)

## Notation

| symbol | meaning |
|---|---|
| `N`, `L` | grid points per side, box side (Mpc/h) |
| `dx = L / N`, `V_cell = dx^3` | cell size and volume |
| `k_f = 2 pi / L`, `k_Nyq = pi / dx` | fundamental and Nyquist wavenumbers |
| `sinc(x) = sin(pi x) / (pi x)` | numpy convention; windows take `k_i dx / 2 pi` (cycles per cell) |
| `W_p(k) = prod_i sinc(k_i dx / 2 pi)^(2p)` | power window of order-`p` in-cell jitter |
| `P(k)` | linear input table of the bin (`pk_file`) |
| `P_gal(k)` | galaxy-target table (`pk_galaxy_file`; equals `P` when not given) |
| `b`, `f`, `nbar` | bin bias, linear growth rate, target density |

Units: lengths in Mpc/h, k in h/Mpc, P in (Mpc/h)^3. Real FFTs compress the last
axis (z). Fourier normalisation: `P = (V / N^6) |delta_k|^2` with `delta_k =
rfftn(delta)`. In the periodic box, cell centres are at `(i + 0.5) dx`.

## Pipeline

Per bin, in `run.generate_realization`:

1. Galaxy target `P_g,grid = b^2 P_gal / W_1` and matter target `P_m,grid = P / W_1`
   on the rfft grid.
2. Each target is transformed to the spectrum `P_G` of a Gaussian field on the same
   grid.
3. One white-noise field is coloured by both `P_G`s.
4. Both Gaussian fields are exponentiated and normalised to lognormal fields
   `delta_g`, `delta_m`.
5. The displacement field `Psi` is computed from `delta_m` by linear continuity.
6. Galaxies are Poisson-sampled from `delta_g` in the cells of the buffered shell,
   placed uniformly within their cells, shifted radially by `f Psi` of their own
   cell, and kept inside the shell and the mask.

Steps 1-5 run in JAX (`field.generate_fields`, eager, float64); step 6 runs in numpy
on a thread pool (`shell.sample_shell`).

## 1. Target grid spectrum

```
P_g,grid(k) = b(k)^2 P_gal(|k|) / W_p(k)        (galaxies)
P_m,grid(k) =        P(|k|)     / W_p(k)        (matter)
```

with DC set to 0, `p = jitter_p` (default 1) and `b(k) = b` unless f_NL is set. The
bias enters before the transform (lognormal bias). Placing a galaxy uniformly within
its cell multiplies the catalog's power by `W_1`, so dividing the target by `W_1`
makes the catalog's continuum power exactly `b^2 P_gal` in the first Brillouin zone.
The matter target carries the same division, so the displacement a galaxy inherits
from its cell is exact in the first zone too.

The table is evaluated once per integer radius `q = i^2 + j^2 + l^2 <= 3 (N/2)^2`
at `|k| = k_f sqrt(q)` and gathered onto the grid (`pk.pk_on_grid`).

**Why the target is deconvolved, and why `p = 1`.** Deconvolving the lognormal field
after exponentiation would leave `1 + delta < 0` in many cells: on a 128^3, 1000
Mpc/h box with `b = 1.76` at z = 0.9, a third of the cells, holding 27% of the
expected galaxies, with minima near -50. Deconvolving the target before the transform
is exact by the grid identity of step 2 and keeps `1 + delta > 0` everywhere. With
triangular jitter (`p = 2`) the target needs `P / sinc^4`, whose `xi` reaches -1.22
at the neighbour lag on the same box; no lognormal reaches that (`log1p` is
undefined) and `pk.grid_pkG` raises. Uniform placement (`p = 1`) is attainable on the
default grids (see [Attainability](#attainability)).

## 2. Grid-native lognormal transform

On the simulation grid itself (`pk.grid_pkG`):

```
xi(x)   = irfftn(P_grid(k)) / V_cell
xi_G(x) = log1p(xi(x))
P_G(k)  = rfftn(xi_G(x)) * V_cell,   P_G(0) = 0
```

For a periodic Gaussian field `G` with grid spectrum `P_G`, the field
`exp(G) / <exp G> - 1` has grid two-point function exactly `exp(xi_G) - 1 = xi` in
the ensemble, so its power equals `P_grid` on every grid mode up to the Nyquist. No
continuum Hankel transform and no separate anti-aliasing step are involved.

Two failure modes are handled explicitly:

- `xi <= -1` anywhere: no lognormal has this spectrum; `grid_pkG` raises.
- `P_G < 0` on some modes: the target is not attainable there. Negative modes are
  set to 0 and **reported**: the number of clipped modes, the clipped power over the
  kept power (`clipped_power_fraction`, written per bin to the catalog metadata),
  `xi_min` and `sigma2 = xi(0)` (the cell variance). A Gaussian run with a single
  table proceeds with clipping reported; a run with f_NL or with a galaxy table raises
  on any clipped galaxy mode (see below).

## 3. Gaussian fields

```
w     ~ unit white noise on the N^3 grid    (numpy PCG64, default_rng(ic_seed), float64)
G_k   = rfftn(w) sqrt(P_G(k) / V_cell)
G     = irfftn(G_k)
```

The **same** `w` colours the galaxy and the matter field, so the two are as
correlated as their spectra allow. With this normalisation, `(V / N^6) |G_k|^2` has
expectation `P_G`.

## 4. Lognormal fields

```
1 + delta = exp(G) / mean_box(exp G)
```

for both fields, with the mean over all cells of the box. `1 + delta > 0` everywhere
by construction, so nothing downstream clips.

## 5. Displacements

Linear continuity on the lognormal **matter** field:

```
Psi_k = i k / k^2 delta_m,k,      Psi_0 = 0      (Mpc/h; all three components)
```

The growth rate multiplies `Psi` at sampling time, and every galaxy takes the
displacement of its **own** cell (no interpolation stencil), which keeps the RSD
prediction free of an extra window.

The lognormal matter field is the velocity source because it correlates with the
galaxy field better than any Gaussian source can. For a Gaussian field `L` jointly
Gaussian with `G_g`, `Cov(delta_g, L) = Cov(G_g, L)` (Gaussian integration by parts),
so by Cauchy-Schwarz the galaxy cross power of any Gaussian velocity source with the
right velocity power is at most `sqrt(P_G,g P_m)`. Because the Gaussian core `P_G,g`
falls below `b^2 P` at low k, that bound sits well below linear theory. Exact ensemble
values of `P(g, source) / (b P_m)`, shell-averaged, on the default grids with the
halofit galaxy target:

| bin | Gaussian source: k = 0.01 / 0.05 / 0.1 | lognormal matter (used): k = 0.01 / 0.05 / 0.1 |
|---|---|---|
| 1 | 0.938 / 0.891 / 0.801 | 0.983 / 0.966 / 0.954 |
| 2 | 0.934 / 0.882 / 0.783 | 0.987 / 0.974 / 0.967 |
| 3 | 0.941 / 0.892 / 0.801 | 0.988 / 0.976 / 0.967 |
| 4 | 0.940 / 0.897 / 0.809 | 0.989 / 0.979 / 0.971 |
| 5 | 0.942 / 0.900 / 0.814 | 0.988 / 0.977 / 0.966 |
| 6 | 0.932 / 0.885 / 0.790 | 0.981 / 0.968 / 0.947 |
| 7 | 0.921 / 0.867 / 0.753 | 0.968 / 0.945 / 0.896 |

**Displacement tail.** Continuity applied to a lognormal field turns each of its few
densest cells (`delta_m` of order 200-300 at 5-5.5 sigma of the Gaussian field) into
a near point source of displacement. Near such a cell `f |Psi|` reaches 100-180
Mpc/h, 20-30 times the rms; in bin 1 about 1e-4 of the galaxies are displaced by more
than 50 Mpc/h and about 2e-6 by more than 100. A Gaussian field with the same target
has the same displacement rms and a maximum near 20 Mpc/h. The radial buffer and its
guard ([below](#buffer-and-guard)) exist to handle this tail exactly.

## 6. Sampling

Per cell,

```
lambda = nbar V_cell (1 + delta_g),      n ~ Poisson(lambda)
```

and each galaxy is placed at `cell corner + (u_1 + ... + u_p - (p - 1)/2) dx` per
axis, `u` uniform in [0, 1); with `p = 1` that is uniform within the cell.

**RNG streams.** The unit of work is one x-slab (`i` fixed, `N^2` cells). Slab `i`
draws from its own counter-based stream, `Philox(key = draw_seed | i << 64)`
(`sample.slab_rng`): first the Poisson counts over the slab's cells, then the
placement uniforms, galaxy-major. The slab index sits in the 128-bit key, which
identifies the stream; the Philox counter is only a position within a stream, so
streams that differed in the counter alone would emit the same numbers shifted. A
catalog therefore depends on `(draw_seed, field)` alone: slabs run on a thread pool
(`n_workers`) and are written in slab order, and the catalog is bitwise identical for
any thread count.

## Seeds

```
seed            = seed_base + 1000 * realization + bin_index       (suite.seed_for)
ic_seed, draw_seed = SeedSequence(seed).spawn(2)                    (sample.split_seed)
```

`realization` must lie in [0, 1000) and `bin_index` in 1-127, so no two (realization,
bin) pairs share a seed. `ic_seed` drives the white noise, `draw_seed` the slab
streams: the field and the draw are independent, and a field can be resampled with a
new `draw_seed`. All three seeds are recorded per bin.

## Shell product

**Frame.** Each bin is its own periodic box centred on the observer: positions are
shifted by `-L/2` per axis, so cell centres sit at `(i + 0.5) dx - L/2` and the
observer is at the origin. All bins share the origin and axis orientation.

**Radial window.** The intensity is multiplied at cell level by the full-sky window

```
r_lo <= |x_centre| <= r_hi,     r_lo = max(0, rmin - buffer),  r_hi = rmax + buffer
```

The lognormal field itself stays periodic and unwindowed, so steps 1-4 are unchanged
by the geometry.

**Radial RSD.** A galaxy at `x` in cell `c` moves along its line of sight by the
projection of its cell's displacement:

```
s = x + f (Psi_c . x / |x|^2) x
```

(a galaxy exactly at the origin stays there). There is no periodic wrap after the
shift: a galaxy can only leave the box from a buffer cell touching a face, and it is
then outside `[rmin, rmax]` either way (counted as `n_left_box`).

**Cut.** A galaxy is kept iff `rmin <= |s| <= rmax` (both edges inclusive) and, with a
mask, the HEALPix pixel of `s` is set (`hp.vec2pix` with the file's ordering).
Because galaxies cross each edge in both directions, the radial density profile is
flat through both edges; for a uniform field the kept count is exactly
`Poisson(nbar fsky V_shell)`.

### Buffer and guard

`Shell.check_box` requires, at config time, `rmax + buffer <= L/2` and `buffer >=
sqrt(3) dx`. At field time, before any galaxy is drawn, `Shell.required_buffer`
computes the smallest buffer that draws every cell able to feed the shell, for this
realization's displacement field, and `sample_shell` raises if the configured buffer
is smaller.

The test is the signed line-of-sight shift (`shell.reach_interval`). For `x` in cell
`c` with centre distance `r_c` and half-diagonal `h = sqrt(3)/2 dx`:

- `|x|` lies in `[max(r_c - h, 0), r_c + h]`;
- `x_hat` lies within the chord `chi = 2 sin(theta / 2)`, `sin theta = h / r_c`, of
  the centre direction (`chi = 2` when `r_c <= h`);
- so the shift `f Psi_c . x_hat` lies in `[a - m chi, a + m chi]` clipped to
  `[-m, m]`, with `a = f Psi_c . c_hat` and `m = |f| |Psi_c|`.

`rho = |x| + f Psi_c . x_hat` therefore lies in an interval, `|s| = |rho|` in the
corresponding range, and a cell can feed the shell iff that range meets
`[rmin, rmax]`. The requirement is the largest distance from the shell of any such
cell, over the whole box. An outward displacement beyond the shell, a tangential one,
or an inward one that overshoots the observer does not count; a large displacement
of a cell already inside the drawn window never counts. The requirement
(`required_buffer`) and the window's largest `|Psi|` (`psi_max`) are written per bin.

**Why the default buffer is 160 Mpc/h.** Over 100 realizations of
`configs/v28_halofit.yaml`, the largest line-of-sight requirement is 150.9 Mpc/h (bin
1); the largest requirement as a fraction of 160 Mpc/h per bin is 0.943, 0.771,
0.705, 0.653, 0.560, 0.315, 0.205 for bins 1-7. The tail is set by single extreme
cells, so a realization can need more; the guard then raises. The default boxes allow
a buffer up to `min(L/2 - rmax) = 172.2` Mpc/h (bin 2; bin 1 allows 181.8).

### Angular pre-cut

With a mask, `angular_precut` (default on) drops the cells whose galaxies cannot land
in a set pixel. Radial RSD keeps a galaxy's direction, so a galaxy of cell `c` stays
within `cell_angular_radius(r_c) = asin(min(1, h / r_c))` of the centre direction,
and its pixel centre within a further pixel radius. A cell is dropped iff the nearest
set-pixel centre is farther than `cell_angular_radius(r_c) + 2 max_pixrad` from the
centre of the pixel containing the cell centre (`AngularMask.distance_to_set`, one
KD-tree query per mask). The kept cells are an exact superset of those that feed the
masked shell. The pre-cut changes which cells consume random numbers, so it changes
the catalog and is part of the config hash.

## Local f_NL

Local primordial non-Gaussianity enters as a scale-dependent galaxy bias (two-point
only; a lognormal carries no f_NL bispectrum):

```
b(k) = b + 2 (b - p) f_NL delta_c / M(k),        delta_c = 1.686
M(k, z) = (2/3) (c/H0)^2 k^2 T(k) D(z) / Omega_m      (delta_m = M Phi)
```

in the **LSS convention**, `D(0) = 1`: `f_NL^LSS = f_NL^CMB / g0` with `g0 = D_md(0)`,
the growth today normalised to `a` in matter domination (0.788 for `Omega_m =
0.3153`, from the flat-LCDM growth integral, `fnl.growth_md`).

`M(k)` is not built from a transfer function. With `Phi = (3/5) zeta` in matter
domination the bin's own linear table is `P(k, z) = M_CMB(k, z)^2 P_Phi(k)`, with

```
P_Phi(k) = (9/25) 2 pi^2 A_s k^-3 (k / k_pivot)^(n_s - 1)
```

so `M_CMB = sqrt(P / P_Phi)` exactly, z enters through the table, and `M = M_CMB /
g0`. The `A_s`, `n_s`, `k_pivot` and `omega_m` in `primordial` must be the ones the
tables were made with; the defaults (2.1e-9, 0.9649, 0.05/Mpc = 0.07423 h/Mpc, 0.3153)
are those of the CAMB tables from `scripts/make_pk_tables.py`. `M` always comes from
the linear `pk_file`, also when a galaxy table is given.

Only the galaxy target changes (`b^2 P_gal -> b(k)^2 P_gal`); the matter field and the
velocities do not. With `f_nl = 0` the stage is bitwise the Gaussian one. Per bin the
metadata records `b(k_f) / b`, `delta_b` at `k_f`, and the lowest grid `|k|` at which
`b(k)` changes sign (`fnl_k_zero`).

With `f_nl != 0`, any clipped galaxy `P_G` mode raises before the bin is sampled.
Where `b(k)` falls toward zero at low k (`f_NL < 0`), the higher orders of
`log(1 + xi)` add a white term that floors the realizable low-k power at these cell
variances, so `P_G` goes negative on exactly the shells that carry the f_NL signal.
The clipped field's power there would be 1.4x to over 300x the target, while the
clipped power fraction (1e-8 to 2e-5) hides it.

## Nonlinear galaxy target

A bin's `pk_galaxy_file` (e.g. Takahashi halofit from CAMB, including the Bird et al.
massive-neutrino terms; CAMB leaves `kh < 0.005` linear) replaces the table in the
galaxy target only: `b(k)^2 P_gal / W_1`. `pk_file` stays linear and drives the
matter field, the velocities (continuity is a linear relation) and `M(k)`
(`P = M^2 P_Phi` holds in linear theory). With distinct tables the matter field and
displacements are bitwise those of the linear run.

**Pair check** (`pk.check_table_pair`, at load). The two tables must start at the
same k node `k0`, and there

```
|P_gal(k0) / P(k0) - 1| <= k0^2 sigma_v^2,     sigma_v^2 = (1 / 6 pi^2) int P dk
```

the leading low-k nonlinear correction (`P_13 -> -k^2 sigma_v^2 P`), which any
nonlinear model of the same linear spectrum satisfies. Tables at another redshift or
cosmology are off at order unity (17-35% between adjacent default bins). The halofit
and kh-10 linear tables from `pk-tables make` agree exactly at `k0 = 1e-4`. The
kh-1 linear tables (the default bin table's `pk_file`) have the same nodes up to
kh = 1 but differ from the kh-10 ones by up to 1.2e-5 in P, more than the bound: pair
a halofit table with the kh-10 linear one.

A run with a galaxy table raises on any clipped galaxy `P_G` mode: the excess power
of a clipped field spreads to every k, not only the clipped shells. Bin 7 keeps the
linear galaxy target in `configs/v28_halofit.yaml` for this reason.

## Attainability

Whether a lognormal reaches the target on a given grid is a property of the target
and the grid, independent of the seed; `scripts/attainability.py attain` measures it
per bin (`xi_min`, `sigma2`, clipped modes and power fraction, `b(k_f) / b`, the sign
change of `b(k)`). On the default grids:

- **Linear targets, f_NL = 0.** Bins 1-6 clip no mode (`xi_min > -0.004`). Bin 7
  (`b = 3.29`, `dx = 15.6` Mpc/h) clips 5 modes, its fundamental shell, carrying 1e-8
  of the power; there the realized ensemble power is 1.109x the target, and every
  other shell is exact. The run proceeds and records `clipped_power_fraction`. Grid
  cell variances: galaxy 2.3-3.3, matter 0.26-2.8.
- **Halofit galaxy targets.** Attainable in bins 1-6 (galaxy cell variance 2.5-5.2);
  bin 7 clips the same 5 fundamental modes with either table. Halofit is 1.06x (bin
  7) to 1.89x (bin 1) the linear power at the grid Nyquist.
- **f_NL** (linear target). Clipped galaxy `P_G` modes per bin; any nonzero entry
  raises:

| bin | N, L | -100 | -10 | -1 | +1 | +10 | +100 |
|---|---|---|---|---|---|---|---|
| 1 | 192, 1500 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 256, 2500 | 5 | 0 | 0 | 0 | 0 | 0 |
| 3 | 384, 3500 | 17 | 0 | 0 | 0 | 0 | 0 |
| 4 | 448, 4500 | 41 | 5 | 0 | 0 | 0 | 0 |
| 5 | 512, 5000 | 85 | 5 | 0 | 0 | 0 | 0 |
| 6 | 512, 7000 | 486 | 33 | 5 | 0 | 0 | 0 |
| 7 | 512, 8000 | 1540 | 94 | 13 | 0 | 0 | 0 |

Positive f_NL up to 100 runs on every bin. Negative f_NL raises on the lowest shells
of bins 2-7, from -1 in bins 6-7, from -10 in bins 4-5 and at -100 in bins 2-3.

Finer grids raise the cell variance: at twice the default `N` a lognormal no longer
reaches the halofit target in bins 1-3 (galaxy cell variance 15-20), while the linear
targets clip nowhere.

## Limits of linear-theory RSD

Linear-theory (Kaiser) RSD is a `k -> 0` limit, and the lognormal pair is not
perfectly correlated at finite k. On a 128^3, 1000 Mpc/h box with bin 5's bias and
growth rate (`f Psi_rms = 3.2` Mpc/h, galaxy cell variance 3.9), `P(g, m) / (b P_m)`
is 0.99 at k = 0.035 and 0.965 at k = 0.19 h/Mpc, and the measured `P_2 / P_0`
exceeds the Kaiser ratio by about 25% at k = 0.19, where `k f Psi_rms = 0.6` and the
mapping is no longer linear. At production amplitude the Kaiser ratio is therefore a
measurement, not a prediction; Kaiser holds exactly only in the low-amplitude limit
(the scaled-amplitude check in [validation.md](validation.md)). The galaxy-velocity
cross power on the default grids is tabulated under [Displacements](#5-displacements).

## Measuring the catalogs: estimator response

A catalog drawn from a grid field is lattice-periodic, so on a finite estimator mesh
the alias images carry the **same** Fourier coefficient up to a sign and add
coherently (`validate.effective_window`):

```
P_mesh(k) = |sum_n s_n W_cic(k_n) T(k_n)|^2 P_grid(k),     k_n = k + 2 pi n / H
```

per axis (separable), with `H` the estimator spacing, `W_cic = prod sinc^2`,
`T = prod sinc^p` the jitter window in generator-cell units, and
`s_n = (-1)^(r n)` from the half-cell offset of the cell centres, `r` the integer
mesh ratio. The standard incoherent alias sum (Jing 2005) applies to the shot noise
only. `validate.estimator_response` is the ratio of the CIC-deconvolved mesh power to
the catalog's true continuum power:

| estimator mesh | k = k_Nyq / 2 (shell average; along an axis) | k = k_Nyq (shell average) |
|---|---|---|
| 2x the generator (every sign +) | 0.998; 0.997 | 0.965 |
| the generator's own (dominant image subtracts) | 0.965; 0.939 | 0.358 |

Measure on a mesh at least twice the generator's, or divide by
`estimator_response`; the statistical checks do both.

## Other properties

- **Precision.** The field stage runs in float64 (`JAX_ENABLE_X64=1`); positions are
  written as float64. `field.generate_fields` also takes `dtype="f32"` and `jit=True`;
  neither is a run-config option, and both change the catalog bits.
- **Reproducibility.** The sampler is deterministic and thread-count independent by
  construction. The eager field stage is bitwise reproducible across processes on
  Linux; on macOS-arm64 it can differ in the last bit. XLA-CPU's reduction order
  follows its thread pool, so bitwise reproduction across machines needs the same CPU
  affinity; CPU and CUDA agree to about 1e-13 relative, not bitwise.
