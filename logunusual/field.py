"""Field stage (JAX, eager): white noise -> Gaussian fields -> lognormal galaxy and
matter fields -> line-of-sight displacement.

Runs on whatever device JAX has (CPU today, CUDA in the `gpu` env). The module
enables x64 on import and float64 is the default; `dtype="f32"` halves the device
footprint (see `resolve_dtype`). No `jax.jit` in M1 (one program, so the
reproducibility gate has one thing to characterise).

Construction (decided 2026-09-04 after measuring the alternatives, see CLAUDE.md):
- The TARGET grid spectrum is `P(k) / prod_i sinc(k_i dx/2pi)^(2p)` with `p` the
  intra-cell jitter order used by the sampler (default 1: uniform within the cell).
  By the grid identity (`pk.grid_pkG`) the lognormal grid field then has exactly that
  power in the ensemble, and the catalog -- cell intensities spread by the jitter
  kernel -- has continuum power exactly `P(k)` inside the first Brillouin zone. No
  post-transform deconvolution, so `1 + delta >= 0` everywhere by construction. (The
  post-transform `sinc^-2` correction clips ~1/3 of the cells and ~27% of the galaxies
  at bin-5 settings; deconvolving a p = 2 target gives `xi < -1`, i.e. no lognormal.)
- `G_k = rfftn(w) * sqrt(P_G(k) / V_cell)`, `w` unit white noise from numpy
  (`default_rng(ic_seed)`), so the estimator normalisation `V/N^6 |G_k|^2` returns P_G.
- `1 + delta = exp(G) / mean_box(exp G)` (box-mean normalisation), both fields.
- Displacement `Psi_k = i k / k^2 delta_m,k` (linearised continuity on the matter
  lognormal field, Agrawal et al. 2017), DC = 0, in Mpc/h; the growth rate multiplies
  it at sampling time. The matter target carries the same jitter deconvolution, so the
  displacement a galaxy inherits from its cell is exact in the first zone too. Any
  subset of the three components can be built (`psi_axes`): the plane-parallel box
  needs `z` only, the observer-centred shell product needs all three.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from dataclasses import dataclass, field  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from . import fnl as _fnl  # noqa: E402
from .grid import Box, jitter_power_window, k_components  # noqa: E402
from .pk import grid_pkG, pk_on_grid  # noqa: E402


def white_noise(box: Box, seed) -> np.ndarray:
    """Unit Gaussian white noise on the real grid, numpy PCG64, float64."""
    return np.random.default_rng(seed).standard_normal(box.shape, dtype=np.float64)


def target_on_grid(spectrum, box: Box, jitter_p: int) -> np.ndarray:
    """`P(|k|) / sinc^(2p)` on the rfft grid, DC = 0 (the grid target)."""
    P = pk_on_grid(spectrum, box)
    if jitter_p:
        P = P / jitter_power_window(box, jitter_p)
    return P


def colour(white_k, pkG_grid, box: Box):
    """`G = irfftn(white_k * sqrt(P_G / V_cell))`; `white_k = rfftn(w)`."""
    amp = jnp.sqrt(jnp.asarray(pkG_grid) / box.v_cell)
    return jnp.fft.irfftn(white_k * amp, s=box.shape, axes=(0, 1, 2))


def lognormal(G):
    """`exp(G) / mean(exp G) - 1`; the box mean is taken over all cells."""
    e = jnp.exp(G)
    return e / jnp.mean(e) - 1.0


AXES = "xyz"

DTYPES = {"f64": ("float64", "complex128"), "f32": ("float32", "complex64")}


def resolve_dtype(dtype):
    """`"f64"` / `"f32"` -> the `(real, complex)` JAX dtypes of the field stage.

    Only the DEVICE arrays take this dtype. The white noise is always drawn in
    float64 (a numpy generator consumes its stream differently per dtype, so drawing
    natively would change the realization rather than its precision) and cast on the
    way to the device, and `Fields.psi_flat` upcasts for the float64 sampler. The
    knob is not a `RunConfig` field: f32 changes every catalog, so it stays an API /
    instrument argument until it is a production choice that the config hash covers.
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    real, cplx = DTYPES[dtype]
    return jnp.dtype(real), jnp.dtype(cplx)


def displacement_k(delta_k, box: Box, axis: str):
    """`i k_axis / k^2 * delta_k`, DC = 0: one component of the continuity
    displacement (`axis` in "xyz")."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES!r}, got {axis!r}")
    # The k components come from numpy in float64, and under x64 a single float64
    # array in this expression widens the whole of it -- that, not the `1j`, is what
    # drags an f32 field stage back to complex128 (measured, jax 0.10.1). Matching
    # them to `delta_k` keeps `1j * comps / k2 * delta_k` at complex64.
    real_dt = jnp.float32 if delta_k.dtype == jnp.complex64 else jnp.float64
    comps = [jnp.asarray(a, dtype=real_dt) for a in k_components(box)]
    k2 = comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2
    k2 = k2.at[0, 0, 0].set(1.0)  # avoid 0/0; the DC value is zeroed below
    psi_k = 1j * comps[AXES.index(axis)] / k2 * delta_k
    return psi_k.at[0, 0, 0].set(0.0)


def displacement_z_k(delta_k, box: Box):
    """`displacement_k(..., "z")` (the M1 plane-parallel line of sight)."""
    return displacement_k(delta_k, box, "z")


# The two pure device functions of the stage, in one place so `jax.jit` has something
# to compile. Everything else in `generate_fields` either runs on the host (the
# spectrum spline) or needs a value back from the device (`grid_pkG`'s diagnostics),
# and neither can live inside a jit. `box` and `axis` are static: `Box` is a frozen
# dataclass, so it hashes, and the shapes have to be concrete for the FFTs anyway.


def coloured_lognormal(white_k, pkG_grid, box: Box):
    """`lognormal(colour(...))` -- the galaxy or matter field from its P_G."""
    return lognormal(colour(white_k, pkG_grid, box))


def displacement(delta_k, box: Box, axis: str):
    """One real-space displacement component, `irfftn(displacement_k(...))`."""
    return jnp.fft.irfftn(
        displacement_k(delta_k, box, axis), s=box.shape, axes=(0, 1, 2)
    )


coloured_lognormal_jit = jax.jit(coloured_lognormal, static_argnums=(2,))
displacement_jit = jax.jit(displacement, static_argnums=(1, 2))


@dataclass
class Fields:
    box: Box
    ic_seed: int
    jitter_p: int
    delta_g: jax.Array  # galaxy lognormal grid field (>= -1 everywhere)
    # displacement components in Mpc/h (growth rate NOT applied), keyed "x"/"y"/"z";
    # only the requested axes are present
    psi: dict = field(default_factory=dict)
    delta_m: jax.Array | None = None  # matter lognormal grid field
    diagnostics: dict = field(default_factory=dict)
    dtype: str = "f64"  # the device dtype the stage ran in ("f64" / "f32")
    jit: bool = False  # whether the pure device functions were compiled

    @property
    def psi_z(self):
        return self.psi.get("z")

    def psi_flat(self, axes=AXES):
        """`(n_cells, len(axes))` float64 numpy array of the displacement per cell,
        C order, for the samplers; raises if a requested axis was not built."""
        missing = [a for a in axes if a not in self.psi]
        if missing:
            raise ValueError(f"fields carry no displacement along {missing}")
        return np.stack(
            [np.asarray(self.psi[a], dtype=np.float64).reshape(-1) for a in axes],
            axis=1,
        )


def generate_fields(
    spectrum,
    b: float,
    box: Box,
    ic_seed: int,
    *,
    jitter_p: int = 1,
    rsd: bool = True,
    keep_matter: bool = False,
    psi_axes: str = "z",
    dtype: str = "f64",
    jit: bool = False,
    fnl=None,
    galaxy_table=None,
    trace=None,
) -> Fields:
    """Galaxy field with target `b^2 P` and (if `rsd`) the displacement components
    `psi_axes` (a subset of "xyz") from the matter field with target `P`; both
    coloured from the SAME white noise, both targets deconvolved by the
    order-`jitter_p` placement window. `spectrum` is the linear P(k);
    `galaxy_table` (a nonlinear P(k), None = `spectrum`) replaces it in the galaxy
    target only, so the matter field and displacements are those of the linear run
    bitwise; with it, a galaxy P_G that clips any mode raises (at small cells halofit
    is beyond a lognormal, and the excess spreads to every k). `fnl` (an
    `fnl.LocalPNG`) makes the galaxy
    target `b(k)^2 P` and leaves the matter field alone; None or f_NL = 0 is the
    scalar-bias stage bitwise. With f_NL != 0 a galaxy P_G that clips any mode
    raises: where b(k)^2 P falls toward zero at low k (f_NL < 0) the lognormal cannot
    reach it, and the clipped field's power sits far above the target on exactly the
    shells that carry the f_NL signal. Gaussian runs report clipping, as before.
    `dtype` (`"f64"` / `"f32"`) is the precision of the DEVICE arrays, see
    `resolve_dtype`. `jit` compiles the stage's two pure device
    functions (`coloured_lognormal`, `displacement`); it is off by default because it
    is not bit-preserving. `trace(label)` is called after each array step (memory
    instrumentation)."""
    trace = trace or (lambda label: None)
    real_dt, _ = resolve_dtype(dtype)
    _lognormal_of = coloured_lognormal_jit if jit else coloured_lognormal
    _psi_of = displacement_jit if jit else displacement
    if any(a not in AXES for a in psi_axes) or len(set(psi_axes)) != len(psi_axes):
        raise ValueError(f"psi_axes must be distinct letters of {AXES!r}: {psi_axes!r}")
    w = white_noise(box, ic_seed)
    trace("white_noise")
    white_k = jnp.fft.rfftn(jnp.asarray(w, dtype=real_dt))
    del w
    trace("white_k")

    pkG_g, diag_g = grid_pkG(
        jnp.asarray(
            target_on_grid(
                _fnl.galaxy_spectrum(spectrum, b, fnl, galaxy_table), box, jitter_p
            ),
            dtype=real_dt,
        ),
        box,
        jnp,
    )
    trace("pkG_g")
    if fnl is not None and fnl.f_nl != 0 and diag_g["n_clipped"]:
        d = _fnl.diagnostics(spectrum, b, fnl, box)
        raise ValueError(
            f"f_NL = {fnl.f_nl:g}: the galaxy target b(k)^2 P is not attainable by a "
            f"lognormal on this grid ({diag_g['n_clipped']} P_G modes < 0, "
            f"b(k_f)/b = {d['b_kf_over_b']:.3f}, b(k) changes sign at "
            f"k = {d['k_zero']}); clipping them would leave the realized power off "
            "the target on the lowest shells"
        )
    if galaxy_table is not None and diag_g["n_clipped"]:
        raise ValueError(
            f"the galaxy table's target is not attainable by a lognormal on this grid "
            f"({diag_g['n_clipped']} P_G modes < 0, sigma^2 = {diag_g['sigma2']:.2f}); "
            "the clipped field's power departs from the target at every k -- use a "
            "coarser grid"
        )
    delta_g = _lognormal_of(white_k, pkG_g, box)
    del pkG_g
    trace("delta_g")
    diag = {
        "galaxy": diag_g,
        "jitter_p": jitter_p,
        "dtype": dtype,
        "jit": jit,
        "delta_g_min": float(delta_g.min()),
        "delta_g_max": float(delta_g.max()),
    }
    if fnl is not None:
        diag["fnl"] = _fnl.diagnostics(spectrum, b, fnl, box)

    psi = {}
    delta_m = None
    if rsd:
        pkG_m, diag_m = grid_pkG(
            jnp.asarray(target_on_grid(spectrum, box, jitter_p), dtype=real_dt),
            box,
            jnp,
        )
        trace("pkG_m")
        delta_m = _lognormal_of(white_k, pkG_m, box)
        del pkG_m
        trace("delta_m")
        delta_m_k = jnp.fft.rfftn(delta_m)
        trace("delta_m_k")
        diag["matter"] = diag_m
        diag["psi_rms"] = {}
        for axis in psi_axes:
            psi[axis] = _psi_of(delta_m_k, box, axis)
            trace(f"psi_{axis}")
            diag["psi_rms"][axis] = float(jnp.sqrt(jnp.mean(psi[axis] ** 2)))
        del delta_m_k
        if "z" in psi:
            diag["psi_z_rms"] = diag["psi_rms"]["z"]
    del white_k
    trace("end")

    return Fields(
        box=box,
        ic_seed=int(ic_seed),
        jitter_p=int(jitter_p),
        delta_g=delta_g,
        psi=psi,
        delta_m=delta_m if keep_matter else None,
        diagnostics=diag,
        dtype=dtype,
        jit=jit,
    )
