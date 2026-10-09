"""Field stage (JAX, eager, x64 enabled on import): white noise -> Gaussian fields ->
lognormal galaxy and matter fields -> displacement. Entry point `generate_fields`;
construction in `docs/construction.md`.

The target `P(k) / prod_i sinc(k_i dx/2pi)^(2p)` (`p` the jitter order) is deconvolved
BEFORE the lognormal transform, so `1 + delta >= 0` and, by the grid identity
(`pk.grid_pkG`), the catalog's continuum power is `P(k)` in the first Brillouin zone.
`G_k = rfftn(w) sqrt(P_G / V_cell)` (so `V/N^6 |G_k|^2` estimates P_G); displacement
`Psi_k = i k / k^2 delta_m,k` on the matter lognormal field (Agrawal et al. 2017).
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

    Only the device arrays take this dtype: the white noise is always drawn in float64
    (numpy's stream depends on the dtype) and cast, and `Fields.psi_flat` upcasts for
    the sampler. f32 changes every catalog, so it is not a `RunConfig` field."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    real, cplx = DTYPES[dtype]
    return jnp.dtype(real), jnp.dtype(cplx)


def displacement_k(delta_k, box: Box, axis: str):
    """`i k_axis / k^2 * delta_k`, DC = 0: one component of the continuity
    displacement (`axis` in "xyz")."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES!r}, got {axis!r}")
    # Under x64 one float64 array widens the whole expression to complex128, so the
    # (numpy, float64) k components are cast to `delta_k`'s precision.
    real_dt = jnp.float32 if delta_k.dtype == jnp.complex64 else jnp.float64
    comps = [jnp.asarray(a, dtype=real_dt) for a in k_components(box)]
    k2 = comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2
    k2 = k2.at[0, 0, 0].set(1.0)  # avoid 0/0; the DC value is zeroed below
    psi_k = 1j * comps[AXES.index(axis)] / k2 * delta_k
    return psi_k.at[0, 0, 0].set(0.0)


def displacement_z_k(delta_k, box: Box):
    """`displacement_k(..., "z")`, the plane-parallel line of sight."""
    return displacement_k(delta_k, box, "z")


# The stage's two pure device functions, what `jit=True` compiles (the spectrum spline
# runs on the host and `grid_pkG` returns diagnostics). `box` and `axis` are static.


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
    # requested displacement components "x"/"y"/"z", Mpc/h, growth rate NOT applied
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
    """Galaxy field with target `b^2 P` and, if `rsd`, the displacement components
    `psi_axes` (a subset of "xyz") of the matter field with target `P`; both coloured
    from the SAME white noise, both targets deconvolved by the placement window.

    `spectrum` is the linear P(k). `galaxy_table` (nonlinear P(k), None = `spectrum`)
    replaces it in the galaxy target only; matter field and displacements stay bitwise
    the linear run's. `fnl` (an `fnl.LocalPNG`) makes the galaxy target `b(k)^2 P`;
    None or f_NL = 0 is the scalar-bias stage bitwise. With f_NL != 0 or a galaxy
    table, any clipped galaxy P_G mode raises (the target is not attainable: f_NL < 0
    on the lowest shells, halofit on fine cells); otherwise clipping is reported in
    `diagnostics`. `dtype`: device precision (`resolve_dtype`). `jit` compiles
    `coloured_lognormal` / `displacement` and is NOT bit-preserving. `trace(label)` is
    called after each array step (memory instrumentation)."""
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
