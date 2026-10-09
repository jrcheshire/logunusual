"""Local primordial non-Gaussianity as a scale-dependent galaxy bias (two-point only).

    b(k) = b + 2 (b - p) f_NL delta_c / M(k),
    M(k, z) = (2/3) (c/H0)^2 k^2 T(k) D(z) / Om        (delta_m = M Phi)

in the LSS convention, D(0) = 1: `f_NL^LSS = f_NL^CMB / g0`, with `g0 = D_md(0)` the
growth today in the normalisation `D_md -> a` in matter domination.

M is not built from a transfer function. With `Phi = (3/5) zeta` in matter domination
the bin's own linear table is `P(k, z) = M_CMB(k, z)^2 P_Phi(k)`, with
`P_Phi = (9/25) 2 pi^2 A_s k^-3 (k / k_pivot)^(n_s - 1)`, so `M_CMB = sqrt(P / P_Phi)`
exactly, z enters through the table, and `M = M_CMB / g0`. The `A_s`, `n_s` and pivot
must be the ones the table was made with (`LocalPNG` defaults: the v28 tables, `suite`).

Only the galaxy target changes (`b^2 P -> b(k)^2 P`); the matter field and the
velocities do not. A lognormal field carries no f_NL bispectrum.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import quad

from . import suite

C_OVER_H0 = 2997.92458  # c / H0 in Mpc/h

CONVENTION = "LSS"  # D(0) = 1; f_NL^LSS = f_NL^CMB / g0

_PHI_PER_ZETA_SQ = 9.0 / 25.0  # (Phi / zeta)^2 in matter domination


def _E(x, omega_m):
    """H(a) / H0 of flat LCDM without radiation."""
    return np.sqrt(omega_m / x**3 + 1.0 - omega_m)


def _heath_integral(a: float, omega_m: float) -> float:
    """`int_0^a da' / (a' E(a'))^3`."""
    integrand = lambda x: 1.0 / (x * _E(x, omega_m)) ** 3  # noqa: E731
    return quad(integrand, 0.0, a, epsabs=0.0, epsrel=1e-12)[0]


def growth_md(z: float, omega_m: float) -> float:
    """Linear growth of flat LCDM (no radiation), normalised to `a` in matter
    domination: `D = (5/2) Om E(a) int_0^a da' / (a' E(a'))^3` (Heath 1977)."""
    a = 1.0 / (1.0 + z)
    return 2.5 * omega_m * _E(a, omega_m) * _heath_integral(a, omega_m)


def growth_rate_md(z: float, omega_m: float) -> float:
    """Linear growth rate `f = dln D / dln a` of `growth_md`'s background, exactly:
    `f = -3/2 Om(a) + 1 / (a^2 E(a)^3 I(a))`, with `I` the Heath integral. The
    source of `suite`'s `Bin.f` literals (at `suite.OMEGA_M_DISTANCE`)."""
    a = 1.0 / (1.0 + z)
    E = _E(a, omega_m)
    omega_m_a = omega_m / (a**3 * E**2)
    return -1.5 * omega_m_a + 1.0 / (a**2 * E**3 * _heath_integral(a, omega_m))


@dataclass(frozen=True)
class LocalPNG:
    """Local-type f_NL (LSS convention) and the primordial normalisation of the input
    P(k) tables. `k_pivot` in h/Mpc; `omega_m` sets `g0` only."""

    f_nl: float
    p: float = 1.0
    delta_c: float = suite.DELTA_C
    A_s: float = suite.PRIMORDIAL_AS
    n_s: float = suite.PRIMORDIAL_NS
    k_pivot: float = suite.PRIMORDIAL_K_PIVOT
    omega_m: float = suite.OMEGA_M_DISTANCE

    def __post_init__(self):
        if not (self.delta_c > 0 and self.A_s > 0 and self.k_pivot > 0):
            raise ValueError("delta_c, A_s and k_pivot must be positive")
        if not 0.0 < self.omega_m <= 1.0:
            raise ValueError(f"omega_m must be in (0, 1], got {self.omega_m}")
        object.__setattr__(self, "g0", growth_md(0.0, self.omega_m))

    def as_dict(self) -> dict:
        return asdict(self)


def phi_power(k, png: LocalPNG):
    """Primordial potential spectrum `P_Phi(k)` in (Mpc/h)^3, k > 0 in h/Mpc."""
    k = np.asarray(k, dtype=np.float64)
    return (
        _PHI_PER_ZETA_SQ
        * 2.0
        * np.pi**2
        * png.A_s
        * k**-3.0
        * (k / png.k_pivot) ** (png.n_s - 1.0)
    )


def poisson_M(k, spectrum, png: LocalPNG):
    """`M(k)` (LSS convention) from the linear table `spectrum`; 0 at k = 0."""
    k = np.asarray(k, dtype=np.float64)
    out = np.zeros_like(k)
    pos = k > 0
    out[pos] = np.sqrt(spectrum(k[pos]) / phi_power(k[pos], png)) / png.g0
    return out


def delta_b(k, b: float, spectrum, png: LocalPNG):
    """`2 (b - p) f_NL delta_c / M(k)`; 0 at k = 0 (the DC mode carries no power)."""
    k = np.asarray(k, dtype=np.float64)
    out = np.zeros_like(k)
    pos = k > 0
    amp = 2.0 * (b - png.p) * png.f_nl * png.delta_c
    out[pos] = amp / poisson_M(k[pos], spectrum, png)
    return out


def galaxy_spectrum(spectrum, b: float, png: LocalPNG | None = None, galaxy_table=None):
    """The galaxy target `b(k)^2 P_gal(k)` as a callable of k. `spectrum` is the
    linear table and sets M(k); `galaxy_table` (e.g. halofit) is `P_gal`, None =
    `spectrum`. With `png` None or `f_nl == 0` it is exactly `b * b * P_gal(k)`, the
    f_NL-free expression, so that path is bitwise unchanged."""
    target = spectrum if galaxy_table is None else galaxy_table
    if png is None or png.f_nl == 0:
        return lambda k: b * b * target(k)

    def P_g(k):
        k = np.asarray(k, dtype=np.float64)
        return (b + delta_b(k, b, spectrum, png)) ** 2 * target(k)

    return P_g


def diagnostics(spectrum, b: float, png: LocalPNG, box) -> dict:
    """What the scale-dependent bias does on `box`: `delta_b` and `b(k)/b` at the
    fundamental, and the lowest grid |k| at or above which `b(k)` has changed sign
    (`k_zero`, None if it never does on the grid)."""
    n2 = box.n_mesh // 2
    kq = box.k_f * np.sqrt(np.arange(1, 3 * n2 * n2 + 1, dtype=np.float64))
    bk = b + delta_b(kq, b, spectrum, png)
    flip = np.nonzero(np.sign(bk[1:]) != np.sign(bk[:-1]))[0]
    db_kf = float(delta_b(np.array([box.k_f]), b, spectrum, png)[0])
    return {
        "f_nl": float(png.f_nl),
        "delta_b_kf": db_kf,
        "b_kf_over_b": (b + db_kf) / b,
        "k_zero": float(kq[flip[0] + 1]) if flip.size else None,
    }
