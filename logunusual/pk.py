"""Input power spectrum and the grid-native lognormal transform P -> P_G.

The transform runs on the simulation grid itself:

    xi(x)   = irfftn(P(k)) / V_cell          (P(0) := 0)
    xi_G(x) = log1p(xi(x))
    P_G(k)  = rfftn(xi_G(x)) * V_cell,       P_G(0) := 0,  P_G < 0 -> 0 (reported)

For a periodic Gaussian field with grid spectrum P_G, `exp(G)/<exp G> - 1` has grid
two-point function exactly `exp(xi_G) - 1 = xi` in the ensemble (Coles & Jones 1991),
so its power is `P` on every grid mode. Negative P_G means the target is not attainable
by a lognormal (Xavier et al. 2016).
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .grid import Box


def load_pk_tsv(path):
    """`# kh\\tPk` two-column table (k in h/Mpc, P in (Mpc/h)^3), strictly increasing
    k, positive P. Returns `(k, P, sha256)`."""
    path = Path(path)
    raw = path.read_bytes()
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{path}: expected two columns, got shape {data.shape}")
    k, P = data[:, 0], data[:, 1]
    if not (np.all(np.diff(k) > 0) and k[0] > 0):
        raise ValueError(f"{path}: k must be positive and strictly increasing")
    if not np.all(P > 0):
        raise ValueError(f"{path}: P must be positive (log-log spline)")
    return k, P, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PowerSpectrum:
    """Cubic spline in (log k, log P) with power-law tails whose slopes are the secant
    log-slopes of the two end nodes on each side. `__call__(0) == 0`."""

    k: np.ndarray
    P: np.ndarray
    file_hash: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "k", np.asarray(self.k, dtype=np.float64))
        object.__setattr__(self, "P", np.asarray(self.P, dtype=np.float64))
        if self.k.ndim != 1 or self.k.shape != self.P.shape or self.k.size < 4:
            raise ValueError("k and P must be 1-d, equal length, >= 4 nodes")
        lk, lp = np.log(self.k), np.log(self.P)
        object.__setattr__(self, "_spline", CubicSpline(lk, lp, extrapolate=False))
        object.__setattr__(self, "slope_lo", (lp[1] - lp[0]) / (lk[1] - lk[0]))
        object.__setattr__(self, "slope_hi", (lp[-1] - lp[-2]) / (lk[-1] - lk[-2]))

    @classmethod
    def from_tsv(cls, path):
        k, P, h = load_pk_tsv(path)
        return cls(k, P, h)

    def scaled(self, factor: float) -> "PowerSpectrum":
        """`factor * P` (e.g. b^2), same nodes and hash."""
        return PowerSpectrum(self.k, self.P * float(factor), self.file_hash)

    def __call__(self, k):
        k = np.asarray(k, dtype=np.float64)
        out = np.zeros_like(k)
        pos = k > 0
        kp = k[pos]
        lk = np.log(kp)
        lo = kp < self.k[0]
        hi = kp > self.k[-1]
        mid = ~(lo | hi)
        val = np.empty_like(kp)
        val[mid] = np.exp(self._spline(lk[mid]))
        val[lo] = self.P[0] * (kp[lo] / self.k[0]) ** self.slope_lo
        val[hi] = self.P[-1] * (kp[hi] / self.k[-1]) ** self.slope_hi
        out[pos] = val
        return out


def check_table_pair(linear: PowerSpectrum, galaxy: PowerSpectrum) -> float:
    """Check that a nonlinear galaxy-target table belongs with a bin's linear table;
    returns `|P_gal / P_lin - 1|` at their common lowest node (both must start at the
    same k), raises if it exceeds `k0^2 sigma_v^2` with
    `sigma_v^2 = (1/6 pi^2) int P_lin dk`, the leading low-k nonlinear correction
    (`P_13 -> -k^2 sigma_v^2 P_lin`); a table at another z or cosmology is off at
    order unity. Pair a halofit table with the kh-10 linear table, not the v28 one."""
    if linear.k[0] != galaxy.k[0]:
        raise ValueError(
            f"P(k) tables start at different k ({linear.k[0]} vs {galaxy.k[0]} h/Mpc)"
        )
    sigma_v2 = np.trapezoid(linear.P, linear.k) / (6.0 * np.pi**2)
    bound = linear.k[0] ** 2 * sigma_v2
    dev = abs(galaxy.P[0] / linear.P[0] - 1.0)
    if not dev <= bound:
        raise ValueError(
            f"galaxy P(k) table departs from the linear one by {dev:.3e} at "
            f"k = {linear.k[0]} h/Mpc (bound k0^2 sigma_v^2 = {bound:.3e}): not the "
            "same z / cosmology"
        )
    return float(dev)


def radius_index(box: Box) -> np.ndarray:
    """`i^2 + j^2 + l^2` (int32) on the rfft grid, the integer mode indices in
    `fftfreq` / `rfftfreq` order, so `|k| = k_f sqrt(radius_index)`."""
    n = box.n_mesh
    i = np.arange(n, dtype=np.int32)
    i2 = np.where(i < n // 2, i, i - n) ** 2
    l2 = np.arange(n // 2 + 1, dtype=np.int32) ** 2
    return i2[:, None, None] + i2[None, :, None] + l2[None, None, :]


def pk_on_grid(spectrum, box: Box):
    """Evaluate `spectrum(|k|)` (any callable of k in h/Mpc) on the rfft grid, DC = 0.
    Calls `spectrum` once on `k_f sqrt(q)` for every integer `q <= 3 (N/2)^2` and
    gathers by `radius_index`; ~1e-15 relative from `spectrum(k_grid(box)[2])`, not
    bitwise (the two `|k|` constructions round differently)."""
    n2 = box.n_mesh // 2
    table = spectrum(box.k_f * np.sqrt(np.arange(3 * n2 * n2 + 1, dtype=np.float64)))
    P = np.asarray(table, dtype=np.float64)[radius_index(box)]
    P[0, 0, 0] = 0.0
    return P


def grid_xi(pk_grid, box: Box, xp=np):
    """`xi(x) = (1/V) sum_k P(k) e^{ikx}` on the grid: `irfftn(P) / V_cell`."""
    return xp.fft.irfftn(xp.asarray(pk_grid), s=box.shape, axes=(0, 1, 2)) / box.v_cell


def grid_pk_from_xi(xi, box: Box, xp=np):
    """Inverse of `grid_xi`: `rfftn(xi) * V_cell` (real part; xi is even)."""
    return xp.real(xp.fft.rfftn(xp.asarray(xi))) * box.v_cell


def grid_pkG(pk_grid, box: Box, xp=np):
    """Grid-native P -> P_G. Returns `(pkG, diagnostics)` with `pkG >= 0`, `pkG[0,0,0]
    == 0`, and diagnostics: `xi_min`, `sigma2` (= xi at zero lag, the cell variance of
    the target field), `n_clipped`, `clipped_power_fraction` (= sum |P_G<0| / sum
    P_G>=0). Raises if `xi <= -1` anywhere (log1p undefined: unattainable target)."""
    xi = grid_xi(pk_grid, box, xp)
    xi_min = float(xi.min())
    sigma2 = float(xi[0, 0, 0])
    if xi_min <= -1.0:
        raise ValueError(f"xi(x) reaches {xi_min} <= -1: no lognormal has this P(k)")
    xiG = xp.log1p(xi)
    pkG = grid_pk_from_xi(xiG, box, xp)
    pkG = pkG.at[0, 0, 0].set(0.0) if hasattr(pkG, "at") else _set_dc(pkG)
    neg = pkG < 0
    n_clipped = int(neg.sum())
    neg_power = float(xp.where(neg, -pkG, 0.0).sum())
    pos_power = float(xp.where(neg, 0.0, pkG).sum())
    pkG = xp.where(neg, 0.0, pkG)
    diag = {
        "xi_min": xi_min,
        "sigma2": sigma2,
        "n_clipped": n_clipped,
        "clipped_power_fraction": neg_power / pos_power if pos_power > 0 else 0.0,
    }
    return pkG, diag


def _set_dc(arr):
    arr = np.array(arr, copy=True)
    arr[0, 0, 0] = 0.0
    return arr
