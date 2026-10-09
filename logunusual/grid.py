"""Periodic cubic mesh: box geometry, k-space grids, and the separable windows.

Conventions: `k_i = 2 pi fftfreq(N, d=dx)`; real FFTs compress the LAST axis (z), which
is also the plane-parallel line of sight. Cell centres sit at `(i + 0.5) dx`. Units:
Mpc/h for lengths, h/Mpc for wavenumbers.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Box:
    n_mesh: int
    box_size: float  # Mpc/h

    def __post_init__(self):
        if self.n_mesh < 2 or self.n_mesh % 2:
            raise ValueError(f"n_mesh must be even and >= 2, got {self.n_mesh}")
        if not self.box_size > 0:
            raise ValueError(f"box_size must be positive, got {self.box_size}")

    @property
    def dx(self) -> float:
        return self.box_size / self.n_mesh

    @property
    def v_cell(self) -> float:
        return self.dx**3

    @property
    def volume(self) -> float:
        return self.box_size**3

    @property
    def n_cells(self) -> int:
        return self.n_mesh**3

    @property
    def k_f(self) -> float:
        """Fundamental wavenumber 2 pi / L."""
        return 2.0 * math.pi / self.box_size

    @property
    def k_nyq(self) -> float:
        """Nyquist wavenumber pi / dx."""
        return math.pi / self.dx

    @property
    def shape(self) -> tuple:
        return (self.n_mesh, self.n_mesh, self.n_mesh)

    @property
    def rfft_shape(self) -> tuple:
        return (self.n_mesh, self.n_mesh, self.n_mesh // 2 + 1)


def frequencies(box: Box):
    """Cycles per cell `k_i dx / 2 pi` along a full and the rfft axis: `fftfreq(N)`,
    `rfftfreq(N)`."""
    n = box.n_mesh
    return np.fft.fftfreq(n), np.fft.rfftfreq(n)


def k_grid(box: Box):
    """`(k_1d, kz_1d, k_mag)` in h/Mpc; `k_mag` has the rfft shape."""
    f, fz = frequencies(box)
    k_1d = 2.0 * np.pi * f / box.dx
    kz_1d = 2.0 * np.pi * fz / box.dx
    k_mag = np.sqrt(
        k_1d[:, None, None] ** 2 + k_1d[None, :, None] ** 2 + kz_1d[None, None, :] ** 2
    )
    return k_1d, kz_1d, k_mag


def k_components(box: Box):
    """Broadcastable `(kx, ky, kz)` with shapes (N,1,1), (1,N,1), (1,1,N//2+1)."""
    k_1d, kz_1d, _ = k_grid(box)
    return k_1d[:, None, None], k_1d[None, :, None], kz_1d[None, None, :]


def hermitian_weights(box: Box):
    """Mode multiplicity of the rfft half-grid, shape (1, 1, N//2+1): 1 on the kz = 0
    and kz = Nyquist planes (each cell's conjugate is another cell of the same plane),
    2 elsewhere (the conjugate is off the half grid). Summing `w * f(k)` over the
    half-grid reproduces the full-grid sum for any even `f`."""
    n = box.n_mesh
    w = np.full(n // 2 + 1, 2.0)
    w[0] = 1.0
    w[-1] = 1.0  # n is even
    return w[None, None, :]


def sinc_window(box: Box, power: float):
    """Separable `prod_i sinc(k_i dx / 2 pi) ** power` on the rfft grid, numpy's
    `sinc(x) = sin(pi x) / (pi x)` of the cycles per cell; exactly 1 at k = 0.
    `power = 2` is the CIC amplitude window."""
    f, fz = frequencies(box)
    wx = np.sinc(f) ** power
    wz = np.sinc(fz) ** power
    return wx[:, None, None] * wx[None, :, None] * wz[None, None, :]


def jitter_power_window(box: Box, p: int):
    """Power window `prod_i sinc(k_i dx / 2 pi) ** (2 p)` of the order-`p` intra-cell
    jitter (sum of `p` uniforms). The target P(k) is divided by it before the lognormal
    transform (`field.target_on_grid`)."""
    return sinc_window(box, 2.0 * p)


def cic_shot_noise_factor(box: Box):
    """`prod_i [1 - (2/3) sin^2(k_i dx/2)]` on the rfft grid; times `1/nbar`, the
    aliased CIC shot-noise power before deconvolution (Jing 2005, ApJ 620, 559)."""
    f, fz = frequencies(box)
    cx = 1.0 - (2.0 / 3.0) * np.sin(np.pi * f) ** 2
    cz = 1.0 - (2.0 / 3.0) * np.sin(np.pi * fz) ** 2
    return cx[:, None, None] * cx[None, :, None] * cz[None, None, :]
