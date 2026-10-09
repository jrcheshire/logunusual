"""Periodic-box P(k) estimator and the analytic references the checks use (numpy).

CIC painting and deconvolution, Jing (2005) shot noise, `power_multipoles`
(`P = V/N^6 |delta_k|^2`, Legendre multipoles about z, Hermitian-weighted rfft
half-grid, shells of width `k_f` from `k_f / 2`), Gaussian SEs, Kaiser boosts, and the
coherent-alias `estimator_response`. Measure a generator of mesh N on a mesh 2N to
keep the estimator's own aliasing out of the generator's first zone (< 0.1% at its
Nyquist for a CDM slope).
"""

import numpy as np

from .grid import (
    Box,
    cic_shot_noise_factor,
    frequencies,
    hermitian_weights,
    k_grid,
    sinc_window,
)


def cic_paint(xyz, box: Box):
    """CIC weights on the mesh (sum = number of particles); positions may be anywhere,
    they are wrapped periodically."""
    pos = np.asarray(xyz, dtype=np.float64)
    n = box.n_mesh
    g = pos / box.dx
    i0 = np.floor(g).astype(np.int64)
    frac = g - i0
    rho = np.zeros(n * n * n, dtype=np.float64)
    for dxi in (0, 1):
        wx = frac[:, 0] if dxi else 1.0 - frac[:, 0]
        ix = (i0[:, 0] + dxi) % n
        for dyi in (0, 1):
            wy = frac[:, 1] if dyi else 1.0 - frac[:, 1]
            iy = (i0[:, 1] + dyi) % n
            for dzi in (0, 1):
                wz = frac[:, 2] if dzi else 1.0 - frac[:, 2]
                iz = (i0[:, 2] + dzi) % n
                flat = (ix * n + iy) * n + iz
                rho += np.bincount(flat, weights=wx * wy * wz, minlength=n * n * n)
    return rho.reshape(n, n, n)


def delta_k_from_positions(xyz, box: Box, *, deconvolve=True):
    """`rfftn(rho / mean - 1)`, divided by the CIC window `sinc^2` per axis if asked."""
    rho = cic_paint(xyz, box)
    delta = rho / rho.mean() - 1.0
    dk = np.fft.rfftn(delta)
    if deconvolve:
        dk = dk / sinc_window(box, 2.0)
    return dk


def effective_window(box: Box, box_est: Box, jitter_p: int, n_alias: int = 64):
    """Per-mode factor relating the CIC-painted (UNdeconvolved) mesh power of a catalog
    drawn from a grid field to the grid power `P_grid(k mod 2pi/dx)`:

        P_mesh(k) = |sum_n W_cic(k_n) T(k_n)|^2 P_grid(k),   k_n = k + 2 pi n / H,

    with `H` the estimator spacing, `W_cic = prod sinc^2`, `T = prod sinc^p` the jitter
    window in generator-cell units. The catalog is lattice-periodic with cell centres
    at half-integer multiples of `dx`, so the alias images carry the SAME grid
    coefficient times `(-1)^(r n)` per axis (`r = dx / H`, the integer mesh ratio) and
    add coherently; the incoherent alias sum (Jing 2005) holds for shot noise only. On
    an even-ratio mesh every sign is +; on the generator's own mesh the dominant image
    SUBTRACTS (power 3.5% low at half the Nyquist, shell average). Separable; each 1-d
    sum is truncated at |n| <= n_alias."""
    r = box_est.n_mesh / box.n_mesh
    if abs(r - round(r)) > 1e-12:
        raise ValueError(
            "estimator mesh must be an integer multiple of the generator's"
        )
    r = int(round(r))
    f, fz = frequencies(box_est)

    def axis(fe):
        acc = np.zeros_like(fe)
        for n in range(-n_alias, n_alias + 1):
            x = fe + n
            sign = -1.0 if (r * n) % 2 else 1.0
            acc += sign * np.sinc(x) ** 2 * np.sinc(r * x) ** jitter_p
        return acc**2

    ax, az = axis(f), axis(fz)
    return ax[:, None, None] * ax[None, :, None] * az[None, None, :]


def estimator_response(box: Box, box_est: Box, jitter_p: int):
    """`effective_window / (W_cic^2 T^2)`: the ratio of the deconvolved mesh power to
    the catalog's true continuum power `T^2 P_grid` in the first zone. NaN where the
    jitter window vanishes (only at the generator's 2x Nyquist and beyond)."""
    f, fz = frequencies(box_est)
    r = box_est.n_mesh / box.n_mesh
    T2 = (
        (np.sinc(r * f) ** (2 * jitter_p))[:, None, None]
        * (np.sinc(r * f) ** (2 * jitter_p))[None, :, None]
        * (np.sinc(r * fz) ** (2 * jitter_p))[None, None, :]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return effective_window(box, box_est, jitter_p) / (
            sinc_window(box_est, 4.0) * T2
        )


def shot_noise_k(box: Box, nbar: float, *, deconvolved=True):
    """Expected shot power per rfft mode of a CIC-painted Poisson process of density
    `nbar` (Jing 2005 eq. 20), divided by the CIC power window when `deconvolved`."""
    s = cic_shot_noise_factor(box) / float(nbar)
    if deconvolved:
        s = s / sinc_window(box, 4.0)
    return s


def shell_edges(box: Box):
    """Edges `k_f/2, 3k_f/2, ...` up to the Nyquist; centres are multiples of k_f."""
    return np.arange(0.5 * box.k_f, box.k_nyq + box.k_f, box.k_f)


_LEGENDRE = {
    0: lambda mu: np.ones_like(mu),
    2: lambda mu: 0.5 * (3.0 * mu**2 - 1.0),
    4: lambda mu: 0.125 * (35.0 * mu**4 - 30.0 * mu**2 + 3.0),
}


def _shell_index(box: Box):
    _, _, k_mag = k_grid(box)
    edges = shell_edges(box)
    idx = np.digitize(k_mag.ravel(), edges) - 1
    nb = edges.size - 1
    valid = (idx >= 0) & (idx < nb)
    return k_mag, np.where(valid, idx, nb), nb


def power_multipoles(delta_k, box: Box, *, ells=(0, 2, 4), shot_k=None, delta_k_b=None):
    """Returns dict with `k` (shell centres), `nmodes` (full-grid mode counts), and
    `P{ell}` in (Mpc/h)^3. `shot_k` (per-mode array or scalar) is subtracted from the
    per-mode power before the Legendre weighting. With `delta_k_b` the cross power
    `Re(a b*)` is used instead of `|a|^2`."""
    n = box.n_mesh
    norm = box.volume / n**6
    if delta_k_b is None:
        P = (np.abs(delta_k) ** 2) * norm
    else:
        P = np.real(delta_k * np.conj(delta_k_b)) * norm
    if shot_k is not None:
        P = P - shot_k
    k_mag, idx, nb = _shell_index(box)
    herm = np.broadcast_to(hermitian_weights(box), k_mag.shape).ravel()
    _, kz_1d, _ = k_grid(box)
    kz = np.broadcast_to(kz_1d[None, None, :], k_mag.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(k_mag > 0, kz / k_mag, 0.0).ravel()
    P = P.ravel()
    wsum = np.bincount(idx, weights=herm, minlength=nb + 1)[:nb]
    out = {
        "k": 0.5 * (shell_edges(box)[1:] + shell_edges(box)[:-1]),
        "nmodes": wsum,  # full-grid count (the weight of the shell)
        "n_indep": 0.5 * wsum,  # independent modes, as `gaussian_se` counts them
    }
    for ell in ells:
        contrib = P * _LEGENDRE[ell](mu) * herm
        sums = np.bincount(idx, weights=contrib, minlength=nb + 1)[:nb]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"P{ell}"] = (2 * ell + 1) * sums / wsum
    return out


def field_power(delta, box: Box, **kw):
    """Multipoles of a real-space grid field."""
    return power_multipoles(
        np.fft.rfftn(np.asarray(delta, dtype=np.float64)), box, **kw
    )


def shell_average(values_k, box: Box):
    """Hermitian-weighted shell average of any per-mode quantity (e.g. a target P)."""
    k_mag, idx, nb = _shell_index(box)
    herm = np.broadcast_to(hermitian_weights(box), k_mag.shape).ravel()
    wsum = np.bincount(idx, weights=herm, minlength=nb + 1)[:nb]
    sums = np.bincount(
        idx, weights=np.asarray(values_k).ravel() * herm, minlength=nb + 1
    )[:nb]
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / wsum


def gaussian_se(pk_grid, box: Box):
    """Standard error of the shell-mean power (the Hermitian-weighted mean) of ONE
    Gaussian realization with per-mode expectation `pk_grid`:

        SE = sqrt(2 sum_c w_c P_c^2) / sum_c w_c        over the shell's half-grid cells

    with `w` the Hermitian weights. Exact: an interior cell stands for itself and its
    conjugate (Var |delta|^2 = P^2, weight 2); a kz = 0 or Nyquist plane cell and its
    conjugate are both on the half grid and equal (weight 1 each); a self-conjugate
    mode is real (Var |delta|^2 = 2 P^2, weight 1). Every case gives `2 w P^2`, so a
    shell holds `nmodes / 2` independent modes."""
    k_mag, idx, nb = _shell_index(box)
    herm = np.broadcast_to(hermitian_weights(box), k_mag.shape).ravel()
    P2 = (np.asarray(pk_grid, dtype=np.float64) ** 2).ravel()
    wsum = np.bincount(idx, weights=herm, minlength=nb + 1)[:nb]
    s2 = np.bincount(idx, weights=2.0 * herm * P2, minlength=nb + 1)[:nb]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(s2) / wsum


def kaiser_boost(ell: int, beta: float) -> float:
    """`P_ell / P_real` in linear theory for `beta = f/b`."""
    if ell == 0:
        return 1.0 + 2.0 * beta / 3.0 + beta**2 / 5.0
    if ell == 2:
        return 4.0 * beta / 3.0 + 4.0 * beta**2 / 7.0
    if ell == 4:
        return 8.0 * beta**2 / 35.0
    raise ValueError(ell)


def kaiser_ratio(beta: float) -> float:
    """`P_2 / P_0` in linear theory."""
    return kaiser_boost(2, beta) / kaiser_boost(0, beta)


def rebin(k, values, weights, edges):
    """Weight-averaged rebinning of shell values into `edges`; returns
    `(k_centre_weighted, value, weight_sum)` with empty bins dropped."""
    k = np.asarray(k)
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)
    idx = np.digitize(k, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(values)
    nb = len(edges) - 1
    w = np.bincount(idx[ok], weights=weights[ok], minlength=nb)
    kk = np.bincount(idx[ok], weights=(weights * k)[ok], minlength=nb)
    vv = np.bincount(idx[ok], weights=(weights * values)[ok], minlength=nb)
    keep = w > 0
    return kk[keep] / w[keep], vv[keep] / w[keep], w[keep]


def gate_bands(box: Box, k_max: float, *, dex=0.1, k_switch=0.05):
    """Edges for the gate bands: kf-shells above `k_switch`, 0.1-dex log bins below it
    (few modes per shell at low k), all below `k_max`."""
    lo = np.arange(np.log10(0.5 * box.k_f), np.log10(k_switch), dex)
    lo = 10.0**lo
    hi = shell_edges(box)
    hi = hi[(hi >= k_switch) & (hi <= k_max)]
    return np.concatenate([lo, hi])
