"""Statistical checks of the periodic box and the shell product, as functions returning
plain dicts, shared by the slow tests and `scripts/gates_{box,shell}.py`.

Each statistic is formed per realization, then averaged; the SE is the scatter across
realizations (never a Gaussian formula on a lognormal quantity); bands come from the
tolerance and seed count (`n_min_indep`). The generator lives on `box`; catalogs are
measured on `box_est` (typically 2x mesh). Bars and usage: `docs/validation.md`.
"""

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import field, fnl as _fnl, sample, shell, validate
from .grid import Box, sinc_window
from .pk import pk_on_grid

TOL = 0.02  # resolution: SE must be <= TOL / 3 in every gated band
Z_MAX = 4.0  # per-band |z| bound; P(|z| > 4) = 6e-5 per band under the null


def n_min_indep(n_real: int, tol: float = TOL) -> int:
    """Independent modes per band so that a Gaussian-scatter SE over `n_real`
    realizations resolves `tol` at 3 sigma: `1 / sqrt(n_indep n_real) <= tol / 3`."""
    return int(np.ceil(9.0 / (tol**2 * n_real)))


def band_edges(box: Box, k_max: float, n_min: int):
    """Merge kf-shells from low k upward until each band holds >= `n_min` independent
    (half-grid) modes; the last partial band is merged into its predecessor."""
    edges = validate.shell_edges(box)
    _, idx, nb = validate._shell_index(box)
    counts = np.bincount(idx, minlength=nb + 1)[:nb]
    out = [edges[0]]
    acc = 0
    for i in range(nb):
        if edges[i + 1] > k_max + 1e-12:
            break
        acc += counts[i]
        if acc >= n_min:
            out.append(edges[i + 1])
            acc = 0
    if len(out) < 2:
        raise ValueError("no band reaches n_min modes below k_max")
    return np.array(out)


def band_ratio(k, num, den, weights, edges):
    """Per-realization band statistic: (sum w num) / (sum w den) over the shells in
    each band. Returns `(k_band, ratio, w_band)`."""
    _, n, w = validate.rebin(k, num, weights, edges)
    kk, d, _ = validate.rebin(k, den, weights, edges)
    return kk, n / d, w


@dataclass
class Ensemble:
    k: np.ndarray
    n_real: int
    mean: np.ndarray
    se: np.ndarray
    z: np.ndarray
    se_ok: np.ndarray  # SE <= TOL / 3 in the band
    passed: bool
    extra: dict = dc_field(default_factory=dict)

    def as_dict(self):
        return {
            "k": self.k.tolist(),
            "n_real": self.n_real,
            "mean": self.mean.tolist(),
            "se": self.se.tolist(),
            "z": self.z.tolist(),
            "se_ok": self.se_ok.tolist(),
            "passed": bool(self.passed),
            **self.extra,
        }


def ensemble(k, ratios, *, target=1.0, tol=TOL, require_se=True, extra=None):
    r = np.asarray(ratios)  # (n_real, n_band)
    n = r.shape[0]
    mean = r.mean(axis=0)
    se = r.std(axis=0, ddof=1) / np.sqrt(n)
    z = (mean - target) / se
    se_ok = se <= tol / 3.0
    passed = bool(np.all(np.abs(z) < Z_MAX) and (np.all(se_ok) or not require_se))
    return Ensemble(np.asarray(k), n, mean, se, z, se_ok, passed, extra or {})


# ------------------------------------------------------------------ grid identity


def gate_field_identity(
    spectrum, b, box: Box, seeds, *, jitter_p=1, fnl=None, galaxy_table=None
):
    """<P(delta_g)> / target = 1 on every band up to the Nyquist (grid identity); the
    matter field likewise. `target` is the deconvolved grid target the field was built
    for (`b(k)^2 P` with `fnl`, an `fnl.LocalPNG`; `P` from `galaxy_table` if given)."""
    tg = validate.shell_average(
        field.target_on_grid(
            _fnl.galaxy_spectrum(spectrum, b, fnl, galaxy_table), box, jitter_p
        ),
        box,
    )
    tm = validate.shell_average(field.target_on_grid(spectrum, box, jitter_p), box)
    edges = band_edges(box, box.k_nyq, n_min_indep(len(seeds)))
    rg, rm, clipped = [], [], []
    for s in seeds:
        F = field.generate_fields(
            spectrum,
            b,
            box,
            s,
            jitter_p=jitter_p,
            keep_matter=True,
            fnl=fnl,
            galaxy_table=galaxy_table,
        )
        pg = validate.field_power(F.delta_g, box)
        pm = validate.field_power(F.delta_m, box)
        kb, r, _ = band_ratio(pg["k"], pg["P0"], tg, pg["nmodes"], edges)
        rg.append(r)
        rm.append(band_ratio(pm["k"], pm["P0"], tm, pm["nmodes"], edges)[1])
        clipped.append(F.diagnostics["galaxy"]["clipped_power_fraction"])
    extra = {
        "clipped_power_fraction_max": float(np.max(clipped)),
        "sigma2_galaxy": F.diagnostics["galaxy"]["sigma2"],
        "sigma2_matter": F.diagnostics["matter"]["sigma2"],
    }
    return {"galaxy": ensemble(kb, rg, extra=extra), "matter": ensemble(kb, rm)}


# ------------------------------------------------- shot noise and fixed-field sampler


def gate_uniform_shot(
    box: Box, box_est: Box, nbar, seeds, *, jitter_p=1, band_seeds=None
):
    """A constant intensity sampled and placed like a real catalog has, after CIC
    deconvolution on the estimator mesh, exactly the Jing (2005) shot spectrum.

    `band_seeds` (default `len(seeds)`): the seed count the bands are designed for, as
    in `gate_catalog`; the super-Gaussian scatter needs more seeds than that through
    the same bands (bands derived from more seeds do not help: the count cancels)."""
    shot = validate.shot_noise_k(box_est, nbar)
    pred = validate.shell_average(shot, box_est)
    edges = band_edges(box_est, box.k_nyq, n_min_indep(band_seeds or len(seeds)))
    ratios = []
    lam = np.full(box.shape, nbar * box.v_cell)
    for s in seeds:
        parts = [p[0] for p in sample.draw_slabs(lam, box, s, jitter_p=jitter_p)]
        xyz = np.concatenate(parts)
        res = validate.power_multipoles(
            validate.delta_k_from_positions(xyz, box_est), box_est
        )
        kb, r, _ = band_ratio(res["k"], res["P0"], pred, res["nmodes"], edges)
        ratios.append(r)
    return ensemble(kb, ratios)


def _gen_to_est_index(box: Box, box_est: Box):
    """For every rfft mode of the estimator mesh, the index of the generator mode with
    the same wavevector modulo the generator's reciprocal lattice (the image mapping).
    """
    n, ne = box.n_mesh, box_est.n_mesh
    m = np.fft.fftfreq(ne, d=1.0 / ne).astype(np.int64)  # signed integer indices
    mz = np.arange(ne // 2 + 1)
    ix = (m % n)[:, None, None]
    iy = (m % n)[None, :, None]
    iz = (mz % n)[None, None, :]
    return ix, iy, iz


def catalog_power_prediction(
    delta_grid, box: Box, box_est: Box, jitter_p, delta_grid_b=None
):
    """Exact per-mode expectation of the DECONVOLVED, shot-free mesh power measured by
    `validate.delta_k_from_positions` on `box_est` for a FIXED grid field: the grid
    power (periodic in k) times `validate.effective_window` over the CIC deconvolution;
    with `delta_grid_b`, the cross power. Exact in real space; in redshift space the
    per-cell displacements break the lattice periodicity, so it is approximate."""
    n = box.n_mesh
    a = np.fft.fftn(np.asarray(delta_grid, dtype=np.float64))
    if delta_grid_b is None:
        P = np.abs(a) ** 2
    else:
        P = np.real(
            a * np.conj(np.fft.fftn(np.asarray(delta_grid_b, dtype=np.float64)))
        )
    P = P * box.volume / n**6
    ix, iy, iz = _gen_to_est_index(box, box_est)
    P_img = P[ix, iy, iz]
    return (
        P_img
        * validate.effective_window(box, box_est, jitter_p)
        / sinc_window(box_est, 4.0)
    )


def gate_fixed_field_sampler(fields, nbar, draw_seeds, box_est: Box, *, k_max=None):
    """For ONE field, draw-averaged catalog power (shot removed) over the exact
    fixed-field prediction = 1, isolating placement + Poisson + estimator from cosmic
    variance. Gated to `k_max` (default the generator Nyquist), reported to the
    estimator Nyquist (the images)."""
    box = fields.box
    k_max = box.k_nyq if k_max is None else k_max
    pred = catalog_power_prediction(fields.delta_g, box, box_est, fields.jitter_p)
    pred_shell = validate.shell_average(pred, box_est)
    shot = validate.shot_noise_k(box_est, nbar)
    edges = band_edges(box_est, k_max, n_min_indep(len(draw_seeds)))
    edges_all = band_edges(box_est, box_est.k_nyq, n_min_indep(len(draw_seeds)))
    ratios, ratios_all = [], []
    for s in draw_seeds:
        cat = sample.sample_catalog(fields, nbar, s, rsd=False)
        res = validate.power_multipoles(
            validate.delta_k_from_positions(cat.xyz, box_est), box_est, shot_k=shot
        )
        kb, r, _ = band_ratio(res["k"], res["P0"], pred_shell, res["nmodes"], edges)
        ka, ra, _ = band_ratio(
            res["k"], res["P0"], pred_shell, res["nmodes"], edges_all
        )
        ratios.append(r)
        ratios_all.append(ra)
    return {
        "first_zone": ensemble(kb, ratios),
        "to_estimator_nyquist": ensemble(ka, ratios_all, require_se=False),
    }


# -------------------------------------------------------- catalog power, RSD, density


def gate_catalog(
    spectrum,
    b,
    f,
    box: Box,
    box_est: Box,
    nbar,
    seeds,
    *,
    jitter_p=1,
    k_max=None,
    band_seeds=None,
):
    """Real-space monopole against `b^2 P_in x estimator_response` and against the
    fixed-field prediction; Poisson density check; redshift-space MEASUREMENTS: P2/P0
    over Kaiser(f/b), the same over the linear prediction from the realization's own
    grid spectra, and the premise P_gm / (b P_mm). Linear RSD is a k -> 0 limit, so
    Kaiser is asserted only in the lowest band (`gate_kaiser_linear_limit` is the exact
    check). `band_seeds` (default `len(seeds)`): the seed count the bands are designed
    for; more seeds through the same bands bring a super-Gaussian scatter under the SE
    floor."""
    k_max = 0.5 * box.k_nyq if k_max is None else k_max
    n_band = n_min_indep(band_seeds or len(seeds))
    edges = band_edges(box_est, k_max, n_band)
    edges_g = band_edges(box, k_max, n_band)
    shot = validate.shot_noise_k(box_est, nbar)
    R = validate.estimator_response(box, box_est, jitter_p)
    target = validate.shell_average(
        pk_on_grid(lambda k: b * b * spectrum(k), box_est) * R, box_est
    )
    r_mono, r_fixed, r_kaiser, r_gen2, r_gen0, premise, z_pois, dens = (
        [] for _ in range(8)
    )
    psi_rms = []
    beta = f / b
    for s in seeds:
        ic, draw = sample.split_seed(s)
        F = field.generate_fields(
            spectrum, b, box, ic, jitter_p=jitter_p, keep_matter=True
        )
        psi_rms.append(F.diagnostics["psi_z_rms"])
        real = sample.sample_catalog(F, nbar, draw, f=f, rsd=False)
        red = sample.sample_catalog(F, nbar, draw, f=f, rsd=True)
        # density
        z_pois.append((real.n_galaxies - real.lam_total) / np.sqrt(real.lam_total))
        dens.append(real.lam_total / (nbar * box.volume))
        # real-space monopole
        pr = validate.power_multipoles(
            validate.delta_k_from_positions(real.xyz, box_est), box_est, shot_k=shot
        )
        kb, r, _ = band_ratio(pr["k"], pr["P0"], target, pr["nmodes"], edges)
        r_mono.append(r)
        pred_fix = validate.shell_average(
            catalog_power_prediction(F.delta_g, box, box_est, jitter_p), box_est
        )
        r_fixed.append(band_ratio(pr["k"], pr["P0"], pred_fix, pr["nmodes"], edges)[1])
        # redshift space
        ps = validate.power_multipoles(
            validate.delta_k_from_positions(red.xyz, box_est), box_est, shot_k=shot
        )
        _, q, _ = band_ratio(ps["k"], ps["P2"], ps["P0"], ps["nmodes"], edges)
        r_kaiser.append(q / validate.kaiser_ratio(beta))
        Pgm = catalog_power_prediction(F.delta_g, box, box_est, jitter_p, F.delta_m)
        Pmm = catalog_power_prediction(F.delta_m, box, box_est, jitter_p)
        Pgg = catalog_power_prediction(F.delta_g, box, box_est, jitter_p)
        p2 = validate.shell_average((4 * f / 3) * Pgm + (4 * f * f / 7) * Pmm, box_est)
        p0 = validate.shell_average(
            Pgg + (2 * f / 3) * Pgm + (f * f / 5) * Pmm, box_est
        )
        r_gen2.append(band_ratio(ps["k"], ps["P2"], p2, ps["nmodes"], edges)[1])
        r_gen0.append(band_ratio(ps["k"], ps["P0"], p0, ps["nmodes"], edges)[1])
        # linear premise on the generator grid: P_gm / (b P_mm)
        gm = validate.power_multipoles(
            np.fft.rfftn(np.asarray(F.delta_g)),
            box,
            delta_k_b=np.fft.rfftn(np.asarray(F.delta_m)),
        )
        mm = validate.field_power(F.delta_m, box)
        kg, pr_ratio, _ = band_ratio(
            gm["k"], gm["P0"], b * mm["P0"], gm["nmodes"], edges_g
        )
        premise.append(pr_ratio)
    kaiser_all = ensemble(kb, r_kaiser, require_se=False)
    kaiser_lowest = ensemble(kb[:1], np.asarray(r_kaiser)[:, :1], require_se=False)
    zp = np.asarray(z_pois)
    density = {
        "z_poisson_mean": float(zp.mean()),
        "z_poisson_std": float(zp.std(ddof=1)),
        "lam_over_nbarV_max_dev": float(np.max(np.abs(np.asarray(dens) - 1.0))),
        "passed": bool(
            abs(zp.mean()) < Z_MAX / np.sqrt(len(seeds))
            and 0.5 < zp.std(ddof=1) < 1.6
            and np.max(np.abs(np.asarray(dens) - 1.0)) < 1e-10
        ),
    }
    return {
        "monopole": ensemble(
            kb,
            r_mono,
            extra={
                "estimator_response_band_max_dev": float(
                    np.max(
                        np.abs(
                            validate.shell_average(R, box_est)[
                                validate.shell_edges(box_est)[:-1] < k_max
                            ]
                            - 1.0
                        )
                    )
                )
            },
        ),
        "monopole_fixed_field": ensemble(kb, r_fixed),
        "kaiser_lowest_band": kaiser_lowest,
        "kaiser_all_bands": kaiser_all,
        "premise": ensemble(kg, premise, require_se=False),
        "quadrupole_generalised": ensemble(kb, r_gen2, require_se=False),
        "monopole_generalised": ensemble(kb, r_gen0, require_se=False),
        "density": density,
        "f_psi_rms": float(f * np.mean(psi_rms)),
    }


# ---------------------------------------------------------------- Kaiser linear limit


def gate_kaiser_linear_limit(
    spectrum,
    b,
    f,
    box: Box,
    box_est: Box,
    nbar,
    seeds,
    *,
    eps=1e-2,
    jitter_p=1,
    k_max=None,
):
    """Kaiser as an exact statement: with the input spectrum scaled by `eps` the fields
    are Gaussian and fully correlated to O(eps) and the RSD mapping is linear to
    O((k f Psi_rms)^2), so `P2 / P0 = kaiser_ratio(f / b)` on every band; the residual
    budget `(k_max f Psi_rms)^2` is returned. Use a dense catalog (shot noise below the
    small signal). A consistency check (|z| < Z_MAX, no SE floor); the RSD path's
    exactness is tested deterministically in `tests/test_sample.py`."""
    k_max = 0.5 * box.k_nyq if k_max is None else k_max
    scaled = lambda k: eps * spectrum(k)  # noqa: E731
    edges = band_edges(box_est, k_max, n_min_indep(len(seeds)))
    shot = validate.shot_noise_k(box_est, nbar)
    ratios, psi_rms = [], []
    for s in seeds:
        ic, draw = sample.split_seed(s)
        F = field.generate_fields(scaled, b, box, ic, jitter_p=jitter_p)
        psi_rms.append(F.diagnostics["psi_z_rms"])
        red = sample.sample_catalog(F, nbar, draw, f=f, rsd=True)
        ps = validate.power_multipoles(
            validate.delta_k_from_positions(red.xyz, box_est), box_est, shot_k=shot
        )
        kb, q, _ = band_ratio(ps["k"], ps["P2"], ps["P0"], ps["nmodes"], edges)
        ratios.append(q / validate.kaiser_ratio(f / b))
    budget = (k_max * f * float(np.mean(psi_rms))) ** 2
    return ensemble(
        kb,
        ratios,
        require_se=False,
        extra={
            "eps": eps,
            "f_psi_rms": float(f * np.mean(psi_rms)),
            "nonlinear_budget_at_k_max": budget,
            "shot_over_signal_at_k_max": float(1.0 / nbar / (b * b * scaled(k_max))),
        },
    )


# ---------------------------------------------------------------------- f_NL ratio


def gate_fnl_ratio(
    spectrum,
    b,
    box: Box,
    box_est: Box,
    nbar,
    seeds,
    png,
    *,
    jitter_p=1,
    k_max=None,
    n_min=1,
):
    """Matched seeds: per realization, the real-space shot-subtracted monopole of the
    catalog with `png` (an `fnl.LocalPNG`) over the one without, from the SAME ic and
    draw seeds, divided by the predicted `<b(k)^2 P R> / <b^2 P R>` (R the estimator
    response, mode-weighted). Target 1 on every band to `k_max` (default half the
    generator Nyquist); bands hold >= `n_min` independent modes (default 1: one
    kf-shell each). A consistency check (|z| < Z_MAX, no SE floor) of the run path;
    the exact statement is `gate_field_identity(..., fnl=)`. The lowest shells stay
    noisy: their lognormal power includes a white term from high-k modes."""
    k_max = 0.5 * box.k_nyq if k_max is None else k_max
    edges = band_edges(box_est, k_max, n_min)
    shot = validate.shot_noise_k(box_est, nbar)
    R = validate.estimator_response(box, box_est, jitter_p)
    num = validate.shell_average(
        pk_on_grid(_fnl.galaxy_spectrum(spectrum, b, png), box_est) * R, box_est
    )
    den = validate.shell_average(
        pk_on_grid(_fnl.galaxy_spectrum(spectrum, b), box_est) * R, box_est
    )
    ratios, pred = [], None
    for s in seeds:
        ic, draw = sample.split_seed(s)
        P0 = []
        for p in (png, None):
            F = field.generate_fields(
                spectrum, b, box, ic, jitter_p=jitter_p, rsd=False, fnl=p
            )
            cat = sample.sample_catalog(F, nbar, draw, rsd=False)
            P0.append(
                validate.power_multipoles(
                    validate.delta_k_from_positions(cat.xyz, box_est),
                    box_est,
                    ells=(0,),
                    shot_k=shot,
                )
            )
        res, res0 = P0
        kb, meas, _ = band_ratio(res["k"], res["P0"], res0["P0"], res["nmodes"], edges)
        if pred is None:
            pred = band_ratio(res["k"], num, den, res["nmodes"], edges)[1]
        ratios.append(meas / pred)
    return ensemble(
        kb,
        ratios,
        require_se=False,
        extra={
            "predicted_ratio": pred.tolist(),
            "fnl": _fnl.diagnostics(spectrum, b, png, box),
        },
    )


# ------------------------------------------------------------------- shell density


def gate_shell_density(
    spectrum, b, f, box: Box, sh: shell.Shell, mask, nbar, seeds, *, n_radial_bins=8
):
    """Shell product at production amplitude: over seeds, the mean of
    `N_kept / (nbar fsky V_shell)` is 1, and so is the radial profile `n(r) / nbar` in
    `n_radial_bins` sub-shells (the edge sub-shells show a missing buffer or wrong RSD
    crossing). The seed scatter carries sample variance, so this is a consistency check
    (|z| < Z_MAX, no SE floor). Also returns the draws' Poisson z against
    `sum(lambda)` and the count of galaxies that left the box (never kept)."""
    fsky = 1.0 if mask is None else mask.fsky
    edges = np.linspace(sh.rmin, sh.rmax, n_radial_bins + 1)
    v_sub = 4.0 / 3.0 * np.pi * np.diff(edges**3) * fsky
    totals, profiles, z_draw, left = [], [], [], []
    for s in seeds:
        ic, draw = sample.split_seed(s)
        F = field.generate_fields(spectrum, b, box, ic, psi_axes="xyz")
        st = None
        for _, st in shell.sample_shell(
            F, sh, mask, nbar, draw, f=f, n_radial_bins=n_radial_bins
        ):
            pass
        totals.append(st.n_kept / (nbar * fsky * sh.volume))
        profiles.append(st.r_hist / (nbar * v_sub))
        z_draw.append((st.n_drawn - st.lam_window) / np.sqrt(st.lam_window))
        left.append(st.n_left_box)
    r_mid = 0.5 * (edges[1:] + edges[:-1])
    zd = np.asarray(z_draw)
    return {
        "total": ensemble(
            np.array([0.5 * (sh.rmin + sh.rmax)]),
            np.asarray(totals)[:, None],
            require_se=False,
        ),
        "profile": ensemble(r_mid, profiles, require_se=False),
        "draws": {
            "z_poisson_mean": float(zd.mean()),
            "z_poisson_std": float(zd.std(ddof=1)),
            "n_left_box_max": int(max(left)),
            "passed": bool(
                abs(zd.mean()) < Z_MAX / np.sqrt(len(seeds))
                and 0.5 < zd.std(ddof=1) < 1.6
            ),
        },
        "fsky": fsky,
    }
