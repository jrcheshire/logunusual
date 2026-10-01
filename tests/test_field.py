"""Field-stage contracts: colouring identity (G2), lognormal positivity and mean,
displacement against a plane wave and the divergence identity, target deconvolution,
in-process reproducibility."""

import platform

import numpy as np
import pytest

from logunusual import field, grid, pk, validate
from logunusual.grid import Box


def test_target_on_grid_is_deconvolved_power(spectrum):
    box = Box(16, 160.0)
    P0 = pk.pk_on_grid(spectrum, box)
    for p in (0, 1, 2):
        T = field.target_on_grid(spectrum, box, p)
        assert T[0, 0, 0] == 0.0
        assert np.allclose(T, P0 / grid.sinc_window(box, 2.0 * p))


def test_colouring_reproduces_pkG_in_the_ensemble():
    # Gaussian field: per-mode |G_k|^2 has variance P_G^2 over the INDEPENDENT
    # (half-grid)
    # modes, so the shell mean over n_seeds has SE = gaussian_se / sqrt(n_seeds).
    # Gate:
    # |z| < 4.5 everywhere and mean z^2 ~ 1 (measured 1.2 with 16 shells).
    box = Box(32, 320.0)
    _, _, kmag = grid.k_grid(box)
    pkG = 500.0 * np.exp(-((kmag * 8.0) ** 2) / 2)  # smooth, positive, DC nonzero
    pkG[0, 0, 0] = 0.0
    n_seeds = 48
    acc = None
    for s in range(n_seeds):
        wk = np.fft.rfftn(field.white_noise(box, 1000 + s))
        G = np.asarray(field.colour(wk, pkG, box))
        res = validate.field_power(G, box)
        acc = res["P0"] if acc is None else acc + res["P0"]
    mean = acc / n_seeds
    target = validate.shell_average(pkG, box)
    z = (mean - target) / (validate.gaussian_se(pkG, box) / np.sqrt(n_seeds))
    assert np.all(np.abs(z) < 4.5), z
    assert 0.4 < np.mean(z**2) < 1.8, np.mean(z**2)


def test_lognormal_has_zero_mean_and_is_bounded_below():
    import jax.numpy as jnp

    G = jnp.asarray(np.random.default_rng(3).standard_normal((8, 8, 8)) * 1.3)
    d = np.asarray(field.lognormal(G))
    assert abs(d.mean()) < 1e-14
    assert d.min() > -1.0


def test_displacement_of_a_plane_wave_is_analytic():
    # delta = A cos(k0 z)  ->  Psi_z = -(A / k0) sin(k0 z)  (so that dPsi_z/dz = -delta)
    import jax.numpy as jnp

    box = Box(16, 32.0)
    z = (np.arange(16) + 0.5) * box.dx
    k0 = 3 * box.k_f
    A = 0.4
    delta = np.broadcast_to(A * np.cos(k0 * z)[None, None, :], box.shape)
    psi = np.asarray(
        jnp.fft.irfftn(
            field.displacement_z_k(jnp.fft.rfftn(jnp.asarray(delta)), box), s=box.shape
        )
    )
    assert np.allclose(psi, -(A / k0) * np.sin(k0 * z)[None, None, :], atol=1e-12)
    # a wave along x displaces nothing along z
    delta_x = np.broadcast_to(A * np.cos(k0 * z)[:, None, None], box.shape)
    psi_x = np.asarray(
        jnp.fft.irfftn(
            field.displacement_z_k(jnp.fft.rfftn(jnp.asarray(delta_x)), box),
            s=box.shape,
        )
    )
    assert np.allclose(psi_x, 0.0, atol=1e-12)


def test_generate_fields_contract(spectrum):
    box = Box(16, 160.0)
    F = field.generate_fields(spectrum, 1.76, box, 21, keep_matter=True)
    assert F.delta_g.shape == box.shape and F.psi_z.shape == box.shape
    assert F.jitter_p == 1 and F.ic_seed == 21
    dg = np.asarray(F.delta_g)
    dm = np.asarray(F.delta_m)
    assert dg.min() > -1 and dm.min() > -1
    assert abs(dg.mean()) < 1e-13 and abs(dm.mean()) < 1e-13
    assert F.diagnostics["galaxy"]["n_clipped"] == 0
    assert F.diagnostics["galaxy"]["sigma2"] > F.diagnostics["matter"]["sigma2"]
    # the two fields share phases: strongly correlated cell by cell
    assert np.corrcoef(np.log1p(dg).ravel(), np.log1p(dm).ravel())[0, 1] > 0.9
    G = field.generate_fields(spectrum, 1.76, box, 21, rsd=False)
    assert G.psi_z is None and G.delta_m is None


def test_in_process_reproducibility_is_bitwise_or_characterised(spectrum):
    # Umbrella record: XLA on macOS-arm64 CPU can differ in the last bit between
    # runs; Linux XLA CPU is deterministic. Assert bitwise on Linux; on macOS record.
    box = Box(32, 320.0)
    a = np.asarray(field.generate_fields(spectrum, 1.76, box, 5).delta_g)
    b = np.asarray(field.generate_fields(spectrum, 1.76, box, 5).delta_g)
    n_diff = int(np.count_nonzero(a != b))
    max_rel = (
        float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))) if n_diff else 0.0
    )
    print(f"\nin-process repeat: {n_diff} differing cells, max rel {max_rel:.3e}")
    if platform.system() == "Linux":
        assert n_diff == 0
    else:
        assert max_rel < 1e-12  # characterised, not pinned


def test_displacement_components_satisfy_the_divergence_identity():
    # Psi_k = i k / k^2 delta_k  ->  sum_i k_i Psi_i,k = i delta_k on every mode but DC
    import jax.numpy as jnp

    from logunusual.grid import k_components

    box = Box(12, 60.0)
    delta = np.random.default_rng(4).standard_normal(box.shape)
    dk = jnp.fft.rfftn(jnp.asarray(delta))
    kx, ky, kz = k_components(box)
    lhs = (
        kx * np.asarray(field.displacement_k(dk, box, "x"))
        + ky * np.asarray(field.displacement_k(dk, box, "y"))
        + kz * np.asarray(field.displacement_k(dk, box, "z"))
    )
    rhs = 1j * np.asarray(dk)
    rhs[0, 0, 0] = 0.0
    assert np.allclose(lhs, rhs, atol=1e-10)
    # the z path is the M1 one, bitwise
    assert np.array_equal(
        np.asarray(field.displacement_k(dk, box, "z")),
        np.asarray(field.displacement_z_k(dk, box)),
    )
    with pytest.raises(ValueError):
        field.displacement_k(dk, box, "w")
    # a wave along x displaces along x only, analytically
    x = (np.arange(12) + 0.5) * box.dx
    k0, A = 2 * box.k_f, 0.3
    wave = np.broadcast_to(A * np.cos(k0 * x)[:, None, None], box.shape)
    wk = jnp.fft.rfftn(jnp.asarray(wave))
    psi_x = np.asarray(jnp.fft.irfftn(field.displacement_k(wk, box, "x"), s=box.shape))
    psi_y = np.asarray(jnp.fft.irfftn(field.displacement_k(wk, box, "y"), s=box.shape))
    assert np.allclose(psi_x, -(A / k0) * np.sin(k0 * x)[:, None, None], atol=1e-12)
    assert np.allclose(psi_y, 0.0, atol=1e-12)


def test_generate_fields_psi_axes(spectrum):
    box = Box(16, 160.0)
    F = field.generate_fields(spectrum, 1.5, box, 9, psi_axes="xyz")
    assert set(F.psi) == {"x", "y", "z"} and set(F.diagnostics["psi_rms"]) == {
        "x",
        "y",
        "z",
    }
    assert F.diagnostics["psi_z_rms"] == F.diagnostics["psi_rms"]["z"]
    flat = F.psi_flat("xyz")
    assert flat.shape == (box.n_cells, 3) and flat.dtype == np.float64
    assert np.array_equal(flat[:, 2], np.asarray(F.psi_z).reshape(-1))
    Z = field.generate_fields(spectrum, 1.5, box, 9, psi_axes="z")
    assert set(Z.psi) == {"z"} and np.array_equal(
        np.asarray(Z.psi_z), np.asarray(F.psi_z)
    )
    with pytest.raises(ValueError, match="displacement"):
        Z.psi_flat("xyz")
    for bad in ("zz", "q", "xyzx"):
        with pytest.raises(ValueError):
            field.generate_fields(spectrum, 1.5, box, 9, psi_axes=bad)


def test_zero_fnl_is_the_gaussian_stage_bitwise(spectrum):
    from logunusual.fnl import LocalPNG

    box = Box(32, 500.0)
    ref = field.generate_fields(spectrum, 1.76, box, 7, psi_axes="xyz")
    zero = field.generate_fields(
        spectrum, 1.76, box, 7, psi_axes="xyz", fnl=LocalPNG(f_nl=0.0)
    )
    np.testing.assert_array_equal(np.asarray(zero.delta_g), np.asarray(ref.delta_g))
    for a in "xyz":
        np.testing.assert_array_equal(np.asarray(zero.psi[a]), np.asarray(ref.psi[a]))
    # nonzero f_NL moves the galaxy field only; the matter field and Psi do not move
    up = field.generate_fields(
        spectrum, 1.76, box, 7, psi_axes="xyz", keep_matter=True, fnl=LocalPNG(10.0)
    )
    ref_m = field.generate_fields(spectrum, 1.76, box, 7, keep_matter=True)
    assert not np.array_equal(np.asarray(up.delta_g), np.asarray(ref.delta_g))
    np.testing.assert_array_equal(np.asarray(up.delta_m), np.asarray(ref_m.delta_m))
    np.testing.assert_array_equal(np.asarray(up.psi["z"]), np.asarray(ref.psi["z"]))
    assert up.diagnostics["fnl"]["b_kf_over_b"] > 1.0


def test_resolve_dtype_rejects_unknown():
    assert field.resolve_dtype("f64") == (np.float64, np.complex128)
    assert field.resolve_dtype("f32") == (np.float32, np.complex64)
    with pytest.raises(ValueError, match="dtype must be one of"):
        field.resolve_dtype("float32")


def test_f64_is_the_default_dtype(spectrum):
    # The knob must not move the default path. macOS XLA CPU can differ in the last
    # bit between two runs of the SAME program (see the reproducibility test above),
    # so this is bitwise on Linux and characterised on macOS -- the repo's
    # reproducibility gate is what pins the f64 path.
    box = Box(32, 320.0)
    a = np.asarray(field.generate_fields(spectrum, 1.76, box, 7).delta_g)
    b = np.asarray(field.generate_fields(spectrum, 1.76, box, 7, dtype="f64").delta_g)
    if platform.system() == "Linux":
        assert np.array_equal(a, b)
    else:
        assert float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))) < 1e-12


def test_f32_stays_f32_end_to_end(spectrum):
    # Every device array the stage returns must be float32. The trap this guards is
    # that `k_components` is numpy float64 and, with x64 enabled, one float64 array
    # in the displacement expression widens the whole of it back to complex128.
    box = Box(16, 160.0)
    F = field.generate_fields(
        spectrum, 1.76, box, 21, keep_matter=True, psi_axes="xyz", dtype="f32"
    )
    assert F.dtype == "f32"
    assert F.delta_g.dtype == np.float32
    assert F.delta_m.dtype == np.float32
    for axis in "xyz":
        assert F.psi[axis].dtype == np.float32, axis
    # the sampler's interface is float64 whatever the field stage ran in
    assert F.psi_flat("xyz").dtype == np.float64


def test_f32_computes_the_same_field_as_f64(spectrum):
    # f32 is a footprint knob, not a different mock: same white noise, same target,
    # so the fields must agree to float32 round-off. The scale of that round-off is
    # eps32 times the LARGEST magnitude in the chain, not the rms -- a lognormal's
    # peak sits 30-100x its rms, so an rms-normalised tolerance would just be
    # measuring that ratio. Measured 2-22 eps32 at (N, L) = (32, 320), (64, 640) and
    # (32, 1000), i.e. sigma2 from 0.6 to 3.0, and it does not grow with N; the gate
    # is 100 eps32, which round-off through three FFTs and an exp cannot exceed.
    box = Box(32, 320.0)
    kw = dict(keep_matter=True, psi_axes="xyz")
    A = field.generate_fields(spectrum, 1.76, box, 11, **kw)
    B = field.generate_fields(spectrum, 1.76, box, 11, dtype="f32", **kw)
    eps32 = float(np.finfo(np.float32).eps)
    for name, a, b in [
        ("delta_g", A.delta_g, B.delta_g),
        ("delta_m", A.delta_m, B.delta_m),
        *[(f"psi_{x}", A.psi[x], B.psi[x]) for x in "xyz"],
    ]:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        ulps = float(np.max(np.abs(a - b))) / (eps32 * float(np.max(np.abs(a))))
        print(f"\n   {name}: max |f32 - f64| = {ulps:.1f} eps32 of max|field|")
        assert ulps < 100.0, (name, ulps)


def test_jit_is_not_bit_preserving_but_is_round_off(spectrum):
    # M3 asks for jit WITH a bitwise-before-jit check. It is not bitwise: XLA fuses
    # and reassociates, so a third to seven eighths of the cells move. What the gate
    # holds is that the move is round-off and nothing else -- 0.25-4.0 float64 eps of
    # the field's own maximum, measured over 8 seeds at N = 64 and 96 (the per-CELL
    # relative figure runs to 1e-10 and higher, but that is the metric blowing up at
    # zero crossings, the same artefact the ULP table warns about). Gate: 20 eps64.
    box = Box(64, 640.0)
    kw = dict(keep_matter=True, psi_axes="xyz")
    A = field.generate_fields(spectrum, 1.76, box, 3, **kw)
    B = field.generate_fields(spectrum, 1.76, box, 3, jit=True, **kw)
    assert A.jit is False and B.jit is True
    eps64 = float(np.finfo(np.float64).eps)
    n_moved = 0
    for name, a, b in [
        ("delta_g", A.delta_g, B.delta_g),
        ("delta_m", A.delta_m, B.delta_m),
        *[(f"psi_{x}", A.psi[x], B.psi[x]) for x in "xyz"],
    ]:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n_moved += int(np.count_nonzero(a != b))
        ulps = float(np.max(np.abs(a - b))) / (eps64 * float(np.max(np.abs(a))))
        print(f"\n   {name}: jit vs eager {ulps:.2f} eps64 of max|field|")
        assert ulps < 20.0, (name, ulps)
    assert n_moved > 0, "jit was bit-preserving here -- the claim above needs redoing"
    # and the field is still a lognormal overdensity
    assert float(np.asarray(B.delta_g).min()) > -1.0
    assert abs(float(np.asarray(B.delta_g).mean())) < 1e-13
