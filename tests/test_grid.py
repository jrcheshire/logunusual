import math

import numpy as np
import pytest

from logunusual import grid
from logunusual.grid import Box


def test_box_geometry():
    b = Box(64, 1000.0)
    assert b.dx == 1000.0 / 64
    assert math.isclose(b.v_cell, b.dx**3)
    assert math.isclose(b.k_f, 2 * math.pi / 1000.0)
    assert math.isclose(b.k_nyq, math.pi / b.dx)
    assert b.shape == (64, 64, 64) and b.rfft_shape == (64, 64, 33)
    with pytest.raises(ValueError):
        Box(63, 1.0)
    with pytest.raises(ValueError):
        Box(64, 0.0)


def test_k_grid_matches_fft_conventions():
    b = Box(16, 32.0)
    k1, kz, kmag = grid.k_grid(b)
    assert kmag.shape == b.rfft_shape
    assert k1[0] == 0.0 and kz[-1] == pytest.approx(b.k_nyq)
    assert kmag[0, 0, 0] == 0.0
    # the corner of the rfft grid is sqrt(3) k_nyq
    assert kmag[8, 8, 8] == pytest.approx(math.sqrt(3) * b.k_nyq)
    kx, ky, kz3 = grid.k_components(b)
    assert np.allclose(np.sqrt(kx**2 + ky**2 + kz3**2), kmag)


def test_hermitian_weights_count_the_full_grid():
    b = Box(16, 1.0)
    w = np.broadcast_to(grid.hermitian_weights(b), b.rfft_shape)
    assert w.sum() == b.n_cells


def test_sinc_window_values():
    b = Box(16, 16.0)
    W = grid.sinc_window(b, 2.0)
    assert W[0, 0, 0] == 1.0
    # one axis at Nyquist: sinc(1/2)^2 = (2/pi)^2
    assert W[8, 0, 0] == pytest.approx((2 / math.pi) ** 2)
    assert W[0, 0, 8] == pytest.approx((2 / math.pi) ** 2)
    assert np.all(W > 0)
    assert np.allclose(grid.jitter_power_window(b, 1), W)
    assert np.allclose(grid.jitter_power_window(b, 2), W**2)
    assert np.allclose(grid.sinc_window(b, -2.0), 1.0 / W)


def test_cic_shot_noise_factor_limits():
    b = Box(16, 16.0)
    C = grid.cic_shot_noise_factor(b)
    assert C[0, 0, 0] == 1.0
    assert C[8, 0, 0] == pytest.approx(1.0 - 2.0 / 3.0)  # one axis at Nyquist
    assert np.all((C > 0) & (C <= 1))
