"""The default bin table, seed schedule and constants (dependency-free).

`BIN_SUITE_V28` is an example seven-bin table built from the public SPHEREx forecast
products (`galaxy_density_v28_base_cbe.txt` in
https://github.com/SPHEREx/Public-products): z = 0-2.2 split into shells, one
observer-centred periodic box per shell; a run config can give any other table with
the same fields. `rmin`, `rmax`: comoving edges (Mpc/h) in FlatLambdaCDM(H0=67.36,
Om0=0.3153), the cosmology of the CAMB P(k) tables. `nbar`: density summed over the
five sigma_z/(1+z) < 0.2 samples; `b`: their number-weighted mean bias. `L_box` holds
`2 (rmax + RADIAL_BUFFER)`; cells are ~8-16 Mpc/h. `f`:
`fnl.growth_rate_md(z_eff, OMEGA_M_DISTANCE)`, stored as literals.
"""

from dataclasses import dataclass

#: `seed = SEED_BASE + realization * 1000 + bin_index` (`seed_for`).
SEED_BASE = 137_000_000
SEED_REALIZATION_STRIDE = 1000

#: Buffer (Mpc/h) on both sides of each shell: covers `shell.Shell.required_buffer`
#: over 100 default realizations, leaving >= 12 Mpc/h of box room in every bin.
RADIAL_BUFFER = 160.0

#: Distance cosmology behind rmin/rmax and the matter-power tables.
H0_DISTANCE = 67.36
OMEGA_M_DISTANCE = 0.3153

#: Primordial normalisation of the CAMB tables (Planck 2018); `fnl.LocalPNG` needs it
#: to recover M(k) from a table.
PRIMORDIAL_AS = 2.1e-9
PRIMORDIAL_NS = 0.9649
PRIMORDIAL_K_PIVOT = 0.05 / (H0_DISTANCE / 100.0)  # CAMB's 0.05 / Mpc, in h/Mpc

#: Spherical-collapse threshold in the f_NL bias response.
DELTA_C = 1.686

#: Survey mask contract: HEALPix NSIDE 128, NESTED, int8 dataset `MASK`, fsky 0.7127.
MASK_NSIDE = 128
MASK_ORDERING = "NESTED"
MASK_DATASET = "MASK"


@dataclass(frozen=True)
class Bin:
    name: str
    index: int  # 1-based; this is the value written to the parquet `bin` column
    z_min: float
    z_max: float
    z_eff: float
    rmin: float  # Mpc/h
    rmax: float  # Mpc/h
    L_box: float  # Mpc/h
    N_grid: int
    nbar: float  # (Mpc/h)^-3, v28 nominal
    b: float
    f: float  # linear growth rate at z_eff (see module docstring)
    pk_file: str = ""  # input P(k) TSV name (relative to the run's pk_dir)
    # nonlinear galaxy-target table; empty = `pk_file` (always the linear table)
    pk_galaxy_file: str = ""

    def __post_init__(self):
        if not self.pk_file:
            object.__setattr__(
                self, "pk_file", f"matterpower_camb_zeff={self.z_eff:g}.tsv"
            )
        if not (0 < self.index < 128):
            raise ValueError(f"bin index must be in 1..127 (int8), got {self.index}")
        if not (0.0 <= self.rmin < self.rmax):
            raise ValueError(f"need 0 <= rmin < rmax for {self.name}")
        if self.N_grid < 2 or self.N_grid % 2:
            raise ValueError(f"N_grid must be even and >= 2 for {self.name}")

    @property
    def cell(self) -> float:
        """Grid spacing in Mpc/h."""
        return self.L_box / self.N_grid

    @property
    def k_nyquist(self) -> float:
        """pi / cell, in h/Mpc."""
        import math

        return math.pi / self.cell

    @property
    def matterpower_file(self) -> str:
        """`data/<pk_file>`: the scripts' default TSV path in a checkout."""
        return f"data/{self.pk_file}"


# `fnl.growth_rate_md(z_eff, OMEGA_M_DISTANCE)`, pinned by tests/test_suite.py.
_F_FLAT_LCDM = {
    0.10: 0.585721401221273,
    0.30: 0.6849019867378581,
    0.50: 0.7612297870023778,
    0.70: 0.8181973393109712,
    0.90: 0.8602004951231208,
    1.30: 0.9141354303816462,
    1.90: 0.9544897124652842,
}

# fmt: off
# name, index, z_min, z_max, z_eff, rmin, rmax, L_box, N_grid, nbar, b
_ROWS = (
    ("bin01", 1, 0.0, 0.2, 0.10,    0.000,  568.215, 1500, 192, 7.347e-2, 1.03),
    ("bin02", 2, 0.2, 0.4, 0.30,  568.215, 1077.778, 2500, 256, 4.166e-2, 1.32),
    ("bin03", 3, 0.4, 0.6, 0.50, 1077.778, 1530.733, 3500, 384, 1.556e-2, 1.45),
    ("bin04", 4, 0.6, 0.8, 0.70, 1530.733, 1931.840, 4500, 448, 1.140e-2, 1.57),
    ("bin05", 5, 0.8, 1.0, 0.90, 1931.840, 2287.330, 5000, 512, 8.482e-3, 1.76),
    ("bin06", 6, 1.0, 1.6, 1.30, 2287.330, 3139.330, 7000, 512, 1.808e-3, 2.22),
    ("bin07", 7, 1.6, 2.2, 1.90, 3139.330, 3764.808, 8000, 512, 3.491e-4, 3.29),
)
# fmt: on

BIN_SUITE_V28 = tuple(Bin(*row, f=_F_FLAT_LCDM[row[4]]) for row in _ROWS)


def seed_for(realization: int, bin_index: int, base: int = SEED_BASE) -> int:
    """`base + realization * 1000 + bin_index`; both arguments are bounded by the
    stride so two (realization, bin) pairs never share a seed."""
    if not 0 <= realization < SEED_REALIZATION_STRIDE:
        raise ValueError(
            f"realization must be in [0, {SEED_REALIZATION_STRIDE}), got {realization}"
        )
    if not 0 < bin_index < SEED_REALIZATION_STRIDE:
        raise ValueError(
            f"bin_index must be in (0, {SEED_REALIZATION_STRIDE}), got {bin_index}"
        )
    return base + realization * SEED_REALIZATION_STRIDE + bin_index
