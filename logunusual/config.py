"""Run configuration: which bins, which inputs, where the output goes.

A run is a list of bins (each a `suite.Bin`: shell edges, box, grid, density, bias,
growth rate, input P(k) file), an optional angular mask, a seed base, and a few knobs.
The default bin list is `suite.BIN_SUITE_V28`; a YAML file can give its own `bins`
with the same fields. The config hash covers the mock's definition (bins and knobs),
not machine paths; input files are hashed separately into the catalog metadata.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from . import suite
from .grid import Box
from .shell import Shell

BIN_FIELDS = tuple(suite.Bin.__dataclass_fields__)


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    output_dir: Path
    pk_dir: Path
    bins: tuple  # of suite.Bin, indices ascending and unique
    mask_path: Path | None = None
    mask_dataset: str = suite.MASK_DATASET
    seed_base: int = suite.SEED_BASE
    nbar_scale: float = 1.0
    grid_scale: float = 1.0
    radial_buffer: float = suite.RADIAL_BUFFER
    jitter_p: int = 1
    n_workers: int | None = None  # threads for the shell sampler; None = core count
    angular_precut: bool = True  # draw only cells that can feed the masked shell
    row_group_rows: int = 2**20
    n_radial_bins: int = 8

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "pk_dir", Path(self.pk_dir))
        if self.mask_path is not None:
            object.__setattr__(self, "mask_path", Path(self.mask_path))
        bins = tuple(self.bins)
        if not bins:
            raise ValueError("bins must not be empty")
        idx = [b.index for b in bins]
        if idx != sorted(idx) or len(set(idx)) != len(idx):
            raise ValueError(f"bin indices must be ascending and unique: {idx}")
        object.__setattr__(self, "bins", bins)
        if self.nbar_scale <= 0 or self.grid_scale <= 0:
            raise ValueError("nbar_scale and grid_scale must be positive")
        if self.jitter_p < 1:
            raise ValueError("jitter_p must be >= 1")
        if self.row_group_rows < 1:
            raise ValueError("row_group_rows must be >= 1")
        if self.n_workers is not None and self.n_workers < 1:
            raise ValueError("n_workers must be >= 1 or null")

    # ---------------------------------------------------------------- effective bins
    def effective_bin(self, b: suite.Bin) -> suite.Bin:
        """The bin as run: `N_grid` scaled by `grid_scale` (rounded to an even number,
        >= 4) and `nbar` by `nbar_scale`."""
        n = max(4, int(round(b.N_grid * self.grid_scale)))
        n += n % 2
        return replace(b, N_grid=n, nbar=b.nbar * self.nbar_scale)

    def box(self, b: suite.Bin) -> Box:
        e = self.effective_bin(b)
        return Box(e.N_grid, float(e.L_box))

    def shell(self, b: suite.Bin) -> Shell:
        return Shell(float(b.rmin), float(b.rmax), float(self.radial_buffer))

    def pk_path(self, b: suite.Bin) -> Path:
        return self.pk_dir / b.pk_file

    def bins_by_index(self, indices=None):
        if indices is None:
            return self.bins
        want = set(int(i) for i in indices)
        have = {b.index for b in self.bins}
        if not want <= have:
            raise ValueError(f"unknown bin indices {sorted(want - have)}; have {have}")
        return tuple(b for b in self.bins if b.index in want)

    # -------------------------------------------------------------------- (de)serial
    def to_dict(self) -> dict:
        d = {
            "run_name": self.run_name,
            "output_dir": str(self.output_dir),
            "pk_dir": str(self.pk_dir),
            "mask": (
                None
                if self.mask_path is None
                else {"path": str(self.mask_path), "dataset": self.mask_dataset}
            ),
            "seed_base": self.seed_base,
            "nbar_scale": self.nbar_scale,
            "grid_scale": self.grid_scale,
            "radial_buffer": self.radial_buffer,
            "jitter_p": self.jitter_p,
            "n_workers": self.n_workers,
            "angular_precut": self.angular_precut,
            "row_group_rows": self.row_group_rows,
            "n_radial_bins": self.n_radial_bins,
            "bins": [asdict(b) for b in self.bins],
        }
        return d

    def hash_dict(self) -> dict:
        """What the config hash covers: everything but names and machine paths."""
        d = self.to_dict()
        for k in ("run_name", "output_dir", "pk_dir", "mask", "n_workers"):
            d.pop(k)
        d["mask"] = self.mask_path is not None
        return d

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.hash_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        unknown = set(d) - {
            "run_name",
            "output_dir",
            "pk_dir",
            "mask",
            "seed_base",
            "nbar_scale",
            "grid_scale",
            "radial_buffer",
            "jitter_p",
            "n_workers",
            "angular_precut",
            "row_group_rows",
            "n_radial_bins",
            "bins",
        }
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        bins_in = d.pop("bins", None)
        if bins_in is None:
            bins = suite.BIN_SUITE_V28
        else:
            bins = tuple(_bin_from_dict(x) for x in bins_in)
        mask = d.pop("mask", None)
        kw = {"bins": bins}
        if mask is not None:
            if isinstance(mask, str):
                mask = {"path": mask}
            kw["mask_path"] = mask["path"]
            kw["mask_dataset"] = mask.get("dataset", suite.MASK_DATASET)
        for k in ("run_name", "output_dir", "pk_dir"):
            if k not in d:
                raise ValueError(f"config needs {k!r}")
        kw.update(d)
        return cls(**kw)

    @classmethod
    def from_yaml(cls, path) -> "RunConfig":
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=None)


def _bin_from_dict(x: dict) -> suite.Bin:
    unknown = set(x) - set(BIN_FIELDS)
    if unknown:
        raise ValueError(f"unknown bin keys {sorted(unknown)}; fields are {BIN_FIELDS}")
    missing = [k for k in BIN_FIELDS if k not in x and k != "pk_file"]
    if missing:
        raise ValueError(f"bin {x.get('name', '?')} misses {missing}")
    return suite.Bin(**x)


def default_config(
    run_name="v28", output_dir="runs", pk_dir="data", mask=None
) -> RunConfig:
    """The seven-bin v28 default with placeholder paths."""
    return RunConfig(
        run_name=run_name,
        output_dir=Path(output_dir),
        pk_dir=Path(pk_dir),
        bins=suite.BIN_SUITE_V28,
        mask_path=None if mask is None else Path(mask),
    )
