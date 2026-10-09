"""Config, driver and CLI contracts on a toy two-bin suite: YAML round trip and hash
scope, end-to-end realizations with a mask, the exact Poisson gate through the whole
path, skip/overwrite, cross-process reproducibility, CLI commands."""

from dataclasses import asdict
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from logunusual import cli, io, suite
from logunusual.config import RunConfig, default_config
from logunusual.run import catalog_path, generate_realization, plan_realization

DATA = Path(__file__).parent / "data"
PK = "matterpower_camb_zeff=0.9.tsv"

TOY_BINS = [
    dict(
        name="t1",
        index=1,
        z_min=0.0,
        z_max=0.1,
        z_eff=0.9,
        rmin=0.0,
        rmax=40.0,
        L_box=170.0,
        N_grid=16,
        nbar=1e-2,
        b=1.5,
        f=0.85,
        pk_file=PK,
    ),
    dict(
        name="t2",
        index=2,
        z_min=0.1,
        z_max=0.2,
        z_eff=0.9,
        rmin=60.0,
        rmax=100.0,
        L_box=300.0,
        N_grid=24,
        nbar=4e-3,
        b=1.8,
        f=0.85,
        pk_file=PK,
    ),
]  # dx ~ 10.6 / 12.5; buffer 45 covers half-diagonal + f max|Psi| (~31 measured)


def _mask(path, nside=8):
    import h5py
    import healpy as hp

    m = np.zeros(hp.nside2npix(nside), np.int8)
    th, _ = hp.pix2ang(nside, np.arange(m.size), nest=True)
    m[np.cos(th) > -0.3] = 1
    with h5py.File(path, "w") as f:
        f.attrs["PIXTYPE"] = "HEALPIX"
        f.attrs["ORDERING"] = "NESTED"
        f.create_dataset("MASK", data=m)
    return m.mean()


def _cfg(tmp_path, pk_dir=DATA, **kw):
    fsky = _mask(tmp_path / "mask.h5")
    d = dict(
        run_name="toy",
        output_dir=str(tmp_path / "runs"),
        pk_dir=str(pk_dir),
        mask={"path": str(tmp_path / "mask.h5")},
        bins=TOY_BINS,
        radial_buffer=45.0,
        row_group_rows=1000,
    )
    d.update(kw)
    return RunConfig.from_dict(d), fsky


# ------------------------------------------------------------------------ config


def test_config_yaml_round_trip_and_hash_scope(tmp_path):
    cfg, _ = _cfg(tmp_path)
    (tmp_path / "c.yaml").write_text(cfg.to_yaml())
    back = RunConfig.from_yaml(tmp_path / "c.yaml")
    assert back == cfg and back.config_hash == cfg.config_hash
    # the hash covers the mock, not names or paths
    other, _ = _cfg(tmp_path, run_name="other", output_dir=str(tmp_path / "x"))
    assert other.config_hash == cfg.config_hash
    assert _cfg(tmp_path, nbar_scale=0.5)[0].config_hash != cfg.config_hash
    assert _cfg(tmp_path, mask=None)[0].config_hash != cfg.config_hash
    # defaults: the v28 table
    d = default_config()
    assert d.bins == suite.BIN_SUITE_V28 and d.seed_base == suite.SEED_BASE
    assert RunConfig.from_dict(
        {"run_name": "a", "output_dir": "o", "pk_dir": "p"}
    ).bins == (suite.BIN_SUITE_V28)
    with pytest.raises(ValueError, match="unknown config keys"):
        RunConfig.from_dict(
            {"run_name": "a", "output_dir": "o", "pk_dir": "p", "zzz": 1}
        )
    with pytest.raises(ValueError, match="ascending"):
        _cfg(tmp_path, bins=[TOY_BINS[1], TOY_BINS[0]])
    with pytest.raises(ValueError, match="unknown bin indices"):
        cfg.bins_by_index([3])


def test_fnl_config_keys_and_hash_scope(tmp_path):
    # the worked v28 config's hash is pinned: the f_NL keys do not enter it at
    # f_NL = 0. It last changed with the 160 Mpc/h buffer (M5); before that 719cfb12...
    # (flat-LCDM growth rates, M5) and f9455b54... (the M3 seven-bin catalogs).
    v28 = RunConfig.from_yaml(Path(__file__).parents[1] / "configs/v28_default.yaml")
    assert v28.config_hash == (
        "29242829ace44f7c916ea9a3f0acfdaf880ed421fba4bef59ee14b8520a1e651"
    )
    assert v28.png is None
    cfg, _ = _cfg(tmp_path)
    fnl_cfg, _ = _cfg(tmp_path, f_nl=10.0, primordial={"n_s": 0.96})
    assert fnl_cfg.config_hash != cfg.config_hash
    assert _cfg(tmp_path, f_nl=10.0, fnl_p=1.6)[0].config_hash != fnl_cfg.config_hash
    # at f_NL = 0 the other f_NL keys do not move the hash
    assert _cfg(tmp_path, fnl_p=1.6)[0].config_hash == cfg.config_hash
    (tmp_path / "f.yaml").write_text(fnl_cfg.to_yaml())
    back = RunConfig.from_yaml(tmp_path / "f.yaml")
    assert back == fnl_cfg and back.config_hash == fnl_cfg.config_hash
    png = back.png
    assert png.f_nl == 10.0 and png.n_s == 0.96 and png.A_s == suite.PRIMORDIAL_AS
    with pytest.raises(ValueError, match="unknown primordial keys"):
        _cfg(tmp_path, f_nl=1.0, primordial={"sigma8": 0.8})


def test_fnl_realization_end_to_end(tmp_path):
    cfg, _ = _cfg(tmp_path, f_nl=10.0)
    s = generate_realization(cfg, 0, log=lambda *a: None)
    p = catalog_path(cfg, 0)
    assert io.check_layout(p).ok
    meta = io.read_metadata(p)
    assert float(meta["f_nl"]) == 10.0 and meta["fnl_convention"] == "LSS"
    assert float(meta["fnl_A_s"]) == suite.PRIMORDIAL_AS
    for b, row in zip(cfg.bins, s["bins"]):
        m = io.bin_metadata(meta, b.index)
        assert float(m["fnl_delta_b_kf"]) == row["fnl_delta_b_kf"] > 0
    # the Gaussian run writes f_nl = 0 and nothing else of the f_NL block
    g, _ = _cfg(tmp_path, output_dir=str(tmp_path / "g"))
    generate_realization(g, 0, log=lambda *a: None)
    gm = io.read_metadata(catalog_path(g, 0))
    assert float(gm["f_nl"]) == 0.0 and "fnl_convention" not in gm


def test_effective_bin_scaling(tmp_path):
    cfg, _ = _cfg(tmp_path, grid_scale=0.3, nbar_scale=0.1)
    e = cfg.effective_bin(cfg.bins[1])
    assert e.N_grid == 8 and e.N_grid % 2 == 0  # round(24 * 0.3) = 7 -> 8
    assert e.nbar == pytest.approx(4e-4)
    assert cfg.box(cfg.bins[1]).n_mesh == 8
    rows = plan_realization(cfg, 3)
    assert [r["seed"] for r in rows] == [suite.seed_for(3, 1), suite.seed_for(3, 2)]
    assert rows[0]["ic_seed"] != rows[0]["draw_seed"]


# ------------------------------------------------------------------------ driver


def test_generate_realization_end_to_end(tmp_path):
    cfg, fsky = _cfg(tmp_path)
    s = generate_realization(cfg, 0, log=lambda *a: None)
    p = catalog_path(cfg, 0)
    assert p.exists() and Path(s["path"]) == p
    rep = io.check_layout(p)
    assert rep.ok, rep.problems
    assert rep.bins == [1, 2]
    meta = io.read_metadata(p)
    assert meta["generator"] == "logunusual" and meta["bins"] == "1,2"
    assert meta["config_hash"] == cfg.config_hash
    assert float(meta["mask_fsky"]) == pytest.approx(fsky)
    assert meta["nbar_overdensity_factor"] == "1.0" and meta["rsd"] == "radial"
    for b in cfg.bins:
        m = io.bin_metadata(meta, b.index)
        assert int(m["n_galaxies"]) == rep.rows_per_bin[b.index] > 0
        assert float(m["nbar_target"]) == b.nbar
        assert m["pk_file"] == PK and len(m["pk_sha256"]) == 64
        assert json.loads(m["psi_rms"]).keys() == {"x", "y", "z"}
        d = io.read_bin(p, b.index)
        r = np.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
        assert r.min() >= b.rmin and r.max() <= b.rmax
    summary = json.loads((p.parent / "summary.json").read_text())
    assert summary["n_galaxies"] == rep.n_rows and len(summary["bins"]) == 2
    # skip unless overwrite; a second realization is a different draw
    assert generate_realization(cfg, 0, log=lambda *a: None) is None
    generate_realization(cfg, 1, log=lambda *a: None)
    assert catalog_path(cfg, 1).read_bytes() != p.read_bytes()
    # a bin subset
    generate_realization(cfg, 2, bins=[2], log=lambda *a: None)
    assert io.check_layout(catalog_path(cfg, 2)).bins == [2]


def test_uniform_input_gives_poisson_counts_through_the_driver(tmp_path):
    # P_in x 1e-4 written as a TSV: the whole path (window, RSD, mask, writer) must
    # return N_kept ~ Poisson(nbar fsky V_shell) per bin, exactly.
    k, P = np.loadtxt(DATA / PK, unpack=True)
    pk_dir = tmp_path / "pk"
    pk_dir.mkdir()
    np.savetxt(pk_dir / PK, np.c_[k, 1e-4 * P], header="kh Pk")
    cfg, fsky = _cfg(tmp_path, pk_dir=pk_dir)
    s = generate_realization(cfg, 0, log=lambda *a: None)
    for row, b in zip(s["bins"], cfg.bins):
        mean = b.nbar * fsky * 4 / 3 * np.pi * (b.rmax**3 - b.rmin**3)
        z = (row["n_galaxies"] - mean) / np.sqrt(mean)
        assert abs(z) < 4.0, (b.name, z)


def test_two_processes_write_the_same_bytes(tmp_path):
    # Two fresh processes, one sampler thread against four: same catalog.
    cfg, _ = _cfg(tmp_path)
    outs = []
    for tag, workers in (("a", 1), ("b", 4)):
        c = RunConfig(**{**cfg.__dict__, "n_workers": workers})
        assert c.config_hash == cfg.config_hash  # threads are not part of the mock
        (tmp_path / f"c_{tag}.yaml").write_text(c.to_yaml())
        subprocess.run(
            [
                sys.executable,
                "-m",
                "logunusual.cli",
                "run",
                "--config",
                str(tmp_path / f"c_{tag}.yaml"),
                "--realizations",
                "0",
                "--output-dir",
                str(tmp_path / tag),
            ],
            check=True,
            capture_output=True,
        )
        outs.append((tmp_path / tag / "toy" / "realization_00000" / "catalog.parq"))
    a, b = (io.read_bin(o, 1) for o in outs)
    if platform.system() == "Linux":
        assert outs[0].read_bytes() == outs[1].read_bytes()
    else:  # macOS XLA can wobble in the last bit: characterised, not pinned
        assert a["x"].size == b["x"].size
        assert np.max(np.abs(a["x"] - b["x"])) < 1e-9


# --------------------------------------------------------------------------- cli


def test_cli_commands(tmp_path, capsys):
    cfg, _ = _cfg(tmp_path)
    (tmp_path / "c.yaml").write_text(cfg.to_yaml())
    assert cli.parse_realizations("2:5") == [2, 3, 4] and cli.parse_realizations(
        "7"
    ) == [7]
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(tmp_path / "c.yaml"),
                "--realizations",
                "0:2",
                "--dry-run",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "realization 00001" in out and "seed=" in out
    assert not (tmp_path / "runs").exists()
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(tmp_path / "c.yaml"),
                "--realizations",
                "0",
                "--bins",
                "1",
            ]
        )
        == 0
    )
    p = catalog_path(cfg, 0)
    assert cli.main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert "layout: ok" in out and "bins [1]" in out
    assert cli.main(["default-config"]) == 0
    import yaml

    d = yaml.safe_load(capsys.readouterr().out)
    assert len(d["bins"]) == 7 and d["seed_base"] == suite.SEED_BASE


# ----------------------------------------------------------------- two P(k) tables


def _two_table_dir(tmp_path, factor):
    """`pk_dir` with the linear test table and `gal.tsv` = `P * factor(k)`."""
    import shutil

    d = tmp_path / "pk"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / PK, d / PK)
    k, P = np.loadtxt(DATA / PK, comments="#").T
    np.savetxt(d / "gal.tsv", np.c_[k, P * factor(k)], header="kh\tPk", comments="# ")
    return d


def _with_galaxy_table(name):
    return [dict(b, pk_galaxy_file=name) for b in TOY_BINS]


SET = ("pk_file", "pk_galaxy_file")  # what the halofit config sets


def test_galaxy_table_config_and_hash_scope(tmp_path):
    cfg, _ = _cfg(tmp_path)
    two, _ = _cfg(tmp_path, bins=_with_galaxy_table("gal.tsv"))
    assert two.config_hash != cfg.config_hash
    assert two.pk_galaxy_path(two.bins[0]).name == "gal.tsv"
    assert cfg.pk_galaxy_path(cfg.bins[0]) == cfg.pk_path(cfg.bins[0])
    # unset, the field neither serialises nor hashes
    assert all("pk_galaxy_file" not in b for b in cfg.to_dict()["bins"])
    (tmp_path / "t.yaml").write_text(two.to_yaml())
    back = RunConfig.from_yaml(tmp_path / "t.yaml")
    assert back == two and back.config_hash == two.config_hash
    # the worked halofit config: the v28 bins, grids included, with two tables
    hf = RunConfig.from_yaml(Path(__file__).parents[1] / "configs/v28_halofit.yaml")
    assert all("lin" in b.pk_file for b in hf.bins)
    assert all("halofit" in b.pk_galaxy_file for b in hf.bins[:6])
    assert hf.bins[6].pk_galaxy_file == ""  # bin 7 clips its fundamental either way
    for b, v in zip(hf.bins, suite.BIN_SUITE_V28):
        keep = lambda x: {k: y for k, y in asdict(x).items() if k not in SET}  # noqa
        assert keep(b) == keep(v)


def test_galaxy_table_realization_end_to_end(tmp_path):
    pk_dir = _two_table_dir(tmp_path, lambda k: 1 + (k / 0.5) ** 2)
    cfg, _ = _cfg(tmp_path, pk_dir=pk_dir, bins=_with_galaxy_table("gal.tsv"))
    generate_realization(cfg, 0, log=lambda *a: None)
    p = catalog_path(cfg, 0)
    assert io.check_layout(p).ok
    meta = io.read_metadata(p)
    gal_sha = hashlib.sha256((pk_dir / "gal.tsv").read_bytes()).hexdigest()
    for b in cfg.bins:
        m = io.bin_metadata(meta, b.index)
        assert m["pk_file"] == PK and m["pk_galaxy_file"] == "gal.tsv"
        assert m["pk_galaxy_sha256"] == gal_sha != m["pk_sha256"]
    # single-table runs stamp the linear table as the galaxy one
    g, _ = _cfg(tmp_path, output_dir=str(tmp_path / "g"))
    generate_realization(g, 0, log=lambda *a: None, bins=[1])
    m = io.bin_metadata(io.read_metadata(catalog_path(g, 0)), 1)
    assert m["pk_galaxy_file"] == PK and m["pk_galaxy_sha256"] == m["pk_sha256"]
    # a galaxy table from another z / cosmology is refused at load
    bad_dir = _two_table_dir(tmp_path / "bad", lambda k: 0.8 + 0 * k)
    bad, _ = _cfg(
        tmp_path,
        pk_dir=bad_dir,
        bins=_with_galaxy_table("gal.tsv"),
        output_dir=str(tmp_path / "b"),
    )
    with pytest.raises(ValueError, match="not the same z"):
        generate_realization(bad, 0, log=lambda *a: None)
