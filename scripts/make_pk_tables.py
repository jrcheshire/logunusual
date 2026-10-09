"""Input P(k) tables for the v28 bins: linear or halofit CAMB, `# kh\\tPk` TSVs.

    pixi run -e tables pk-tables make-default [--out-dir data]
    pixi run -e tables pk-tables check [--pk-dir data]
    pixi run -e tables pk-tables make --model {linear,halofit} [--kh-max 10]
        [--npoints 12501] [--out-dir data] [--z 0.1 0.3 ...]

Planck 2018 cosmology, `delta_tot`, log-spaced kh: the v28 tables' CAMB call. `halofit`
is Takahashi et al. 2012 as CAMB implements it, including the Bird et al. 2012
massive-neutrino terms; CAMB leaves kh < `Min_kh_nonlinear` (0.005) linear.

`make-default` writes the seven v28 linear tables (kh 1e-4 to 1, 10001 nodes) under
the names the default bin table gives them (`Bin.pk_file`); `check` regenerates them
and compares byte for byte with those files in `--pk-dir`, reporting the maximum
relative difference and exiting 1 on a mismatch. `make` writes
`matterpower_camb_{lin,halofit}_kmax<kh-max>_zeff=<z>.tsv`. Runs in the `tables` env
(CAMB, no JAX); `logunusual.suite` is dependency-free and read from the checkout.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import camb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logunusual import suite  # noqa: E402

# Planck 2018 TT,TE,EE+lowE+lensing best-fit, the v28 tables' cosmology.
PLANCK18 = dict(
    H0=67.36,
    ombh2=0.02237,
    omch2=0.1200,
    omk=0.0,
    mnu=0.06,
    tau=0.0544,
    As=2.1e-9,
    ns=0.9649,
)
V28_KH = (1e-4, 1.0, 10001)  # the v28 tables' kh_min, kh_max, npoints
MODELS = ("linear", "halofit")
MODEL_TAG = {"linear": "lin", "halofit": "halofit"}
HALOFIT_VERSION = "takahashi"


def make_pk(z, model="linear", kh_min=1e-4, kh_max=1.0, npoints=10001, cosmo=PLANCK18):
    """`(kh, P)` at redshift `z` from CAMB; `model` is "linear" or "halofit"."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=cosmo["H0"],
        ombh2=cosmo["ombh2"],
        omch2=cosmo["omch2"],
        omk=cosmo["omk"],
        mnu=cosmo["mnu"],
        tau=cosmo["tau"],
    )
    pars.InitPower.set_params(As=cosmo["As"], ns=cosmo["ns"])
    pars.set_matter_power(redshifts=[z], kmax=2.0 * kh_max)
    if model == "linear":
        pars.NonLinear = camb.model.NonLinear_none
    else:
        pars.NonLinear = camb.model.NonLinear_pk
        pars.NonLinearModel.set_params(halofit_version=HALOFIT_VERSION)
    results = camb.get_results(pars)
    kh, _, pk = results.get_matter_power_spectrum(
        minkh=kh_min, maxkh=kh_max, npoints=npoints
    )
    return kh, pk[0]


def tsv_bytes(kh, pk) -> bytes:
    """The v28 table format: tab-separated `kh Pk` rows under a `# kh Pk` header (read
    by `pk.load_pk_tsv`)."""
    lines = ["# kh\tPk\n"] + [f"{k}\t{p}\n" for k, p in zip(kh, pk)]
    return "".join(lines).encode()


def table_name(z, model, kh_max) -> str:
    """`make`'s output name: `matterpower_camb_<tag>_kmax<kh_max>_zeff=<z>.tsv`."""
    return f"matterpower_camb_{MODEL_TAG[model]}_kmax{kh_max:g}_zeff={z:g}.tsv"


def v28_z():
    return [b.z_eff for b in suite.BIN_SUITE_V28]


def make_default(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for b in suite.BIN_SUITE_V28:
        blob = tsv_bytes(*make_pk(b.z_eff, "linear", *V28_KH))
        path = out / b.pk_file
        path.write_bytes(blob)
        print(f"{path}: sha256 {hashlib.sha256(blob).hexdigest()}")
    return 0


def check(args) -> int:
    ok = True
    for b in suite.BIN_SUITE_V28:
        z = b.z_eff
        ref_path = Path(args.pk_dir) / b.pk_file
        ref = ref_path.read_bytes()
        kh, pk = make_pk(z, "linear", *V28_KH)
        new = tsv_bytes(kh, pk)
        if new == ref:
            print(f"z {z:g}: IDENTICAL ({hashlib.sha256(ref).hexdigest()[:12]})")
            continue
        ok = False
        old = np.loadtxt(ref_path, comments="#")
        if old.shape != (len(kh), 2):
            print(f"z {z:g}: DIFFERENT, shape {old.shape} vs {(len(kh), 2)}")
            continue
        dk = np.max(np.abs(kh / old[:, 0] - 1.0))
        dp = np.max(np.abs(pk / old[:, 1] - 1.0))
        print(f"z {z:g}: DIFFERENT, max |dk/k| {dk:.3e}, max |dP/P| {dp:.3e}")
    print("tables:", "IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


def make(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for z in args.z or v28_z():
        kh, pk = make_pk(z, args.model, args.kh_min, args.kh_max, args.npoints)
        path = out / table_name(z, args.model, args.kh_max)
        blob = tsv_bytes(kh, pk)
        path.write_bytes(blob)
        print(f"{path}: {len(kh)} rows, sha256 {hashlib.sha256(blob).hexdigest()}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser(
        "make-default", help="write the default bin table's linear tables"
    )
    d.add_argument("--out-dir", default="data")
    c = sub.add_parser("check", help="regenerate the v28 linear tables byte for byte")
    c.add_argument("--pk-dir", default="data")
    m = sub.add_parser("make", help="write tables for the v28 z_eff")
    m.add_argument("--model", choices=MODELS, required=True)
    m.add_argument("--kh-min", type=float, default=1e-4)
    m.add_argument("--kh-max", type=float, default=10.0)
    m.add_argument("--npoints", type=int, default=12501)
    m.add_argument("--z", type=float, nargs="+", default=None)
    m.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    run = {"make-default": make_default, "check": check, "make": make}[args.mode]
    sys.exit(run(args))


if __name__ == "__main__":
    main()
