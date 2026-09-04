# Landscape: lognormal galaxy mocks and fast mock catalogs

Written 2026-09-04 for the design of `logunusual`. Every arXiv entry below was checked
against its abstract page before being written; the claim attached to each is what the
abstract supports, nothing more. Coles & Jones is pre-arXiv and was checked via its
CrossRef record.

## The lognormal construction and its pathologies

- **Coles & Jones 1991**, MNRAS 248, 1, "A lognormal model for the cosmological mass
  distribution". The origin of the lognormal density field. The standard pipeline every
  code below shares: P(k) -> xi(r) -> xi_G = ln(1 + xi) -> P_G(k) -> Gaussian field ->
  exponentiate -> Poisson-sample.
- **Xavier, Abdalla & Joachimi 2016**, arXiv:1602.08503, MNRAS 459, 3693 (the FLASK
  paper). The lognormal distribution's shape can make some target correlations
  unattainable; they fix this by slightly distorting the input power spectra with one of
  two algorithms. Relevant to `pk.py`: the xi -> xi_G -> P_G inversion is where a
  non-attainable target shows up as a negative P_G, and an f_NL 1/k^2 low-k boost pushes
  on exactly this.

Practical fixes not tied to one paper (mean subtraction after exponentiation, voxel-window
deconvolution, clipping P_G >= 0, aliasing) are documented in code: Henry's
`pk_to_pkG.jl` / `pixel_window!` and nbodykit's `LogNormalCatalog`.

## Public codes

- **Agrawal, Makiya, Chiang, Jeong, Saito & Komatsu 2017**, arXiv:1706.09195, JCAP 10
  (2017) 003, `lognormal_galaxies`. Poisson-samples the lognormal field and derives the
  velocity field from the linearised continuity equation of the matter field (zero
  vorticity), giving redshift-space mocks. This is the construction Henry's code and
  `logunusual` follow. The abstract does not mention f_NL.
- **nbodykit** (Hand et al. 2018, arXiv:1712.05834, AJ), `LogNormalCatalog`: lognormal
  field, Poisson sampling, Zel'dovich displacement. CPU / MPI.
- **GLASS**, Tessore, Loureiro, Joachimi, von Wietersheim-Kramsta & Jeffrey 2023,
  arXiv:2302.01942, OJAp. Builds a lightcone as a sequence of nested shells and introduces
  a technique to generate transformed Gaussian fields (including lognormal) to essentially
  arbitrary precision. Angular fields per shell; the nested-shell idea is the same
  geometry as the per-bin suite.
- **CoLoRe**, Ramirez-Perez, Sanchez, Alonso & Font-Ribera 2022, arXiv:2111.05069,
  JCAP. All-sky lightcone with lognormal, 1LPT or 2LPT density fields and multiple tracers
  (galaxy positions and velocities, lensing, ISW, line intensity mapping).
- **JAX precedent:** pmwd, Li et al. 2022, arXiv:2211.09958, a differentiable
  particle-mesh N-body library in JAX; DISCO-DJ (private Vienna repo; used by
  `disco-mocks`). Engineering references for GPU field stages, not lognormal codes.
- No public JAX or GPU lognormal galaxy mock generator was found in this survey. That is
  "none found", not "none exists".

## f_NL and lognormal mocks

- **Dalal, Dore, Huterer & Shirokov 2008**, arXiv:0710.4560, PRD 77, 123514, and
  **Slosar, Hirata, Seljak, Ho & Padmanabhan 2008**, arXiv:0805.3580, JCAP 08 (2008) 031:
  local f_NL induces a scale-dependent halo bias whose amplitude grows with scale. This is
  the Delta b(k) that milestone M4 injects through the input galaxy P(k).
- No paper was found documenting f_NL injection inside a public lognormal code. Because a
  lognormal realisation is fixed by its target two-point function, the injection is an
  input-P(k) substitution; M4 documents its own construction and gates it on the recovered
  low-k power ratio.
- **Limits.** Colavincenzo et al. 2019, arXiv:1806.09499, MNRAS ("Comparing approximate
  methods for mock catalogues and covariance matrices III: bispectrum"): most fast methods
  reproduce N-body bispectrum errors within 10%, with exceptions among methods whose
  clustering calibration is limited to two-point statistics. A lognormal field is by
  construction a two-point object; `logunusual` is scoped to two-point uses (null tests,
  covariance shape).
- SPHEREx f_NL forecasts that frame the target: Heinrich, Dore & Krause 2023,
  arXiv:2311.13082 (multi-tracer redshift-space bispectrum, photo-z errors); Wen,
  Grasshorn Gebhardt & Dore 2026, arXiv:2607.06697 (SFB power spectrum, relativistic
  effects; Fisher plus simulated Bayesian inference, no mocks); Shiveshwarkar, Brinckmann,
  Loverde & McQuinn 2023, arXiv:2306.07517, PRD (post-inflationary scale-dependent bias
  from light relics and ionising radiation can bias f_NL by 0.1-1 sigma for SPHEREx; a
  caution about which b(k) an f_NL-injected mock should carry).

## Lognormal mocks for covariance and wide-angle work

- **Grasshorn Gebhardt & Dore 2024**, arXiv:2310.17677: validates the SPHEREx SFB power
  spectrum pipeline on lognormal simulations and on complete and realistic eBOSS DR16 LRG
  EZmocks, with wide-angle effects handled exactly by the SFB basis. The lognormal mocks
  are Henry's LogNormalGalaxies, the reference implementation for this package.
- **Beutler & McDonald 2021**, arXiv:2106.06324, JCAP 11 (2021) 031: matrix-based
  framework for power spectrum multipoles including wide-angle effects and the survey
  window, applied to 6dFGS, BOSS and eBOSS.
- **Wadekar & Scoccimarro 2020**, arXiv:1910.02914, PRD 102, 123517: analytic
  perturbation-theory covariance of the multipoles, in agreement with BOSS DR12 PATCHY
  mocks to k = 0.6 h/Mpc. An analytic route that reduces the number of mocks a covariance
  needs.

## The wider fast-mock landscape (one line each)

- EZmocks: Chuang, Kitaura, Prada, Zhao & Yepes 2015, arXiv:1409.1124, MNRAS 446, 2621.
- PATCHY: Kitaura, Yepes & Prada 2014, arXiv:1307.3285, MNRAS Letters.
- L-PICOLA: Howlett, Manera & Percival 2015, arXiv:1506.03737 (COLA; includes primordial
  non-Gaussianity and run-time lightcone output).
- PINOCCHIO: Monaco et al. 2013, arXiv:1305.1505, MNRAS.
- GLAM: Klypin & Prada 2018, arXiv:1701.05690, MNRAS (PPM-GLAM).
- FastPM: Feng, Chu, Seljak & McDonald 2016, arXiv:1603.00476, MNRAS.
- HALOGEN: Avila et al. 2015, arXiv:1412.5228, MNRAS.
- BAM: Balaguera-Antolinez et al. 2019, arXiv:1806.05870, MNRAS Letters.
- Quijote: Villaescusa-Navarro et al. 2020, arXiv:1909.05273, ApJS 250, 2 (the N-body
  benchmark suite fast methods are validated against).
- pmwd (arXiv:2211.09958) and DISCO-DJ (private): JAX particle-mesh codes.
- The "Comparing approximate methods" series: I correlation function, Lippich et al.
  2019, arXiv:1806.09477; III bispectrum, Colavincenzo et al. 2019, arXiv:1806.09499.

## What this means for logunusual

The construction is settled and every code above shares it; the differences are in the
P_G inversion, the velocity assignment, the lightcone geometry and the engineering. The
gap `logunusual` fills is engineering: a vectorised, GPU-capable, lightcone-shell lognormal
generator with velocities and RSD, whose input P(k) interface can take f_NL and nonlinear
inputs, gated against a reference implementation that already has a validated consumer.
