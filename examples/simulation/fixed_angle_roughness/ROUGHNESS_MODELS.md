# How the two roughness models are computed

**Debye-Waller** applies a scalar damping factor to the diffraction
efficiencies, without touching the grating geometry:

    damping = exp(-(4*pi*sigma*sin(theta)/wavelength)^2)

`sigma` is the rms roughness height and `theta` the grazing incidence
angle. It approximates the loss from unresolved, sub-wavelength
roughness and has no lateral correlation length of its own.

**Random-interface** roughness instead perturbs the actual interface
shape with a periodic, zero-mean Gaussian random field sampled per
grating period. Its lateral correlation is set via a Gaussian power
spectrum:

    amplitude(k) = exp(-0.25 * (k * xi)^2)

where `k` are the spatial frequencies over one period and `xi` is
`correlation_length_nm`. The field is built in Fourier space, inverse
transformed, then rescaled to the target rms `sigma`. Larger `xi` means
smoother, longer-wavelength height variations; `xi = 0` gives white
noise (fully uncorrelated); `xi = None` defaults to `period/10`.

## Supercells: correlation lengths beyond one period

A single grating period caps how much lateral structure the
random-interface field can represent — a correlation length approaching
or exceeding the period has nowhere to develop. `RoughnessSpec.num_supercells`
(`N`, default 1) lifts that cap: instead of one period, the Gaussian random
field is synthesized once as a **single continuous field spanning `N`
periods**, not `N` independent periods stitched together. This is a full
physics change, not just a wider plot window: the RCWA solver's fundamental
Fourier period becomes `N * grating.period_nm`, so the simulation resolves
whatever the disorder actually does to the diffraction pattern.

Two consequences follow directly from simulating `N` real periods at once:

- **Diffraction orders become fractional.** With period `N * period_nm`,
  the solver's natural order spacing is `1/N` of the nominal grating orders.
  Physical order `m` (the ideal grating's `m`-th diffraction order) still sits
  at exactly `m`, but the intervening solver orders (`m + 1/N`, `m + 2/N`, ...)
  are real satellite/diffuse orders introduced by the disorder breaking
  perfect periodicity — the "diffuse background" scattered around each
  specular order. `roughness_kind_comparison.py`'s per-run order-spectrum
  plots show exactly this: sharp peaks at integer orders, small satellite
  peaks at the fractional orders in between, growing with `N`.
- **The default correlation length does not scale with `N`.** It's still
  `period_nm/10` (one grating period, not `N` periods) unless set explicitly.
  Enabling `num_supercells` without also raising `correlation_length_nm`
  (typically toward or beyond one period) buys extra compute for no new
  physics — the field simply repeats the same short-range statistics `N`
  times.

Cost scales badly with `N`: to keep the same physical angular resolution,
the number of Fourier orders passed to the solver must also scale by `N`
(`fourier_orders * num_supercells`), and the RCWA eigenproblem cost grows
worse than linearly with that count — an `N=10` supercell at the same
per-groove order count can be roughly `10^3`, not `10x`, slower than the
single-period case. Both example scripts work around this by using a
smaller `SUPERCELL_FOURIER_ORDERS` for `num_supercells > 1` runs, trading
some resolution of the base grating profile for tractable runtime.
