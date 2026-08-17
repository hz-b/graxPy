# Nevière differential method

The differential method is one of `grax`'s two selectable electromagnetic solver
paths, selected with `solver="neviere"`. It solves the same physical problem as the RCWA solver in
[RCWA theory](rcwa-theory.md) and shares most of its machinery; this page
describes only what differs and why the two agree.

The method is due to Nevière, Vincent and Petit {cite}`neviere1974differential`,
extended to multilayer (Bragg–Fresnel) stacks in {cite}`neviere1994multilayer`,
and treated at length in {cite}`neviere2003light`.

## What differs from RCWA

Both solvers expand the fields and the permittivity in the same truncated
Fourier basis along `x`, and both reduce one grating period to a set of coupled
ordinary differential equations in the depth direction `z`. They differ in one
step only:

- **RCWA** treats each layer as invariant in `z`, eigen-decomposes its field
  operator once, and propagates the layer analytically through the resulting
  modal exponentials.
- **The differential method** integrates the coupled system numerically in `z`
  with a fourth-order Runge–Kutta scheme, without ever forming the modes.

Everything before that step (`res1`, the Fourier coefficients, the layer field
operators) and everything after it (the interface cascade, the Rayleigh
expansion in the semi-infinite media, the Poynting-flux efficiencies) is one
shared implementation in `grax.solvers.common`.

## The coupled system

Write `F` for the tangential field that is continuous across every interface,
and `G` for its conjugate tangential partner:

| Polarization | `F` | `G` |
| --- | --- | --- |
| TE (`s`) | `E_y` | `dE_y/dz` |
| TM (`p`) | `H_y` | `E_x = [[1/ε]] dH_y/dz` |

Maxwell's equations then reduce to a first-order system in the Fourier
coefficients:

```
d/dz [F; G] = [[0, A], [B, 0]] [F; G]
```

with

- **TE:** `A = I`, `B = Kx² − k0² [[ε]]`
- **TM:** `A = [[1/ε]]⁻¹`, `B = Kx [[ε]]⁻¹ Kx − k0² I`

where `Kx = diag(kx_m)` holds the in-plane wavevector of each retained order and
`[[·]]` denotes a Toeplitz convolution matrix built from Fourier coefficients.

Eliminating `G` recovers the second-order form `d²F/dz² = (A·B) F`, and `A·B` is
exactly the operator the RCWA solver eigen-decomposes. That is the reason the
two solvers agree: they are two numerical treatments of one system, not two
models.

## Fast Fourier factorization

The TM operators above are not the naive Laurent products. `[[ε]]⁻¹` and
`[[1/ε]]⁻¹` are different matrices, and using the wrong one is the classic cause
of the differential method's notoriously slow TM convergence. The rules applied
here are Li's {cite}`li1996formulation`, in the form Popov and Nevière give for
the differential method {cite}`popov2000gratingtheory`: the inverse rule for the
product of the discontinuous permittivity with the discontinuous normal field
component, and the direct rule where both factors are continuous.

In the code these operators come from `layer_field_operators()`, shared with
RCWA, so the factorization is inherited rather than re-derived.

## Numerical stability

A transfer matrix across a whole grating grows like `exp(|q| d)` in the
evanescent orders, which overflows for anything but shallow structures. The
solver therefore never forms one:

1. Each layer is split into sub-blocks whose optical thickness `|q| d` is capped
   by `NeviereOptions.block_phase`. `|q|` is bounded per layer from the row norm
   of the layer operator, which is tight for the low-contrast permittivities of
   X-ray optics.
2. Each sub-block is integrated with Runge–Kutta into a transfer matrix, then
   immediately converted into a Dirichlet-to-Neumann interface-response block
   mapping `[F_top; F_bottom]` to `[G_top; G_bottom]`.
3. Blocks are combined with `_cascade_boundary_pair()`, which eliminates the
   shared interface field through a linear solve.

Step 3 is an R-matrix (impedance) propagation, not a transfer-matrix product, so
no growing exponential is ever accumulated across the stack. The conversion in
step 2 reproduces the modal block `[[-q coth(qd), q csch(qd)], [-q csch(qd), q
coth(qd)]]` exactly for a homogeneous slab, which is a useful check when reading
the code.

In practice this makes the differential method the more robust of the two on
deep structures: the modal solver evaluates `q / sinh(q d)` across a full layer
and overflows above roughly seven wavelengths of depth for a high-contrast
lamellar grating, where the differential method still conserves energy to `1e-9`
at 167 wavelengths.

## Choosing the step size

Step and sub-block sizes are expressed in optical phase rather than nanometers,
so one setting behaves consistently across photon energies, grazing angles and
truncation orders:

- `step_phase` bounds `|q| h` for one Runge–Kutta step. Accuracy improves as its
  fourth power, while cost grows far more slowly, because most of the per-layer
  work is the interface conversion and cascade rather than the Runge–Kutta
  stages.
- `block_phase` bounds `|q| d` for one explicitly formed transfer matrix. It
  trades conditioning only; the converged answer does not depend on it.

The Fourier truncation order is the existing `fourier_orders` argument, retaining
`2·fourier_orders + 1` components. There is no separate truncation setting for
this solver.

## Sampling the permittivity along z

`NeviereOptions.z_sampling` selects where `ε(x, z)` comes from:

- `"textures"` (default) integrates through the same z-sliced permittivity RCWA
  uses, so the two solvers see byte-identical geometry and any difference between
  them is Runge–Kutta truncation error alone. This is the mode used for
  cross-validation.
- `"continuous"` re-expands `ε(x, z)` from the true grating profile at every
  sub-block, which is the textbook differential method. It drops the staircase
  approximation entirely: its result does not depend on `z_resolution_nm` at all,
  and it reaches the z-converged answer more cheaply than refining the staircase
  does.

`"continuous"` needs the grating object to sample the profile between solver
rows, so it is selected through `grax.run_simulation`, which builds the sampler
with `build_grating_epsilon_sampler()`.

## Energy balance

`propagating_energy_balance()` sums the propagating reflected and transmitted
orders. For a lossless structure the total is one; the absorbing materials used
in the X-ray regime make it legitimately smaller, and the shortfall is the
absorbed fraction rather than a solver error. Only an upper bound is enforced,
through `NeviereOptions.energy_balance_tolerance`.

## Internal solver reference

```{autoclass} grax.solvers.neviere.NeviereOptions
:members:
```

```{autofunction} grax.solvers.neviere.res2_dm
```

```{autofunction} grax.solvers.neviere.build_grating_epsilon_sampler
```

```{autoclass} grax.solvers.neviere.EpsilonSampler
:members:
```

```{autofunction} grax.solvers.common.layer_field_operators
```

```{autofunction} grax.solvers.common.solve_stack_from_layer_blocks
```

```{autofunction} grax.solvers.common.propagating_energy_balance
```

The full bibliography is rendered on the [RCWA theory](rcwa-theory.md) page.
