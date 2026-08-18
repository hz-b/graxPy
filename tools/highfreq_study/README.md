# High-frequency (carrier-extracted) integral method

Whether the boundary-integral solver's node count can be decoupled from
`d / lambda`, so that the X-ray regime needs geometric degrees of freedom rather
than wavelength-resolution points.

The classical solver in `src/grax/solvers/integral.py` is untouched and remains
the reference. The new formulation lives in `src/grax/solvers/_carrier.py` and is
reached through `res2_hf`, which mirrors `res2_im` argument for argument.

```bash
python tools/highfreq_study/similarity_check.py
python tools/highfreq_study/test1_flat.py --energies 50 100 200
python tools/highfreq_study/test2_sinusoid.py --energy 100 --nodes 1024
python tools/highfreq_study/test2_sinusoid.py --energy 200 --nodes 2048 --modes 2 4
```

## The substitution that does not work

`u(x) = exp(i alpha_0 x) u~(x)` carried through the integral equation gives the
reduced kernel `exp(-i alpha_0 (x_t - x_s)) G(r_t, r_s)` and the right-hand side
`exp(-i beta_0 y_t)`, which is genuinely smooth — the carrier cancels against the
incident field exactly.

On the same node set this is a **diagonal similarity transform**: column `j`
scaled by `exp(i alpha_0 x_j)`, row `i` by its conjugate. It cannot change the
spectrum, the conditioning, or the quadrature error. Measured:

| E (eV) | d/lambda | cond(classical) | cond(extracted) | max eigenvalue diff |
| --- | --- | --- | --- | --- |
| 100 | 202 | 8.923359e+00 | 8.923359e+00 | 1.5e-13 |
| 500 | 1008 | 2.619946e+00 | 2.619946e+00 | 7.0e-14 |

This is worth stating plainly because it is the trap: phase-rotating the unknowns
and leaving the discretization alone buys exactly nothing.

## What was implemented instead

The **representation** of the unknown changes. The envelope is expanded in a
truncated Floquet-Fourier basis

    phi(x) = sum_m phi~_m exp(i alpha_m x),    alpha_m = alpha_0 + 2 pi m / d

and the equation is imposed in the Galerkin sense against the same basis. The
basis is orthonormal under the exact inner product, because `alpha_m - alpha_m'`
is a plain integer harmonic — the carrier cancels between analysis and synthesis.
Measured Gram defect on the node grid: **1.3e-15**.

The projection is formed from the classical nodal blocks, so the stack, the jump
terms, the `tau` continuity factors, the cross-interface coupling, the incident
field and the Rayleigh reconstruction are inherited unchanged. The two
formulations cannot drift apart on the physics; only the basis differs.

## Test 1 — flat interface, against analytic Fresnel

`d = 2500 nm`, 2 degrees grazing, TM, tolerance 1e-4 absolute.

| E (eV) | d/lambda | classical N | dev | hf dof/density | dev |
| --- | --- | --- | --- | --- | --- |
| 50 | 101 | 512 | 3.9e-08 | **1** | 3.9e-08 |
| 100 | 202 | 1024 | 3.3e-08 | **1** | 3.3e-08 |

One unknown per density at either energy, to the same digit as a classical solve
with a thousand. A flat interface couples no orders, so its envelope is exactly
constant and the mode count is energy-independent by construction.

Sweeping the two requirements independently at fixed energy shows they are
genuinely separate — the mode count is irrelevant, the node count is everything:

| d/lambda | Nquad | M = 0 | M = 2 | M = 8 |
| --- | --- | --- | --- | --- |
| 101 | 128 | 1.8e-01 | 1.8e-01 | 1.8e-01 |
| 101 | 256 | 3.4e-03 | 3.4e-03 | 3.4e-03 |
| 101 | 512 | 3.9e-08 | 3.9e-08 | 3.9e-08 |
| 202 | 512 | 7.8e-04 | 7.8e-04 | 7.8e-04 |
| 202 | 1024 | 3.3e-08 | 3.3e-08 | 3.3e-08 |

## Test 2 — shallow sinusoid, against RCWA

`h/d = 0.008`, 100 eV, `d/lambda = 202`, 2 degrees grazing, TM.

| method | m=0 | m=1 | m=2 | max dev | unknowns | secs |
| --- | --- | --- | --- | --- | --- | --- |
| RCWA | 0.4318516 | 0.0184597 | 0.0012127 | — | — | 1.0 |
| classical BIE, N=1024 | 0.4317565 | 0.0184838 | 0.0012204 | 9.5e-05 | 2048 | 161.5 |
| hf BIE, M=1 | 0.4318266 | 0.0188098 | 0.0009897 | 3.5e-04 | 6 | 147.2 |
| hf BIE, M=2 | 0.4317563 | 0.0184811 | 0.0012261 | 9.5e-05 | **10** | 134.9 |
| hf BIE, M=4 | 0.4317565 | 0.0184838 | 0.0012204 | 9.5e-05 | 18 | 136.5 |
| hf BIE, M=16 | 0.4317565 | 0.0184838 | 0.0012204 | 9.5e-05 | 66 | 137.3 |

From `M=4` upward the hf answer equals the classical one to every printed digit.
The envelope spectrum is cleanly geometric — roughly a decade per harmonic, which
is what an analytic envelope looks like:

| \|m\| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \|phi~_m\| | 1.0 | 2.3e-1 | 2.2e-2 | 1.3e-3 | 1.1e-4 | 7.8e-6 | 5.2e-7 | 3.5e-8 | 1.6e-9 |

## Where this leaves the two bottlenecks

**A. Unknown count — removed.** 2048 unknowns become 10 at equal accuracy, and
the count is set by how much the grating modulates the field, not by how many
wavelengths fit in a period.

**B. Kernel quadrature — untouched, and now the whole problem.** Runtime is
unchanged (135 s against 161 s) because the projection is built *from* the nodal
blocks and assembly is 99.4% of the cost. `Nquad` still has to resolve the
kernel, whose Fourier content spans the diffraction orders.

## Recommended next stage

Stop building the nodal blocks at all. In the projected matrix element

    A[m', m] = (1/d) int int exp(-i alpha_m' x_t) G exp(i alpha_m x_s) J ds dt

substituting the plane-wave form of `G` makes the carrier cancel *analytically*:

    exp(-i alpha_m' x_t) exp(i alpha_n (x_t - x_s)) exp(i alpha_m x_s)
      = exp(i 2 pi (n - m') x_t / d) exp(-i 2 pi (n - m) x_s / d)

Every surviving factor is a low integer harmonic. Nothing in the integrand
oscillates at `d / lambda`, so the quadrature that evaluates these elements is
set by the profile and by `|n - m|`, not by the wavelength — which is the same
structural move that lets PCGrate-class modified integral methods run at
`lambda / d ~ 1e-3`.

The honest difficulty is `exp(i beta_n |y_t - y_s|)`: the absolute value blocks a
clean outer-product factorization, and the plane-wave series does not converge at
coincident points, which is exactly why Ewald is used now. The rigorous route
splits the integration domain at `y_t = y_s` rather than dropping the modulus —
dropping it is the Rayleigh hypothesis and is not acceptable here. For a shallow
graph profile most node pairs sit in the near region, so this is real numerics
rather than a rewrite of bookkeeping.

That is the one optimization worth doing next. It is the only remaining lever
that changes the exponent rather than the constant.
