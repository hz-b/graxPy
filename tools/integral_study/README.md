# Integral-method study

Measurements establishing whether the boundary-integral solver in
`src/grax/solvers/integral.py` can be made practical for graxPy's X-ray regime
(`d / lambda` from ~100 to several thousand).

The solver is **not** wired into `grax.run_simulation`. These scripts drive it at
the `res2` level, through the same grating objects the reference solvers use, so
what is compared is numerics and not two different geometries. Nothing here
changes package behaviour.

## Scripts

| script | purpose |
| --- | --- |
| `_cases.py` | Benchmark cases and the three solver drivers |
| `compare.py` | Stage 0: validate against RCWA and Nevière |
| `scaling.py` | Panel ladder, plus quadrature-order and Ewald-splitting sweeps |
| `fixed_n_probe.py` | Stage 1: measure the convergence order and project `N_required` |
| `spectrum.py` | Diagnostic: harmonic content of the boundary densities |

```bash
python tools/integral_study/compare.py
python tools/integral_study/fixed_n_probe.py --ratios 10 25 50 --panels 384 768
python tools/integral_study/scaling.py --sweep quadrature
python tools/integral_study/scaling.py --sweep splitting
python tools/integral_study/spectrum.py --ratios 5 10 25 50 --depth-nm 20 --panels 384
```

Result `*.log` files are gitignored, so the findings are recorded below.

## Stage 0 — validation

The formulation is correct. On a flat interface the double-layer operator is
identically zero and the single layer converges to `i / 2 beta_0`, both analytic;
the assembled system reproduces analytic Fresnel in TE and TM. Convergence is
clean `O(h^2)`:

| N | deviation vs RCWA | energy balance |
| --- | --- | --- |
| 64 | 2.39e-3 | 1.0524 |
| 128 | 4.32e-4 | 1.0123 |
| 256 | 9.78e-5 | 1.0030 |
| 512 | 2.37e-5 | 1.0008 |

## Stage 1 — how `N_required` scales

Case: shallow sinusoid on Si, 4 degrees grazing, TM, 20 nm depth, period varied
at fixed wavelength so `d / lambda` grows while `h / lambda` stays fixed.
Tolerance 1e-4 absolute on efficiency.

| `d/lambda` | `h/d` | measured order `p` | `N_required` | time at N=768 |
| --- | --- | --- | --- | --- |
| 10 | 0.161 | 2.10 | 1599 | 31.9 s |
| 25 | 0.065 | 2.07 | 3672 | 62.0 s |
| 50 | 0.032 | 2.20 | 6266 | 110.0 s |

`N_required ~ (d/lambda)^0.85` — linear, not saturating. Extrapolates to ~12000
unknowns at `d/lambda = 100`. Cross-validated: `d/lambda = 10` gives 1599 from
panels 384/768 and 1549 from panels 96/192.

Since `N_required ~ (C / tol)^(1/p)` with `C` measured, raising the
discretization order from 2 to 4 gives ~110 unknowns at `d/lambda = 100` and
~290 at `d/lambda = 1000`. Those bracket the ~50 collocation points per period
Goray reports for the modified integral method at `lambda / d = 1e-3`, so the
published figures and these measurements describe the same method at different
discretization orders.

## Stage 1 diagnostic — why `N` grows

Significant Fourier harmonics of the boundary densities, pseudo-periodic carrier
removed, at a 1e-4 relative threshold:

| `d/lambda` | `phi` harmonics | `psi` harmonics | propagating orders | `N_required` |
| --- | --- | --- | --- | --- |
| 5 | 13 | 108 | 10 | 1055 |
| 10 | 11 | 124 | 20 | 1599 |
| 25 | 10 | 54 | 50 | 3672 |
| 50 | 8 | 31 | 100 | 6266 |

The information content and the panel count move in **opposite** directions. As
`d / lambda` grows at fixed depth, `h / d` shrinks, the grating perturbs less and
the density envelope gets smoother, while the kernel oscillates faster. A
piecewise-constant panel method uses one grid for both and is dragged by the
faster requirement. At `d/lambda = 50` it spends 6266 panels on densities
carrying 8 and 31 significant harmonics.

`psi`'s count rises once (108 to 124) before falling; only that first point is
out of line, and some of `psi`'s content at this threshold is numerical noise, so
treat its absolute value as indicative. `phi`'s trend is clean.

## Free speedups, measured

**Quadrature order is accuracy-neutral.** At `d/lambda = 25`, N = 272:

| order | deviation | time |
| --- | --- | --- |
| 2 | 1.859e-4 | 8.0 s |
| 8 | 1.867e-4 | 31.5 s |
| 16 | 1.867e-4 | 62.1 s |

Four significant figures apart across an 8x range, while cost is linear. Worth
~4x.

**The Ewald splitting default is 2x off.** Same case, sweeping
`ewald_splitting` in multiples of `|k|/2`, which is what
`default_ewald_splitting` floors at:

| splitting | deviation | time |
| --- | --- | --- |
| `\|k\|/8` | 2.221e-2 | 14.9 s |
| `\|k\|/4` | 2.219e-2 | 19.1 s |
| `\|k\|/2` (default) | 2.219e-2 | 29.9 s |
| `2\|k\|` | 2.219e-2 | 82.8 s |

Time is monotonic in the splitting parameter across the whole range; the floor
only guarantees the lattice series converges, it does not minimise total work.
The gain should grow with `d / lambda`, because the spectral half — the one that
shrinks as the splitting shrinks — is the half that lengthens with `d / lambda`.

The deviation is identical to three significant figures across a 16x range of
splitting parameter. That is splitting-invariance confirmed in the full solve
rather than only in the kernel unit test.

## Method notes for whoever runs these next

Two ways to get a confidently wrong answer here, both hit during this study and
both now guarded in `fixed_n_probe.py`:

1. **Tolerance comparable to the signal.** The first version of the sinusoid case
   ran at 40 degrees grazing, where every order sits between 2e-6 and 4.2e-4. An
   absolute tolerance of 1e-4 then permits 25 to 100 percent relative error and
   the measurement means nothing. The case now runs at 4 degrees grazing in TM,
   matching the validation suite, where order zero carries 12 to 15 percent.

2. **Fitting outside the asymptotic regime.** Projecting `N_required` from
   `error ~ N^-p` requires both solves to be converging. A coarse solve that has
   not started converging reports an absurd apparent order, projects a tiny
   `N_required` from it, and can turn a linear trend into a false "saturating"
   verdict. Rows are now rejected when the coarse deviation exceeds a physical
   bound or the measured order is implausible, and rejected rows are named.

Also worth knowing: reading a reflection coefficient off `phi` alone flatters the
solver by about three orders of magnitude, because `phi` is superconvergent
`O(h^3)` while `psi` is `O(h^2)` and the Rayleigh projection needs both.
