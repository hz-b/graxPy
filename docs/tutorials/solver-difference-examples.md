# Where the two solvers differ

GraxPy's modal RCWA and Nevière differential-method paths solve the same
one-dimensional grating problem through different layer-propagation methods.
The `neviere_solver.py` example sweeps energy for a representative grating and
compares the selected diffraction-order efficiency from both paths.

```bash
python examples/simulation/neviere_solver/neviere_solver.py
```

The example writes the comparison plot and one CSV per solver to
`examples/simulation/neviere_solver/results/`.

```{figure} images/simulation/neviere_solver_comparison.png
:alt: Energy sweep comparing RCWA and Nevière diffraction efficiencies, with their absolute difference below.
:width: 100%

The upper panel compares the selected-order efficiencies from RCWA and
Nevière across the energy sweep. The lower panel shows their absolute
difference.
```

For this example, the curves in the upper panel make the agreement directly
visible, while the lower panel exposes differences that would otherwise be
hidden by the efficiency scale. It is a useful first comparison when checking a
new grating setup or deciding whether to cross-check a result with the other
solver path.

For guidance on when to use the default RCWA path or the Nevière path, see
{doc}`choosing-a-solver`.
