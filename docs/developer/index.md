# Developer documentation

This section is for contributors and maintainers who need to understand how the
high-level API reaches the numerical solvers.

`grax` ships two selectable solver paths behind one interface: modal RCWA and
the Nevière differential method. They share Fourier/discretization and
boundary-response infrastructure, but use different layer-propagation methods.
Read [RCWA theory](rcwa-theory.md) and the
[Nevière differential method](neviere-theory.md) together for the two paths.

```{toctree}
:maxdepth: 1

rcwa-theory
neviere-theory
physics-to-code
solver-flow
module-guide
```
