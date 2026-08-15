# Developer documentation

This section is for contributors and maintainers who need to understand how the
high-level API reaches the numerical solvers.

`grax` ships two independent solvers behind one interface. Read
[RCWA theory](rcwa-theory.md) first: the
[Nevière differential method](neviere-theory.md) page describes only what the
second solver does differently.

```{toctree}
:maxdepth: 1

rcwa-theory
neviere-theory
physics-to-code
solver-flow
module-guide
```
