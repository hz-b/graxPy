# Gratings

This section groups grating-construction tutorials.

```{toctree}
:maxdepth: 1

gratings/laminar
gratings/sinusoidal
gratings/blazed
gratings/blazed-multilayer-custom-stack
gratings/afm-preprocessing-profile
```


## Roughness in the web grating builder

Select **None**, **Debye–Waller**, or **Random interface** once when creating or
editing a grating. This setting is saved with the grating and used by its runs.
All grating profiles and coating stack types support it.

Each layer's RMS roughness σ describes its upper interface. The substrate has
its own settings.
For multilayers, A and B settings repeat through the block; the individual
interfaces receive independent random shapes.

Debye–Waller preserves the smooth geometry and applies the existing effective
RMS damping approximation to efficiencies. Random interface changes the actual
geometry and additionally takes a lateral correlation length ξ for each
interface. Blank σ means zero; blank ξ means one tenth of the nominal grating
period. ξ = 0 produces uncorrelated noise.

Random-interface settings share a seed (default 0), simulated period count
(default 1), and realization count (default 1). The extra preview shows the
first simulation realization and its material map, using its resolved seed.
The ordinary annotated profile remains the nominal geometry. Increasing the
period count or realization count increases simulation cost.

Roughness perturbs vertical interface heights. Coating interfaces that cross
are rejected with their indices and
realization seed. Reduce their RMS values or increase the intervening layer
thickness. Resolution warnings suggest finer x/z spacing where appropriate.
Existing saved gratings without a model load with roughness disabled.

In Python, stack constructors accept optional `*_correlation_length_nm` values
for substrate, layer/A/B, and cap interfaces; `LayerSpec` accepts
`correlation_length_nm`. Existing `RoughnessSpec` settings remain supported.
