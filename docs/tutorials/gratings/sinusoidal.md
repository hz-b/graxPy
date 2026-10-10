# Sinusoidal Gratings

`SinusoidalGrating` models a sinusoidally profiled substrate. Its only
shape-specific parameter is `depth_nm`, the peak-to-valley depth. The groove is
zero-depth at each period boundary and reaches its maximum depth at the center
of the period. Any configured single layer, multilayer, custom stack, or top cap
follows the sinusoidal surface conformally.

```python
import grax

grating = grax.SinusoidalGrating(
    period_lpermm=600,
    depth_nm=30.0,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    x_resolution_nm=1.0,
    z_resolution_nm=0.5,
)

grating.plot_profile("sinusoidal.png")
```

The runnable [`sinusoidal.py`](../../../examples/grating/sinusoidal.py) example
generates the same profile plot. The separate
[`sinusoidal_custom_profile.py`](../../../examples/grating/sinusoidal_custom_profile.py)
example remains an illustration of how to implement an application-specific
profile by subclassing `BaseGrating`.

In the web UI, choose **Sinusoidal**, enter the peak-to-valley depth, and select
the substrate, coating stack, and roughness model in the same way as for
laminar and blazed gratings. The saved grating can be used by every available
run workflow and either solver.
