# Blazed Multilayer With Custom Stack

This example composes a custom stack around an actual `MultilayerStack`
definition and explicitly includes:

- a `2 nm` Pt layer above the Si substrate,
- a Cr/C multilayer block from `grax.MultilayerStack`,
- a `2 nm` Te layer above the multilayer,
- a `2 nm` O top cap.

```python
import grax

d_period_nm = 6.0
gamma = 0.4
n_bilayers = 50

multilayer_stack = grax.MultilayerStack(
    substrate_material="Si",
    material_a="Cr",
    material_b="C",
    d_period_nm=d_period_nm,
    gamma=gamma,
    n_bilayers=n_bilayers,
    top_material="C",
)

layers_bottom_up = [grax.LayerSpec(material="Pt", thickness_nm=2.0)]
layers_bottom_up.extend(multilayer_stack.layer_specs_bottom_up())
layers_bottom_up.append(grax.LayerSpec(material="Te", thickness_nm=2.0))

custom_stack = grax.assemble_custom_stack(
    substrate_material="Si",
    layers_bottom_up=layers_bottom_up,
    top_cap_material="O",
    top_cap_thickness_nm=2.0,
)

blazed_grating = grax.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=custom_stack,
    x_resolution_nm=0.5,
    z_resolution_nm=0.5,
)

blazed_grating.plot_profile("blazed_multilayer_custom_stack.png")
custom_stack.plot_schematic("blazed_multilayer_custom_stack_schematic.png")
```

![Blazed multilayer with custom stack](../images/gratings/blazed_multilayer_custom_stack.png)

![Multilayer stack schematic for custom stack](../images/gratings/blazed_multilayer_custom_stack_schematic.png)
