# Double Blazed Gratings

`DoubleBlazedGrating` has two rising facets and one falling facet in each
period. From the foot at `x = 0`, the surface rises at `blaze_angle_1_deg`,
continues at the shallower `blaze_angle_2_deg` up to the apex, then falls at
`anti_blaze_angle_deg` to zero at the end of the period. Any configured coating
follows the surface conformally.

```python
import grax

grating = grax.DoubleBlazedGrating(
    period_lpermm=600,
    blaze_angle_1_deg=1.5,
    blaze_angle_2_deg=0.75,
    anti_blaze_angle_deg=30.0,
    second_facet_fraction=0.5,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    x_resolution_nm=1.0,
    z_resolution_nm=0.5,
)

grating.plot_profile("double_blazed.png")
```

The runnable [`double_blazed.py`](../../../examples/grating/double_blazed.py)
example generates the same profile plot.

## Geometry

The inputs must satisfy `0 < α2 < α1 < 90°`, `0 < β ≤ 90°` and `0 < prep < 1`,
where `α1` and `α2` are the lower and upper blaze angles, `β` is the anti-blaze
angle and `prep` is `second_facet_fraction`. The horizontal widths of the two
rising facets are `A1` and `A2`, and `prep = A2 / (A1 + A2)`.

The facet heights `h1` and `h2` follow from the period `P`, using
`f = prep / (1 - prep)` and `B`, the width of the triangle above the kink:

```text
B  = P f / (f + (1 + ctg β / ctg α1) / (1 + ctg β / ctg α2))
h1 = (P - B) / (ctg α1 + ctg β)
h2 = B / (ctg α2 + ctg β)
A1 = h1 ctg α1
A2 = h2 ctg α2
```

The profile points of one period are:

```text
x = [0, A1, A1 + A2, P]
z = [0, h1, h1 + h2, 0]
```

The groove depth `h1 + h2` is available as `depth_nm` and is derived from the
angles, so it is not an input.

In the web UI, choose **Double blazed**, enter the three angles and the upper
facet share `p` (`prep` above; the drawing marks `A1` and `A2`), and select the substrate, coating stack and
roughness model as for other gratings.
