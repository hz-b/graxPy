"""Coating stack descriptions for grating simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .materials import material_label


@dataclass
class BaseStack(ABC):
    """Base class for coating stacks applied on top of a grating substrate."""

    substrate_material: Any
    top_cap_material: Any | None = None
    top_cap_thickness_nm: float = 0.0

    @abstractmethod
    def layer_sequence_bottom_up(self) -> list[tuple[Any, float]]:
        """Return the material sequence above the substrate from bottom to top."""

    @property
    def total_thickness_nm(self) -> float:
        """Return the total thickness above the substrate surface."""

        return float(sum(thickness for _, thickness in self.layer_sequence_bottom_up()))

    def _layer_roughness_sigmas_bottom_up(self) -> list[float | None]:
        """Return the optional per-layer roughness sigma for each layer.

        The list is ordered bottom-up and matches ``layer_sequence_bottom_up()``
        one-to-one. Each entry is the roughness of that layer's *top* interface,
        or ``None`` when the layer defers to the grating-level default.
        """

        return [None] * len(self.layer_sequence_bottom_up())

    def interface_roughness_sigmas_bottom_up(self, default_sigma_nm: float) -> list[float]:
        """Return one roughness sigma per material interface, bottom-up.

        There are ``n_layers + 1`` interfaces: index 0 is the substrate boundary
        (always the ``default_sigma_nm``), and interface ``j + 1`` is the top of
        layer ``j``. Per-layer overrides fall back to ``default_sigma_nm``.
        """

        default = float(default_sigma_nm)
        layer_sigmas = self._layer_roughness_sigmas_bottom_up()
        return [default] + [
            default if sigma is None else float(sigma) for sigma in layer_sigmas
        ]

    def has_per_layer_roughness(self) -> bool:
        """Return whether any layer specifies its own roughness sigma."""

        return any(sigma is not None for sigma in self._layer_roughness_sigmas_bottom_up())

    def material_names(self, *, include_incident_medium: bool = False) -> list[str]:
        """Return ordered unique material names used by the stack.

        Args:
            include_incident_medium: When ``True``, include the incident medium
                as the first label.

        Returns:
            Ordered unique material labels. Plotting order is semantic: top cap
            first, coating materials next, substrate last.
        """

        names: list[str] = []
        if include_incident_medium:
            names.append("Incident Medium")

        if self.top_cap_material is not None and self.top_cap_thickness_nm > 0.0:
            top_cap_label = material_label(self.top_cap_material)
            if top_cap_label not in names:
                names.append(top_cap_label)

        for material_name, _ in reversed(self.layer_sequence_bottom_up()):
            label = material_label(material_name)
            if label == material_label(self.top_cap_material) if self.top_cap_material is not None else False:
                continue
            if label not in names:
                names.append(label)

        substrate_label = material_label(self.substrate_material)
        if substrate_label not in names:
            names.append(substrate_label)
        return names

    def plot_material_names(self) -> list[str]:
        """Return ordered unique material names used in the plot legend."""

        return self.material_names(include_incident_medium=False)

    def _schematic_layers_and_summary(
        self,
        *,
        max_layers: int | None = None,
    ) -> tuple[list[tuple[Any, float]], list[str]]:
        """Return schematic layers and summary lines for stack plotting.

        Args:
            max_layers: Optional maximum number of drawn layers for literal
                stack rendering.

        Returns:
            A tuple containing top-down schematic layers and summary lines.
        """

        sequence = self.layer_sequence_bottom_up()
        if not sequence:
            raise ValueError("Stack schematic requires at least one layer above the substrate.")

        summary_lines = [f"Total thickness: {self.total_thickness_nm:.3f} nm"]
        if isinstance(self, MultilayerStack):
            layers_top_down: list[tuple[Any, float]] = []
            if self.top_cap_material is not None and self.top_cap_thickness_nm > 0.0:
                layers_top_down.append((self.top_cap_material, self.top_cap_thickness_nm))
            bottom_material, top_material = self.bilayer_materials_bottom_up
            bottom_thickness_nm, top_thickness_nm = self.bilayer_thicknesses_bottom_up
            layers_top_down.extend(
                [
                    (top_material, top_thickness_nm),
                    (bottom_material, bottom_thickness_nm),
                ]
            )
            summary_lines.append(f"{self.n_bilayers} bilayers")
            summary_lines.append(f"Bilayer period: {self.d_period_nm:.3f} nm")
            return layers_top_down, summary_lines

        layers_top_down = list(reversed(sequence))
        if max_layers is not None and max_layers > 0:
            layers_top_down = layers_top_down[:max_layers]
        return layers_top_down, summary_lines

    def plot_schematic(
        self,
        output_filename: str | Path,
        *,
        max_layers: int | None = None,
    ) -> None:
        """Save a stack-only schematic plot for visual inspection.

        Args:
            output_filename: Output image path.
            max_layers: Optional maximum number of layers to draw. When provided,
                the uppermost ``max_layers`` are shown.
        """

        layers_top_down, summary_lines = self._schematic_layers_and_summary(max_layers=max_layers)

        material_labels = self.plot_material_names()
        color_map = {
            label: color
            for label, color in zip(
                material_labels,
                plt.cm.tab20.colors[: max(len(material_labels), 1)],
            )
        }
        substrate_label = material_label(self.substrate_material)
        if substrate_label not in color_map:
            color_map[substrate_label] = "#b0b0b0"

        figure_height = max(5.5, 0.45 * len(layers_top_down) + 1.8)
        figure, axis = plt.subplots(figsize=(7, figure_height))

        current_top_nm = 0.0
        for material, thickness_nm in layers_top_down:
            label = material_label(material)
            axis.barh(
                y=current_top_nm + (0.5 * thickness_nm),
                width=1.0,
                height=thickness_nm,
                left=0.0,
                color=color_map.get(label, "#cccccc"),
                edgecolor="black",
                linewidth=0.5,
            )
            axis.text(
                0.5,
                current_top_nm + (0.5 * thickness_nm),
                f"{label}\n{thickness_nm:.3f} nm",
                ha="center",
                va="center",
                fontsize=8,
            )
            current_top_nm += thickness_nm

        substrate_thickness_nm = max(10.0, 0.08 * max(current_top_nm, 1.0))
        axis.barh(
            y=current_top_nm + (0.5 * substrate_thickness_nm),
            width=1.0,
            height=substrate_thickness_nm,
            left=0.0,
            color=color_map.get(substrate_label, "#b0b0b0"),
            edgecolor="black",
            linewidth=0.5,
        )
        axis.text(
            0.5,
            current_top_nm + (0.5 * substrate_thickness_nm),
            f"{substrate_label}\nsubstrate",
            ha="center",
            va="center",
            fontsize=8,
        )

        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(current_top_nm + substrate_thickness_nm, 0.0)
        axis.set_xticks([])
        axis.set_ylabel("Depth from top surface (nm)")
        axis.set_title(f"{self.__class__.__name__} Schematic")
        axis.text(
            1.02,
            0.98,
            "\n".join(summary_lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        figure.tight_layout()
        figure.savefig(output_filename, dpi=150, bbox_inches="tight")
        plt.close(figure)


@dataclass
class SingleLayerStack(BaseStack):
    """Single-layer coating stack with optional top cap."""

    layer_material: Any = "Pt"
    layer_thickness_nm: float = 28.77
    layer_roughness_sigma_nm: float | None = None
    top_cap_roughness_sigma_nm: float | None = None

    def layer_sequence_bottom_up(self) -> list[tuple[Any, float]]:
        """Return the single-layer sequence above the substrate."""

        sequence = [(self.layer_material, self.layer_thickness_nm)]
        if self.top_cap_material is not None and self.top_cap_thickness_nm > 0.0:
            sequence.append((self.top_cap_material, self.top_cap_thickness_nm))
        return sequence

    def _layer_roughness_sigmas_bottom_up(self) -> list[float | None]:
        """Return the per-layer roughness sigmas for the single layer and cap."""

        sigmas: list[float | None] = [self.layer_roughness_sigma_nm]
        if self.top_cap_material is not None and self.top_cap_thickness_nm > 0.0:
            sigmas.append(self.top_cap_roughness_sigma_nm)
        return sigmas


@dataclass
class MultilayerStack(BaseStack):
    """Multilayer coating stack described by period, gamma, and bilayer count."""

    material_a: Any = "Cr"
    material_b: Any = "C"
    d_period_nm: float = 6.5
    gamma: float = 0.45
    n_bilayers: int = 40
    top_material: Any = "C"
    material_a_roughness_sigma_nm: float | None = None
    material_b_roughness_sigma_nm: float | None = None
    top_cap_roughness_sigma_nm: float | None = None

    def __post_init__(self) -> None:
        """Validate multilayer parameters."""

        if not _same_material(self.top_material, self.material_a) and not _same_material(
            self.top_material,
            self.material_b,
        ):
            raise ValueError("top_material must match material_a or material_b.")
        if self.d_period_nm <= 0.0:
            raise ValueError("d_period_nm must be > 0.")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1).")
        if self.n_bilayers < 1:
            raise ValueError("n_bilayers must be >= 1.")

    @property
    def material_a_thickness_nm(self) -> float:
        """Return the thickness of material A within one bilayer."""

        return self.gamma * self.d_period_nm

    @property
    def material_b_thickness_nm(self) -> float:
        """Return the thickness of material B within one bilayer."""

        return (1.0 - self.gamma) * self.d_period_nm

    @property
    def bilayer_materials_bottom_up(self) -> tuple[Any, Any]:
        """Return the material order within one bilayer from bottom to top."""

        if _same_material(self.top_material, self.material_a):
            return self.material_b, self.material_a
        return self.material_a, self.material_b

    @property
    def bilayer_thicknesses_bottom_up(self) -> tuple[float, float]:
        """Return the material thicknesses within one bilayer from bottom to top."""

        bottom_material, top_material = self.bilayer_materials_bottom_up
        bottom_thickness = (
            self.material_a_thickness_nm
            if _same_material(bottom_material, self.material_a)
            else self.material_b_thickness_nm
        )
        top_thickness = (
            self.material_a_thickness_nm
            if _same_material(top_material, self.material_a)
            else self.material_b_thickness_nm
        )
        return bottom_thickness, top_thickness

    def layer_sequence_bottom_up(self) -> list[tuple[Any, float]]:
        """Return the multilayer sequence above the substrate from bottom to top."""
        bottom_material, top_material = self.bilayer_materials_bottom_up
        bottom_thickness_nm, top_thickness_nm = self.bilayer_thicknesses_bottom_up
        sequence: list[tuple[str, float]] = []
        for _ in range(self.n_bilayers):
            sequence.append((bottom_material, bottom_thickness_nm))
            sequence.append((top_material, top_thickness_nm))
        if self.top_cap_material is not None and self.top_cap_thickness_nm > 0.0:
            sequence.append((self.top_cap_material, self.top_cap_thickness_nm))
        return sequence

    def _material_roughness_sigma(self, material: Any) -> float | None:
        """Return the per-material roughness sigma for a bilayer material."""

        if _same_material(material, self.material_a):
            return self.material_a_roughness_sigma_nm
        return self.material_b_roughness_sigma_nm

    def _layer_roughness_sigmas_bottom_up(self) -> list[float | None]:
        """Return per-layer roughness sigmas matching the generated sequence."""

        bottom_material, top_material = self.bilayer_materials_bottom_up
        bottom_sigma = self._material_roughness_sigma(bottom_material)
        top_sigma = self._material_roughness_sigma(top_material)
        sigmas: list[float | None] = []
        for _ in range(self.n_bilayers):
            sigmas.append(bottom_sigma)
            sigmas.append(top_sigma)
        if self.top_cap_material is not None and self.top_cap_thickness_nm > 0.0:
            sigmas.append(self.top_cap_roughness_sigma_nm)
        return sigmas

    def layer_specs_bottom_up(self) -> list[LayerSpec]:
        """Return the multilayer sequence as ``LayerSpec`` entries.

        Returns:
            Bottom-up layers converted to ``LayerSpec`` objects.
        """

        return [
            LayerSpec(material=material, thickness_nm=thickness_nm)
            for material, thickness_nm in self.layer_sequence_bottom_up()
        ]


def _same_material(left: Any, right: Any) -> bool:
    """Return whether two material inputs should be treated as the same layer material."""

    if left is right:
        return True
    try:
        comparison = left == right
    except Exception:
        return False
    if isinstance(comparison, bool):
        return comparison
    return False


@dataclass(frozen=True)
class LayerSpec:
    """User-facing layer descriptor used by custom stack assembly APIs.

    Args:
        material: Layer material descriptor accepted by material lookup.
        thickness_nm: Layer thickness in nanometers. Must be > 0.
        roughness_sigma_nm: Optional rms roughness in nanometers applied to this
            layer's top interface. When ``None``, the layer defers to the
            grating-level roughness default. Must be >= 0 when provided.
    """

    material: Any
    thickness_nm: float
    roughness_sigma_nm: float | None = None

    def __post_init__(self) -> None:
        """Validate layer parameters."""

        if float(self.thickness_nm) <= 0.0:
            raise ValueError("LayerSpec.thickness_nm must be > 0.")
        if self.roughness_sigma_nm is not None and float(self.roughness_sigma_nm) < 0.0:
            raise ValueError("LayerSpec.roughness_sigma_nm must be >= 0 when provided.")


@dataclass
class CustomStack(BaseStack):
    """Arbitrary user-defined stack assembled from explicit layer specs.

    This stack preserves the same plotting and grating integration pathways as
    built-in stack classes because it derives from :class:`BaseStack`.
    """

    layers_bottom_up: list[LayerSpec] | tuple[LayerSpec, ...] = ()
    top_cap_roughness_sigma_nm: float | None = None

    def __post_init__(self) -> None:
        """Validate custom layer list."""

        if len(self.layers_bottom_up) == 0:
            raise ValueError("CustomStack requires at least one layer above the substrate.")

    def layer_sequence_bottom_up(self) -> list[tuple[Any, float]]:
        """Return the user-defined sequence above the substrate."""

        sequence = [(layer.material, float(layer.thickness_nm)) for layer in self.layers_bottom_up]
        if self.top_cap_material is not None and self.top_cap_thickness_nm > 0.0:
            sequence.append((self.top_cap_material, float(self.top_cap_thickness_nm)))
        return sequence

    def _layer_roughness_sigmas_bottom_up(self) -> list[float | None]:
        """Return per-layer roughness sigmas from the explicit layer specs."""

        sigmas: list[float | None] = [layer.roughness_sigma_nm for layer in self.layers_bottom_up]
        if self.top_cap_material is not None and self.top_cap_thickness_nm > 0.0:
            sigmas.append(self.top_cap_roughness_sigma_nm)
        return sigmas

    def _schematic_layers_and_summary(
        self,
        *,
        max_layers: int | None = None,
    ) -> tuple[list[tuple[Any, float]], list[str]]:
        """Return schematic layers and summary lines for custom stacks.

        Multilayer-like repeated bilayer regions are collapsed to one
        representative bilayer while preserving explicit one-off layers such as
        interlayers and top caps.
        """

        sequence = self.layer_sequence_bottom_up()
        if not sequence:
            raise ValueError("Stack schematic requires at least one layer above the substrate.")

        repeated_block = self._detect_repeated_bilayer_block(sequence)
        if repeated_block is None:
            return super()._schematic_layers_and_summary(max_layers=max_layers)

        start_index, bilayer_count = repeated_block
        summary_lines = [f"Total thickness: {self.total_thickness_nm:.3f} nm"]
        repeated_layers = sequence[start_index : start_index + 2]
        bilayer_period_nm = float(sum(thickness for _, thickness in repeated_layers))
        explicit_layers_below = sequence[:start_index]
        explicit_layers_above = sequence[start_index + (2 * bilayer_count) :]

        layers_top_down = list(reversed(explicit_layers_above))
        layers_top_down.extend(reversed(repeated_layers))
        layers_top_down.extend(reversed(explicit_layers_below))

        summary_lines.append(f"{bilayer_count} bilayers")
        summary_lines.append(f"Bilayer period: {bilayer_period_nm:.3f} nm")
        return layers_top_down, summary_lines

    @staticmethod
    def _detect_repeated_bilayer_block(
        sequence: list[tuple[Any, float]],
    ) -> tuple[int, int] | None:
        """Return the start index and bilayer count of a repeated bilayer block.

        Args:
            sequence: Bottom-up explicit layer sequence.

        Returns:
            ``(start_index, bilayer_count)`` when a contiguous alternating
            repeated bilayer region is found, otherwise ``None``.
        """

        if len(sequence) < 4:
            return None

        for start_index in range(len(sequence) - 3):
            bottom_material, bottom_thickness = sequence[start_index]
            top_material, top_thickness = sequence[start_index + 1]
            if _same_material(bottom_material, top_material):
                continue

            bilayer_count = 1
            next_index = start_index + 2
            while next_index + 1 < len(sequence):
                next_bottom_material, next_bottom_thickness = sequence[next_index]
                next_top_material, next_top_thickness = sequence[next_index + 1]
                if not _same_material(next_bottom_material, bottom_material):
                    break
                if not _same_material(next_top_material, top_material):
                    break
                if abs(float(next_bottom_thickness) - float(bottom_thickness)) > 1e-9:
                    break
                if abs(float(next_top_thickness) - float(top_thickness)) > 1e-9:
                    break
                bilayer_count += 1
                next_index += 2

            if bilayer_count >= 2:
                return start_index, bilayer_count
        return None


def build_single_layer_stack(
    *,
    substrate_material: Any,
    layer_material: Any,
    layer_thickness_nm: float,
    top_cap_material: Any | None = None,
    top_cap_thickness_nm: float = 0.0,
    layer_roughness_sigma_nm: float | None = None,
    top_cap_roughness_sigma_nm: float | None = None,
) -> SingleLayerStack:
    """Return a user-facing helper-built :class:`SingleLayerStack`."""

    return SingleLayerStack(
        substrate_material=substrate_material,
        layer_material=layer_material,
        layer_thickness_nm=float(layer_thickness_nm),
        top_cap_material=top_cap_material,
        top_cap_thickness_nm=float(top_cap_thickness_nm),
        layer_roughness_sigma_nm=layer_roughness_sigma_nm,
        top_cap_roughness_sigma_nm=top_cap_roughness_sigma_nm,
    )


def build_multilayer_stack(
    *,
    substrate_material: Any,
    material_a: Any,
    material_b: Any,
    d_period_nm: float,
    gamma: float,
    n_bilayers: int,
    top_material: Any,
    top_cap_material: Any | None = None,
    top_cap_thickness_nm: float = 0.0,
    material_a_roughness_sigma_nm: float | None = None,
    material_b_roughness_sigma_nm: float | None = None,
    top_cap_roughness_sigma_nm: float | None = None,
) -> MultilayerStack:
    """Return a user-facing helper-built :class:`MultilayerStack`."""

    return MultilayerStack(
        substrate_material=substrate_material,
        material_a=material_a,
        material_b=material_b,
        d_period_nm=float(d_period_nm),
        gamma=float(gamma),
        n_bilayers=int(n_bilayers),
        top_material=top_material,
        top_cap_material=top_cap_material,
        top_cap_thickness_nm=float(top_cap_thickness_nm),
        material_a_roughness_sigma_nm=material_a_roughness_sigma_nm,
        material_b_roughness_sigma_nm=material_b_roughness_sigma_nm,
        top_cap_roughness_sigma_nm=top_cap_roughness_sigma_nm,
    )


def assemble_custom_stack(
    *,
    substrate_material: Any,
    layers_bottom_up: list[LayerSpec] | tuple[LayerSpec, ...],
    top_cap_material: Any | None = None,
    top_cap_thickness_nm: float = 0.0,
    top_cap_roughness_sigma_nm: float | None = None,
) -> CustomStack:
    """Assemble and return a custom stack from explicit layer specs.

    Example:
        ``assemble_custom_stack(substrate_material=si, layers_bottom_up=[LayerSpec(cr, 2.0), LayerSpec(c, 3.0)])``
    """

    return CustomStack(
        substrate_material=substrate_material,
        top_cap_material=top_cap_material,
        top_cap_thickness_nm=float(top_cap_thickness_nm),
        top_cap_roughness_sigma_nm=top_cap_roughness_sigma_nm,
        layers_bottom_up=list(layers_bottom_up),
    )
