"""Grating profile objects used by RCWA simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .materials import material_label, resolve_refractive_index, validate_material_input
from .roughness import RoughnessSpec
from .stacks import BaseStack, MultilayerStack, SingleLayerStack

# Default lateral autocorrelation length for random-interface roughness, as a
# fraction of the grating period, used when RoughnessSpec.correlation_length_nm
# is unset. period/10 keeps visible per-period structure at any pitch.
_DEFAULT_CORRELATION_PERIOD_FRACTION = 0.1


@dataclass
class BaseGrating(ABC):
    """Base class for grating profiles and material stacks.

    Attributes:
        period_lpermm: Grating period in lines per millimeter.
        coating_stack: Optional coating stack configuration.
        substrate_material: Substrate material identifier.
        layer_material: Layer material identifier.
        layer_thickness_nm: Layer thickness in nanometers.
        top_cap_material: Optional top cap material identifier.
        top_cap_thickness_nm: Top cap thickness in nanometers.
        z_resolution_nm: Vertical resolution for profile discretization.
        x_resolution_nm: Horizontal resolution for profile discretization.
        roughness: Optional roughness model selected at grating construction time.
    """

    period_lpermm: int = 400
    coating_stack: BaseStack | None = None
    substrate_material: Any = "Si"
    layer_material: Any = "Pt"
    layer_thickness_nm: float = 28.77
    top_cap_material: Any | None = None
    top_cap_thickness_nm: float = 0.0
    z_resolution_nm: float = 0.1
    x_resolution_nm: float = 1.0
    roughness: RoughnessSpec | None = None

    @property
    def period_nm(self) -> float:
        """Return the grating period in nanometers."""

        return 1e6 / self.period_lpermm

    @abstractmethod
    def profile_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one grating period as x and height arrays."""

    @abstractmethod
    def profile_depth_nm(self) -> float:
        """Return the grating profile depth in nanometers."""

    def _tiled_profile_points(self, num_periods: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Return profile points repeated over multiple periods.

        Args:
            num_periods: Number of periods to tile.

        Returns:
            Tiled x positions and heights.
        """

        if num_periods < 1:
            raise ValueError("num_periods must be at least 1.")

        base_positions, base_heights = self.profile_points()
        tiled_positions = []
        tiled_heights = []
        for period_index in range(num_periods):
            offset = period_index * self.period_nm
            if period_index == 0:
                tiled_positions.extend((base_positions + offset).tolist())
                tiled_heights.extend(base_heights.tolist())
            else:
                tiled_positions.extend((base_positions[1:] + offset).tolist())
                tiled_heights.extend(base_heights[1:].tolist())
        return np.asarray(tiled_positions, dtype=float), np.asarray(tiled_heights, dtype=float)

    def resolved_stack(self) -> BaseStack:
        """Return the common coating stack used by the grating."""

        if self.coating_stack is not None:
            return self.coating_stack
        return SingleLayerStack(
            substrate_material=self.substrate_material,
            layer_material=self.layer_material,
            layer_thickness_nm=self.layer_thickness_nm,
            top_cap_material=self.top_cap_material,
            top_cap_thickness_nm=self.top_cap_thickness_nm,
        )

    def _material_plot_data(
        self,
        num_periods: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Return discretized material data for profile visualization.

        Args:
            num_periods: Number of periods to include.

        Returns:
            X grid, z grid, material-code map, and material labels.
        """

        coating_stack = self.resolved_stack()
        x_grid = self._build_x_grid(num_periods=num_periods)
        z_grid = self._build_plot_z_grid(coating_stack)
        surface = self._surface_profile_on_grid(x_grid, num_periods=num_periods)
        material_labels = coating_stack.plot_material_names()
        material_codes = {label: index for index, label in enumerate(material_labels)}
        material_map = self._build_material_code_grid(
            x_grid=x_grid,
            z_grid=z_grid,
            surface=surface,
            coating_stack=coating_stack,
            material_codes=material_codes,
            include_incident_medium=False,
        )

        return x_grid, z_grid, material_map, material_labels

    def plot_profile(self, output_filename: str | Path) -> None:
        """Save a plot showing the profile and the material stack."""

        positions, heights = self._tiled_profile_points(num_periods=3)
        x_grid, z_grid, material_map, material_labels = self._material_plot_data(num_periods=3)

        figure, axes = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        profile_axis, material_axis = axes

        profile_axis.fill_between(
            positions,
            0.0,
            heights,
            color="tab:blue",
            alpha=0.35,
            label="Grating groove",
        )
        profile_axis.plot(positions, heights, color="tab:blue", linewidth=2.0)
        profile_axis.set_xlim(0.0, 3.0 * self.period_nm)
        profile_axis.set_ylim(0.0, max(float(np.max(heights)) * 1.15, 1.0))
        profile_axis.set_ylabel("Depth (nm)")
        profile_axis.set_title(f"{self.__class__.__name__} Profile (Three Periods)")
        profile_axis.grid(True, alpha=0.3)
        profile_axis.legend(loc="best")

        material_colors = self._plot_material_colors(coating_stack=self.resolved_stack(), material_labels=material_labels)
        color_map = plt.matplotlib.colors.ListedColormap(material_colors)
        color_map.set_bad(color="#ffffff", alpha=1.0)
        masked_material_map = np.ma.masked_less(material_map, 0)
        material_axis.imshow(
            masked_material_map,
            origin="lower",
            aspect="auto",
            extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
            cmap=color_map,
            interpolation="nearest",
            vmin=0,
            vmax=max(len(material_labels) - 1, 1),
        )
        legend_handles = [
            plt.matplotlib.patches.Patch(color=color_map.colors[index], label=label)
            for index, label in enumerate(material_labels)
        ]
        material_axis.set_xlabel("x (nm)")
        material_axis.set_ylabel("z (nm)")
        material_axis.set_title("Material Stack")
        material_axis.legend(handles=legend_handles, loc="upper right")
        material_axis.grid(False)

        figure.tight_layout()
        figure.savefig(output_filename, dpi=150, bbox_inches="tight")
        plt.close(figure)

    def save_structure_debug_data(
        self,
        photon_energy_ev: float,
        output_directory: str | Path,
        *,
        num_periods: int = 1,
        n_inc: complex = 1.0 + 0.0j,
    ) -> None:
        """Write discretized structure debug data to CSV files.

        Args:
            photon_energy_ev: Photon energy used to interpolate optical constants.
            output_directory: Destination directory for the CSV files.
            num_periods: Number of grating periods to export in x.
            n_inc: Refractive index of the incident medium.
        """

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        coating_stack = self.resolved_stack()
        x_grid = self._build_x_grid(num_periods=num_periods)
        z_grid = self._build_solver_z_grid(coating_stack)
        surface = self._surface_profile_on_grid(x_grid, num_periods=num_periods)
        material_labels = coating_stack.material_names(include_incident_medium=True)
        material_codes = {label: index for index, label in enumerate(material_labels)}
        material_map = self._build_material_code_grid(
            x_grid=x_grid,
            z_grid=z_grid,
            surface=surface,
            coating_stack=coating_stack,
            material_codes=material_codes,
            include_incident_medium=True,
        )
        index_grid = self._build_refractive_index_grid(
            x_grid=x_grid,
            z_grid=z_grid,
            surface=surface,
            coating_stack=coating_stack,
            photon_energy_ev=photon_energy_ev,
            n_inc=n_inc,
        )

        np.savetxt(output_path / "x.csv", x_grid, delimiter=",")
        np.savetxt(output_path / "z.csv", z_grid, delimiter=",")
        np.savetxt(output_path / "surface.csv", surface, delimiter=",")
        np.savetxt(output_path / "material_id.csv", material_map, fmt="%d", delimiter=",")
        np.savetxt(output_path / "index_real.csv", np.real(index_grid), delimiter=",")
        np.savetxt(output_path / "index_imag.csv", np.imag(index_grid), delimiter=",")
        (output_path / "material_labels.txt").write_text("\n".join(material_labels), encoding="utf-8")

    def build_textures(
        self,
        photon_energy_ev: float,
        *,
        n_inc: complex = 1.0 + 0.0j,
        _memory_mode: str = "low_memory",
    ) -> tuple[list[object], tuple[np.ndarray, np.ndarray]]:
        """Build textures and profile arrays for the discretized profile.

        Args:
            photon_energy_ev: Photon energy used to resolve optical constants.
            n_inc: Incident-medium refractive index.
            _memory_mode: Internal texture-generation mode. ``"low_memory"``
                generates one solver row at a time and compresses consecutive
                identical rows before RCWA conversion. ``"legacy_dense"`` keeps
                the older dense-grid path available for regression/debug use.

        Returns:
            Texture descriptors and the RCWA profile tuple.
        """

        if _memory_mode not in {"legacy_dense", "low_memory"}:
            raise ValueError("memory_mode must be 'low_memory' or 'legacy_dense'.")
        self._validate_simulation_materials()
        self._warn_if_roughness_underresolved()

        if _memory_mode == "low_memory":
            return self._build_textures_low_memory(
                photon_energy_ev,
                n_inc=n_inc,
            )
        return self._build_textures_legacy_dense(
            photon_energy_ev,
            n_inc=n_inc,
        )

    def _validate_simulation_materials(self) -> None:
        """Validate stack materials before resolving optical constants."""

        coating_stack = self.resolved_stack()
        validate_material_input(coating_stack.substrate_material, field_name="substrate_material")
        if isinstance(coating_stack, MultilayerStack):
            validate_material_input(coating_stack.material_a, field_name="material_a")
            validate_material_input(coating_stack.material_b, field_name="material_b")
        else:
            for layer_index, (material_name, _) in enumerate(coating_stack.layer_sequence_bottom_up()):
                field_name = "layer_material" if layer_index == 0 else f"layer_sequence[{layer_index}]"
                validate_material_input(material_name, field_name=field_name)
        if coating_stack.top_cap_material is not None and coating_stack.top_cap_thickness_nm > 0.0:
            validate_material_input(coating_stack.top_cap_material, field_name="top_cap_material")

    def _build_textures_legacy_dense(
        self,
        photon_energy_ev: float,
        *,
        n_inc: complex,
    ) -> tuple[list[object], tuple[np.ndarray, np.ndarray]]:
        """Build textures using the dense 2D refractive-index grid path."""

        coating_stack = self.resolved_stack()
        n_sub = resolve_refractive_index(
            coating_stack.substrate_material,
            photon_energy_ev,
        )
        x = self._build_x_grid(num_periods=1)
        z = self._build_solver_z_grid(coating_stack)
        surface = self._surface_profile_on_grid(x, num_periods=1)
        index_grid = self._build_refractive_index_grid(
            x_grid=x,
            z_grid=z,
            surface=surface,
            coating_stack=coating_stack,
            photon_energy_ev=photon_energy_ev,
            n_inc=n_inc,
        )

        textures: list[object] = [n_inc]
        for row in range(index_grid.shape[0]):
            row_values = index_grid[row]
            row_changes = np.nonzero(np.diff(row_values) != 0)[0]
            if len(row_changes) == 0:
                textures.append(complex(row_values[0]))
            else:
                textures.append([x[row_changes + 1], row_values[row_changes]])

        textures.append(n_sub)
        thicknesses = np.concatenate(
            (
                np.asarray([0.0], dtype=float),
                np.full(index_grid.shape[0], self.z_resolution_nm, dtype=float),
                np.asarray([0.0], dtype=float),
            )
        )
        texture_sequence = np.arange(len(textures), dtype=int)
        return textures, (np.asarray(thicknesses, dtype=float), np.asarray(texture_sequence, dtype=int))

    def _build_textures_low_memory(
        self,
        photon_energy_ev: float,
        *,
        n_inc: complex,
    ) -> tuple[list[object], tuple[np.ndarray, np.ndarray]]:
        """Build textures row-by-row without materializing a full 2D grid."""

        coating_stack = self.resolved_stack()
        n_sub = resolve_refractive_index(coating_stack.substrate_material, photon_energy_ev)
        x_grid = self._build_x_grid(num_periods=1)
        z_grid = self._build_solver_z_grid(coating_stack)
        surface = self._surface_profile_on_grid(x_grid, num_periods=1)
        texture_registry: dict[tuple[object, ...], int] = {}
        textures: list[object] = []
        prepared_multilayer = None
        if isinstance(coating_stack, MultilayerStack) and not (
            self.roughness is not None and self.roughness.kind == "random-interface"
        ):
            prepared_multilayer = _prepare_multilayer_texture_builder(
                x_grid=x_grid,
                surface=surface,
                coating_stack=coating_stack,
                photon_energy_ev=photon_energy_ev,
                n_inc=complex(n_inc),
                n_sub=complex(n_sub),
            )

        def register_texture(texture: object, *, signature: tuple[object, ...] | None = None) -> int:
            texture_signature = (
                self._texture_descriptor_signature(texture)
                if signature is None
                else signature
            )
            if texture_signature in texture_registry:
                return texture_registry[texture_signature]
            texture_index = len(textures)
            textures.append(texture)
            texture_registry[texture_signature] = texture_index
            return texture_index

        incident_index = register_texture(complex(n_inc))
        profile_thicknesses: list[float] = [0.0]
        profile_indices: list[int] = [incident_index]

        for z_value in z_grid:
            if prepared_multilayer is not None:
                row_texture, row_signature = _texture_descriptor_for_prepared_multilayer_row(
                    z_value=float(z_value),
                    prepared=prepared_multilayer,
                )
                row_index = register_texture(row_texture, signature=row_signature)
            else:
                row_texture = self._texture_descriptor_for_row(
                    z_value=float(z_value),
                    x_grid=x_grid,
                    surface=surface,
                    coating_stack=coating_stack,
                    photon_energy_ev=photon_energy_ev,
                    n_inc=complex(n_inc),
                    n_sub=complex(n_sub),
                )
                row_index = register_texture(row_texture)
            if len(profile_indices) > 1 and profile_indices[-1] == row_index:
                profile_thicknesses[-1] += float(self.z_resolution_nm)
            else:
                profile_thicknesses.append(float(self.z_resolution_nm))
                profile_indices.append(row_index)

        substrate_index = register_texture(complex(n_sub))
        profile_thicknesses.append(0.0)
        profile_indices.append(substrate_index)
        return textures, (
            np.asarray(profile_thicknesses, dtype=float),
            np.asarray(profile_indices, dtype=int),
        )

    def _texture_descriptor_for_row(
        self,
        *,
        z_value: float,
        x_grid: np.ndarray,
        surface: np.ndarray,
        coating_stack: BaseStack,
        photon_energy_ev: float,
        n_inc: complex,
        n_sub: complex,
    ) -> object:
        """Return one RCWA texture descriptor for a single solver row."""

        row_values = self._refractive_index_row(
            z_value=z_value,
            x_grid=x_grid,
            surface=surface,
            coating_stack=coating_stack,
            photon_energy_ev=photon_energy_ev,
            n_inc=n_inc,
            n_sub=n_sub,
        )
        row_changes = np.nonzero(np.diff(row_values) != 0)[0]
        if len(row_changes) == 0:
            return complex(row_values[0])
        return [x_grid[row_changes + 1], row_values[row_changes]]

    def _texture_descriptor_signature(self, texture: object) -> tuple[object, ...]:
        """Return a hashable signature for one texture descriptor."""

        if not isinstance(texture, (list, tuple)):
            value = complex(texture)
            return ("homogeneous", round(float(np.real(value)), 12), round(float(np.imag(value)), 12))
        x_positions = np.asarray(texture[0], dtype=float).ravel()
        n_left = np.asarray(texture[1], dtype=complex).ravel()
        return (
            "patterned",
            tuple(np.round(x_positions, 12)),
            tuple(
                (round(float(np.real(value)), 12), round(float(np.imag(value)), 12))
                for value in n_left
            ),
        )

    def _refractive_index_row(
        self,
        *,
        z_value: float,
        x_grid: np.ndarray,
        surface: np.ndarray,
        coating_stack: BaseStack,
        photon_energy_ev: float,
        n_inc: complex,
        n_sub: complex,
    ) -> np.ndarray:
        """Return refractive indices for one solver row."""

        row_values = np.full(surface.shape, n_sub, dtype=complex)

        if isinstance(coating_stack, MultilayerStack):
            n_material_a = resolve_refractive_index(coating_stack.material_a, photon_energy_ev)
            n_material_b = resolve_refractive_index(coating_stack.material_b, photon_energy_ev)
            bottom_material, top_material = coating_stack.bilayer_materials_bottom_up
            bottom_thickness_nm, _ = coating_stack.bilayer_thicknesses_bottom_up
            stack_base = surface + 1.0
            for bilayer_index in range(coating_stack.n_bilayers):
                lower_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * bilayer_index,
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index,
                )
                middle_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * bilayer_index + bottom_thickness_nm,
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index + 1,
                )
                upper_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * (bilayer_index + 1),
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index + 2,
                )
                lower_mask = (z_value >= lower_interface) & (z_value < middle_interface)
                upper_mask = (z_value >= middle_interface) & (z_value < upper_interface)
                if np.any(lower_mask):
                    row_values[lower_mask] = (
                        n_material_a if _material_matches(bottom_material, coating_stack.material_a) else n_material_b
                    )
                if np.any(upper_mask):
                    row_values[upper_mask] = (
                        n_material_a if _material_matches(top_material, coating_stack.material_a) else n_material_b
                    )
            current_top = self._rough_interface(
                stack_base + coating_stack.d_period_nm * coating_stack.n_bilayers,
                x_grid=x_grid,
                interface_index=2 * coating_stack.n_bilayers,
            )
            if coating_stack.top_cap_material is not None and coating_stack.top_cap_thickness_nm > 0.0:
                n_top_cap = resolve_refractive_index(coating_stack.top_cap_material, photon_energy_ev)
                top_cap_upper = self._rough_interface(
                    stack_base
                    + coating_stack.d_period_nm * coating_stack.n_bilayers
                    + coating_stack.top_cap_thickness_nm,
                    x_grid=x_grid,
                    interface_index=2 * coating_stack.n_bilayers + 1,
                )
                top_cap_mask = (z_value >= current_top) & (z_value < top_cap_upper)
                if np.any(top_cap_mask):
                    row_values[top_cap_mask] = n_top_cap
                current_top = top_cap_upper
            incident_mask = z_value >= current_top
            if np.any(incident_mask):
                row_values[incident_mask] = n_inc
            return row_values

        cumulative_thickness_nm = 0.0
        current_lower = self._rough_interface(
            surface + cumulative_thickness_nm,
            x_grid=x_grid,
            interface_index=0,
        )
        for layer_index, (material_name, layer_thickness_nm) in enumerate(coating_stack.layer_sequence_bottom_up()):
            refractive_index = resolve_refractive_index(material_name, photon_energy_ev)
            cumulative_thickness_nm += layer_thickness_nm
            current_upper = self._rough_interface(
                surface + cumulative_thickness_nm,
                x_grid=x_grid,
                interface_index=layer_index + 1,
            )
            layer_mask = (z_value >= current_lower) & (z_value < current_upper)
            if np.any(layer_mask):
                row_values[layer_mask] = refractive_index
            current_lower = current_upper
        incident_mask = z_value >= current_lower
        if np.any(incident_mask):
            row_values[incident_mask] = n_inc
        return row_values

    def _build_x_grid(self, *, num_periods: int) -> np.ndarray:
        """Return the x grid used for discretized structure generation."""

        x_span_nm = num_periods * self.period_nm
        return np.linspace(0.0, x_span_nm, int(round(x_span_nm / self.x_resolution_nm)) + 1)

    def _build_solver_z_grid(self, coating_stack: BaseStack) -> np.ndarray:
        """Return the solver z grid, matching the Octave helper orientation."""

        if isinstance(coating_stack, MultilayerStack):
            thickness_nm = self.profile_depth_nm() + coating_stack.d_period_nm * coating_stack.n_bilayers
            thickness_nm += coating_stack.top_cap_thickness_nm + 25.0 * self.z_resolution_nm
        else:
            thickness_nm = self.profile_depth_nm() + coating_stack.total_thickness_nm + 5.0
        return np.linspace(thickness_nm, 0.0, int(round(thickness_nm / self.z_resolution_nm)) + 1)

    def _build_plot_z_grid(self, coating_stack: BaseStack) -> np.ndarray:
        """Return the plot z grid with increasing depth."""

        solver_z = self._build_solver_z_grid(coating_stack)
        return solver_z[::-1]

    def _surface_profile_on_grid(self, x_grid: np.ndarray, *, num_periods: int) -> np.ndarray:
        """Return the surface profile interpolated on the requested x grid."""

        positions, heights = self._tiled_profile_points(num_periods=num_periods)
        return np.interp(x_grid, positions, heights)

    def _interface_sigmas(self) -> list[float]:
        """Return the per-interface roughness sigmas for the resolved stack.

        Interface 0 is the substrate boundary; interface ``j + 1`` is the top of
        layer ``j``. Layers without an explicit sigma fall back to the
        grating-level ``roughness.sigma_nm``. The result is cached because it is
        consulted once per interface per solver row.
        """

        cached = self.__dict__.get("_interface_sigmas_cache")
        if cached is not None:
            return cached
        default = float(self.roughness.sigma_nm) if self.roughness is not None else 0.0
        sigmas = self.resolved_stack().interface_roughness_sigmas_bottom_up(default)
        self.__dict__["_interface_sigmas_cache"] = sigmas
        return sigmas

    def _warn_if_roughness_underresolved(self) -> None:
        """Warn when enabled roughness is finer than the configured grid."""

        if self.roughness is None:
            return
        max_sigma_nm = max(self._interface_sigmas(), default=0.0)
        if max_sigma_nm == 0.0:
            return
        threshold_nm = max_sigma_nm / self.roughness.resolution_factor
        underresolved_axes = []
        if self.x_resolution_nm >= threshold_nm:
            underresolved_axes.append(f"x_resolution_nm={self.x_resolution_nm:g}")
        if self.z_resolution_nm >= threshold_nm:
            underresolved_axes.append(f"z_resolution_nm={self.z_resolution_nm:g}")
        if not underresolved_axes:
            return
        warnings.warn(
            "Roughness may be underresolved: "
            + ", ".join(underresolved_axes)
            + f" should be smaller than roughness/factor={threshold_nm:g} nm.",
            UserWarning,
            stacklevel=3,
        )

    def _roughness_correlation_length_nm(self) -> float:
        """Return the resolved lateral autocorrelation length in nanometers.

        Falls back to ``period / 10`` when ``correlation_length_nm`` is unset so
        the roughness always keeps visible per-period structure.
        """

        if self.roughness is None:
            return 0.0
        if self.roughness.correlation_length_nm is None:
            return self.period_nm * _DEFAULT_CORRELATION_PERIOD_FRACTION
        return float(self.roughness.correlation_length_nm)

    def _roughness_interface_offset(self, x_grid: np.ndarray, interface_index: int) -> np.ndarray:
        """Return the deterministic roughness offset for one interface.

        The offset is a periodic Gaussian random field with a Gaussian
        autocorrelation of length ``correlation_length_nm`` (see
        :class:`~grax.roughness.RoughnessSpec`). A correlation length of ``0``
        reproduces the legacy uncorrelated (white-noise) interface.
        """

        if self.roughness is None or self.roughness.kind != "random-interface":
            return np.zeros_like(x_grid, dtype=float)
        sigmas = self._interface_sigmas()
        sigma_nm = (
            sigmas[interface_index]
            if 0 <= interface_index < len(sigmas)
            else float(self.roughness.sigma_nm)
        )
        if sigma_nm == 0.0:
            return np.zeros_like(x_grid, dtype=float)
        rng = np.random.default_rng(int(self.roughness.seed) + int(interface_index))
        correlation_length_nm = self._roughness_correlation_length_nm()
        offsets = self._roughness_random_field(x_grid, rng, correlation_length_nm)
        offsets = offsets - float(np.mean(offsets))
        rms = float(np.sqrt(np.mean(offsets**2)))
        if rms == 0.0:
            return np.zeros_like(x_grid, dtype=float)
        return offsets * (sigma_nm / rms)

    def _roughness_random_field(
        self,
        x_grid: np.ndarray,
        rng: np.random.Generator,
        correlation_length_nm: float,
    ) -> np.ndarray:
        """Return a zero-mean periodic random field on one grating period.

        When ``correlation_length_nm > 0`` the field is synthesised from a
        Gaussian power spectrum (Gaussian autocorrelation). When it is ``0`` an
        uncorrelated white-noise field is returned. The last sample duplicates
        the first so the field is periodic over the grating period.
        """

        num_samples = x_grid.size
        if correlation_length_nm <= 0.0 or num_samples < 3:
            offsets = rng.normal(loc=0.0, scale=1.0, size=num_samples)
            if offsets.size > 1:
                offsets[-1] = offsets[0]
            return offsets

        # x_grid[-1] duplicates x_grid[0] of the next period, so synthesise on the
        # N unique samples and append the first to restore periodicity.
        n_unique = num_samples - 1
        dx_nm = self.period_nm / n_unique
        k = 2.0 * np.pi * np.fft.rfftfreq(n_unique, d=dx_nm)
        amplitude = np.exp(-0.25 * (k * correlation_length_nm) ** 2)
        spectrum = (
            rng.normal(size=k.size) + 1j * rng.normal(size=k.size)
        ) * amplitude
        spectrum[0] = 0.0  # zero mean
        field = np.fft.irfft(spectrum, n=n_unique)
        return np.concatenate([field, field[:1]])

    def _rough_interface(
        self,
        base_interface_nm: np.ndarray,
        *,
        x_grid: np.ndarray,
        interface_index: int,
    ) -> np.ndarray:
        """Return one material interface with optional grating-level roughness."""

        return np.asarray(base_interface_nm, dtype=float) + self._roughness_interface_offset(
            x_grid,
            interface_index,
        )

    def _build_material_code_grid(
        self,
        *,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        surface: np.ndarray,
        coating_stack: BaseStack,
        material_codes: dict[str, int],
        include_incident_medium: bool,
    ) -> np.ndarray:
        """Return the discretized material-code grid for plotting."""

        substrate_code = material_codes[material_label(coating_stack.substrate_material)]
        z_mesh = np.repeat(z_grid[:, None], x_grid.size, axis=1)
        material_map = np.full((z_grid.size, x_grid.size), substrate_code, dtype=int)
        incident_code = material_codes["Incident Medium"] if include_incident_medium else -1

        if isinstance(coating_stack, MultilayerStack):
            bottom_material, top_material = coating_stack.bilayer_materials_bottom_up
            bottom_thickness_nm, top_thickness_nm = coating_stack.bilayer_thicknesses_bottom_up
            stack_base = surface + 1.0
            for bilayer_index in range(coating_stack.n_bilayers):
                lower_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * bilayer_index,
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index,
                )
                middle_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * bilayer_index + bottom_thickness_nm,
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index + 1,
                )
                upper_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * (bilayer_index + 1),
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index + 2,
                )
                lower_mask = (z_mesh >= lower_interface[None, :]) & (z_mesh < middle_interface[None, :])
                upper_mask = (z_mesh >= middle_interface[None, :]) & (z_mesh < upper_interface[None, :])
                material_map[lower_mask] = material_codes[material_label(bottom_material)]
                material_map[upper_mask] = material_codes[material_label(top_material)]

            current_top = self._rough_interface(
                stack_base + coating_stack.d_period_nm * coating_stack.n_bilayers,
                x_grid=x_grid,
                interface_index=2 * coating_stack.n_bilayers,
            )
            if (
                coating_stack.top_cap_material is not None
                and coating_stack.top_cap_thickness_nm > 0.0
            ):
                top_cap_upper = self._rough_interface(
                    stack_base
                    + coating_stack.d_period_nm * coating_stack.n_bilayers
                    + coating_stack.top_cap_thickness_nm,
                    x_grid=x_grid,
                    interface_index=2 * coating_stack.n_bilayers + 1,
                )
                top_cap_mask = (z_mesh >= current_top[None, :]) & (z_mesh < top_cap_upper[None, :])
                material_map[top_cap_mask] = material_codes[
                    material_label(coating_stack.top_cap_material)
                ]
                current_top = top_cap_upper

            if include_incident_medium:
                material_map[z_mesh >= current_top[None, :]] = incident_code
            else:
                material_map[z_mesh >= current_top[None, :]] = -1
            return material_map

        cumulative_thickness_nm = 0.0
        current_lower = self._rough_interface(
            surface + cumulative_thickness_nm,
            x_grid=x_grid,
            interface_index=0,
        )
        for layer_index, (material_name, layer_thickness_nm) in enumerate(coating_stack.layer_sequence_bottom_up()):
            cumulative_thickness_nm += layer_thickness_nm
            current_upper = self._rough_interface(
                surface + cumulative_thickness_nm,
                x_grid=x_grid,
                interface_index=layer_index + 1,
            )
            layer_mask = (z_mesh >= current_lower[None, :]) & (z_mesh < current_upper[None, :])
            material_map[layer_mask] = material_codes[material_label(material_name)]
            current_lower = current_upper
        if include_incident_medium:
            material_map[z_mesh >= current_lower[None, :]] = incident_code
        else:
            material_map[z_mesh >= current_lower[None, :]] = -1
        return material_map

    def _plot_material_colors(
        self,
        *,
        coating_stack: BaseStack,
        material_labels: list[str],
    ) -> list[str]:
        """Return deterministic unique colors for ordered material labels."""

        del coating_stack
        unique_labels = list(dict.fromkeys(material_labels))
        base_palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
        if not base_palette:
            return ["#808080" for _ in material_labels]

        palette_size = len(base_palette)
        assigned: dict[str, str] = {}
        used_indices: set[int] = set()
        for label in unique_labels:
            digest = hashlib.sha256(label.encode("utf-8")).digest()
            start_index = int.from_bytes(digest[:8], byteorder="big", signed=False) % palette_size
            candidate_index = start_index
            while candidate_index in used_indices:
                candidate_index = (candidate_index + 1) % palette_size
                if candidate_index == start_index:
                    break
            assigned[label] = base_palette[candidate_index]
            used_indices.add(candidate_index)

        return [assigned[label] for label in material_labels]

    def _build_refractive_index_grid(
        self,
        *,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        surface: np.ndarray,
        coating_stack: BaseStack,
        photon_energy_ev: float,
        n_inc: complex,
    ) -> np.ndarray:
        """Return the refractive-index grid used to derive the textures."""

        n_sub = resolve_refractive_index(
            coating_stack.substrate_material,
            photon_energy_ev,
        )
        z_mesh = np.repeat(z_grid[:, None], x_grid.size, axis=1)
        index_grid = np.full((z_grid.size, x_grid.size), n_sub, dtype=complex)

        if isinstance(coating_stack, MultilayerStack):
            n_material_a = resolve_refractive_index(
                coating_stack.material_a,
                photon_energy_ev,
            )
            n_material_b = resolve_refractive_index(
                coating_stack.material_b,
                photon_energy_ev,
            )
            refractive_index_by_material = [
                (coating_stack.material_a, n_material_a),
                (coating_stack.material_b, n_material_b),
            ]
            bottom_material, top_material = coating_stack.bilayer_materials_bottom_up
            bottom_thickness_nm, _ = coating_stack.bilayer_thicknesses_bottom_up
            stack_base = surface + 1.0
            for bilayer_index in range(coating_stack.n_bilayers):
                lower_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * bilayer_index,
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index,
                )
                middle_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * bilayer_index + bottom_thickness_nm,
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index + 1,
                )
                upper_interface = self._rough_interface(
                    stack_base + coating_stack.d_period_nm * (bilayer_index + 1),
                    x_grid=x_grid,
                    interface_index=2 * bilayer_index + 2,
                )
                lower_mask = (z_mesh >= lower_interface[None, :]) & (z_mesh < middle_interface[None, :])
                upper_mask = (z_mesh >= middle_interface[None, :]) & (z_mesh < upper_interface[None, :])
                index_grid[lower_mask] = _index_for_material(
                    bottom_material,
                    refractive_index_by_material,
                )
                index_grid[upper_mask] = _index_for_material(
                    top_material,
                    refractive_index_by_material,
                )

            current_top = self._rough_interface(
                stack_base + coating_stack.d_period_nm * coating_stack.n_bilayers,
                x_grid=x_grid,
                interface_index=2 * coating_stack.n_bilayers,
            )
            if (
                coating_stack.top_cap_material is not None
                and coating_stack.top_cap_thickness_nm > 0.0
            ):
                n_top_cap = resolve_refractive_index(
                    coating_stack.top_cap_material,
                    photon_energy_ev,
                )
                top_cap_upper = self._rough_interface(
                    stack_base
                    + coating_stack.d_period_nm * coating_stack.n_bilayers
                    + coating_stack.top_cap_thickness_nm,
                    x_grid=x_grid,
                    interface_index=2 * coating_stack.n_bilayers + 1,
                )
                top_cap_mask = (z_mesh >= current_top[None, :]) & (z_mesh < top_cap_upper[None, :])
                index_grid[top_cap_mask] = n_top_cap
                current_top = top_cap_upper

            index_grid[z_mesh >= current_top[None, :]] = n_inc
            return index_grid

        cumulative_thickness_nm = 0.0
        current_lower = self._rough_interface(
            surface + cumulative_thickness_nm,
            x_grid=x_grid,
            interface_index=0,
        )
        for layer_index, (material_name, layer_thickness_nm) in enumerate(coating_stack.layer_sequence_bottom_up()):
            refractive_index = resolve_refractive_index(
                material_name,
                photon_energy_ev,
            )
            cumulative_thickness_nm += layer_thickness_nm
            current_upper = self._rough_interface(
                surface + cumulative_thickness_nm,
                x_grid=x_grid,
                interface_index=layer_index + 1,
            )
            layer_mask = (z_mesh >= current_lower[None, :]) & (z_mesh < current_upper[None, :])
            index_grid[layer_mask] = refractive_index
            current_lower = current_upper
        index_grid[z_mesh >= current_lower[None, :]] = n_inc
        return index_grid


def _index_for_material(
    material: Any,
    refractive_index_by_material: list[tuple[Any, complex]],
) -> complex:
    """Return a refractive index from a small material/index lookup list."""

    for candidate, refractive_index in refractive_index_by_material:
        if material is candidate:
            return refractive_index
        try:
            is_match = material == candidate
        except Exception:
            is_match = False
        if isinstance(is_match, bool) and is_match:
            return refractive_index
    raise KeyError(f"Unable to resolve refractive index for {material_label(material)}.")


@dataclass(frozen=True)
class _PreparedMultilayerTextureBuilder:
    """Precomputed constants for one multilayer low-memory texture build."""

    x_grid: np.ndarray
    surface_base_nm: np.ndarray
    bilayer_period_nm: float
    stack_height_nm: float
    bottom_thickness_nm: float
    top_cap_thickness_nm: float
    n_sub: complex
    n_inc: complex
    n_bottom: complex
    n_top: complex
    n_top_cap: complex | None


def _prepare_multilayer_texture_builder(
    *,
    x_grid: np.ndarray,
    surface: np.ndarray,
    coating_stack: MultilayerStack,
    photon_energy_ev: float,
    n_inc: complex,
    n_sub: complex,
) -> _PreparedMultilayerTextureBuilder:
    """Return cached constants for one multilayer texture-generation pass."""

    n_material_a = resolve_refractive_index(coating_stack.material_a, photon_energy_ev)
    n_material_b = resolve_refractive_index(coating_stack.material_b, photon_energy_ev)
    refractive_index_by_material = [
        (coating_stack.material_a, n_material_a),
        (coating_stack.material_b, n_material_b),
    ]
    bottom_material, top_material = coating_stack.bilayer_materials_bottom_up
    bottom_thickness_nm, _ = coating_stack.bilayer_thicknesses_bottom_up
    n_top_cap = None
    if coating_stack.top_cap_material is not None and coating_stack.top_cap_thickness_nm > 0.0:
        n_top_cap = resolve_refractive_index(coating_stack.top_cap_material, photon_energy_ev)
    return _PreparedMultilayerTextureBuilder(
        x_grid=np.asarray(x_grid, dtype=float),
        surface_base_nm=np.asarray(surface, dtype=float) + 1.0,
        bilayer_period_nm=float(coating_stack.d_period_nm),
        stack_height_nm=float(coating_stack.d_period_nm * coating_stack.n_bilayers),
        bottom_thickness_nm=float(bottom_thickness_nm),
        top_cap_thickness_nm=float(coating_stack.top_cap_thickness_nm),
        n_sub=complex(n_sub),
        n_inc=complex(n_inc),
        n_bottom=_index_for_material(bottom_material, refractive_index_by_material),
        n_top=_index_for_material(top_material, refractive_index_by_material),
        n_top_cap=None if n_top_cap is None else complex(n_top_cap),
    )


def _texture_descriptor_for_prepared_multilayer_row(
    *,
    z_value: float,
    prepared: _PreparedMultilayerTextureBuilder,
) -> tuple[object, tuple[object, ...]]:
    """Return one row texture descriptor and its signature for a multilayer row."""

    row_values = np.full(prepared.surface_base_nm.shape, prepared.n_sub, dtype=complex)
    relative_height_nm = float(z_value) - prepared.surface_base_nm
    multilayer_mask = (relative_height_nm >= 0.0) & (relative_height_nm < prepared.stack_height_nm)
    if np.any(multilayer_mask):
        bilayer_phase_nm = np.mod(relative_height_nm[multilayer_mask], prepared.bilayer_period_nm)
        row_values[multilayer_mask] = np.where(
            bilayer_phase_nm < prepared.bottom_thickness_nm,
            prepared.n_bottom,
            prepared.n_top,
        )
    if prepared.top_cap_thickness_nm > 0.0 and prepared.n_top_cap is not None:
        top_cap_mask = (
            (relative_height_nm >= prepared.stack_height_nm)
            & (relative_height_nm < prepared.stack_height_nm + prepared.top_cap_thickness_nm)
        )
        if np.any(top_cap_mask):
            row_values[top_cap_mask] = prepared.n_top_cap
    incident_mask = relative_height_nm >= prepared.stack_height_nm + prepared.top_cap_thickness_nm
    if np.any(incident_mask):
        row_values[incident_mask] = prepared.n_inc

    row_changes = np.flatnonzero(np.diff(row_values) != 0)
    if row_changes.size == 0:
        value = complex(row_values[0])
        return value, ("homogeneous", round(float(np.real(value)), 12), round(float(np.imag(value)), 12))

    x_positions = prepared.x_grid[row_changes + 1]
    n_left = row_values[row_changes]
    signature = (
        "patterned",
        tuple(np.round(x_positions, 12)),
        tuple(
            (round(float(np.real(value)), 12), round(float(np.imag(value)), 12))
            for value in n_left
        ),
    )
    return [x_positions, n_left], signature


def _material_matches(material: Any, candidate: Any) -> bool:
    """Return whether two material identifiers should be treated as equal."""

    if material is candidate:
        return True
    try:
        is_match = material == candidate
    except Exception:
        return False
    return bool(is_match) if isinstance(is_match, bool) else False


@dataclass
class ProfileGrating(BaseGrating):
    """Grating profile defined by explicit sampled points.

    Attributes:
        period_lpermm: Grating period in lines per millimeter.
        coating_stack: Optional coating stack configuration.
        substrate_material: Substrate material identifier.
        layer_material: Layer material identifier.
        layer_thickness_nm: Layer thickness in nanometers.
        top_cap_material: Optional top cap material identifier.
        top_cap_thickness_nm: Top cap thickness in nanometers.
        z_resolution_nm: Vertical resolution for profile discretization.
        x_resolution_nm: Horizontal resolution for profile discretization.
        x_points_nm: Profile x coordinates for one period in nanometers.
        z_points_nm: Profile z coordinates for one period in nanometers.
    """

    x_points_nm: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    z_points_nm: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __post_init__(self) -> None:
        """Validate explicit profile arrays."""

        self.x_points_nm = np.asarray(self.x_points_nm, dtype=float)
        self.z_points_nm = np.asarray(self.z_points_nm, dtype=float)
        if self.x_points_nm.ndim != 1 or self.z_points_nm.ndim != 1:
            raise ValueError("x_points_nm and z_points_nm must be one-dimensional arrays.")
        if self.x_points_nm.size != self.z_points_nm.size:
            raise ValueError("x_points_nm and z_points_nm must have equal length.")
        if self.x_points_nm.size < 2:
            raise ValueError("x_points_nm and z_points_nm must contain at least two points.")
        if not np.all(np.isfinite(self.x_points_nm)) or not np.all(np.isfinite(self.z_points_nm)):
            raise ValueError("x_points_nm and z_points_nm must contain finite values.")
        if not np.all(np.diff(self.x_points_nm) >= 0.0):
            raise ValueError("x_points_nm must be monotonically non-decreasing.")

    def profile_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return explicit profile points for one period."""

        return self.x_points_nm, self.z_points_nm

    def profile_depth_nm(self) -> float:
        """Return the explicit profile peak-to-trough depth."""

        return float(np.max(self.z_points_nm) - np.min(self.z_points_nm))


@dataclass
class LaminarGrating(BaseGrating):
    """Laminar or trapezoidal grating profile.

    The substrate surface places the land top at ``z = 0`` and the groove
    floor at ``z = depth_nm``. ``width_to_period_ratio`` is the land-width
    fraction of one period. Sidewall angles are measured from the horizontal,
    so ``90`` degrees corresponds to vertical walls.

    Attributes:
        period_lpermm: Grating period in lines per millimeter.
        coating_stack: Optional coating stack configuration.
        substrate_material: Substrate material identifier.
        layer_material: Layer material identifier.
        layer_thickness_nm: Layer thickness in nanometers.
        top_cap_material: Optional top cap material identifier.
        top_cap_thickness_nm: Top cap thickness in nanometers.
        z_resolution_nm: Vertical resolution for profile discretization.
        x_resolution_nm: Horizontal resolution for profile discretization.
        width_to_period_ratio: Land width fraction of one period.
        depth_nm: Groove depth in nanometers.
        left_wall_angle_deg: Left sidewall angle from horizontal.
        right_wall_angle_deg: Right sidewall angle from horizontal.
    """

    width_to_period_ratio: float = 0.67
    depth_nm: float = 14.9
    left_wall_angle_deg: float = 15.0
    right_wall_angle_deg: float = 15.0

    def profile_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the laminar profile for one period."""

        if not 0.0 < self.width_to_period_ratio < 1.0:
            raise ValueError("width_to_period_ratio must be in (0, 1).")
        if self.left_wall_angle_deg <= 0.0 or self.left_wall_angle_deg > 90.0:
            raise ValueError("left_wall_angle_deg must be in (0, 90].")
        if self.right_wall_angle_deg <= 0.0 or self.right_wall_angle_deg > 90.0:
            raise ValueError("right_wall_angle_deg must be in (0, 90].")

        groove_floor_nm = (1.0 - self.width_to_period_ratio) * self.period_nm
        left_wall_foot_nm = self.depth_nm / np.tan(np.deg2rad(self.left_wall_angle_deg))
        right_wall_foot_nm = self.depth_nm / np.tan(np.deg2rad(self.right_wall_angle_deg))
        land_nm = self.period_nm - groove_floor_nm - left_wall_foot_nm - right_wall_foot_nm

        if land_nm < 0.0:
            raise ValueError(
                "LaminarGrating walls overlap. Reduce depth, increase wall angles, "
                "or reduce width_to_period_ratio."
            )

        left_land_nm = land_nm / 2.0
        x_land_r = left_land_nm
        x_floor_l = x_land_r + left_wall_foot_nm
        x_floor_r = x_floor_l + groove_floor_nm
        x_land_l = x_floor_r + right_wall_foot_nm

        positions = np.array(
            [0.0, x_land_r, x_floor_l, x_floor_r, x_land_l, self.period_nm],
            dtype=float,
        )
        heights = np.array([0.0, 0.0, self.depth_nm, self.depth_nm, 0.0, 0.0], dtype=float)
        return positions, heights

    def profile_depth_nm(self) -> float:
        """Return the laminar groove depth."""

        return float(self.depth_nm)


@dataclass
class BlazedGrating(BaseGrating):
    """Blazed grating profile.

    The legacy form uses a single blaze angle and an abrupt reset at the end
    of the period. When ``anti_blaze_angle_deg`` is provided, the profile is
    built as a two-facet triangle when ``anti_blaze_angle_deg`` is provided.

    Attributes:
        period_lpermm: Grating period in lines per millimeter.
        coating_stack: Optional coating stack configuration.
        substrate_material: Substrate material identifier.
        layer_material: Layer material identifier.
        layer_thickness_nm: Layer thickness in nanometers.
        top_cap_material: Optional top cap material identifier.
        top_cap_thickness_nm: Top cap thickness in nanometers.
        z_resolution_nm: Vertical resolution for profile discretization.
        x_resolution_nm: Horizontal resolution for profile discretization.
        blaze_angle_deg: Blaze angle in degrees.
        anti_blaze_angle_deg: Optional anti-blaze angle for two-facet triangle profile.
    """

    blaze_angle_deg: float = 0.9
    anti_blaze_angle_deg: float | None = None

    def __post_init__(self) -> None:
        """Set depth from the configured blaze geometry."""

        if self.blaze_angle_deg <= 0.0:
            raise ValueError("blaze_angle_deg must be > 0.")

        if self.anti_blaze_angle_deg is None:
            self.depth_nm = self.period_nm * np.tan(np.deg2rad(self.blaze_angle_deg))
            return

        if self.anti_blaze_angle_deg <= 0.0:
            raise ValueError("anti_blaze_angle_deg must be > 0 when provided.")

        blaze_tangent = np.tan(np.deg2rad(self.blaze_angle_deg))
        anti_blaze_tangent = np.tan(np.deg2rad(self.anti_blaze_angle_deg))
        self.depth_nm = (
            self.period_nm
            * blaze_tangent
            * anti_blaze_tangent
            / (blaze_tangent + anti_blaze_tangent)
        )

    def profile_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the blazed profile for one period."""

        if self.anti_blaze_angle_deg is not None:
            blaze_tangent = np.tan(np.deg2rad(self.blaze_angle_deg))
            anti_blaze_tangent = np.tan(np.deg2rad(self.anti_blaze_angle_deg))
            apex_x = self.period_nm * anti_blaze_tangent / (blaze_tangent + anti_blaze_tangent)
            positions = np.array([0.0, apex_x, self.period_nm], dtype=float)
            heights = np.array([0.0, self.depth_nm, 0.0], dtype=float)
            return positions, heights

        # Use a one-step drop at period end to emulate the sawtooth reset.
        edge_step_nm = max(self.x_resolution_nm, self.period_nm * 1e-6)
        x_peak = self.period_nm - edge_step_nm

        positions = np.array([0.0, x_peak, self.period_nm], dtype=float)
        heights = np.array([0.0, self.depth_nm, 0.0], dtype=float)
        return positions, heights

    def profile_depth_nm(self) -> float:
        """Return the blazed groove depth derived from blaze geometry."""

        return float(self.depth_nm)
