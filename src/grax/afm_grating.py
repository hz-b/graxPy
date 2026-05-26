"""AFM-derived profile grating adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .afm_preprocessing import AFMPreprocessing
from .gratings import ProfileGrating


@dataclass
class AFMGrating(ProfileGrating):
    """Profile grating built from processed AFM data."""

    @classmethod
    def from_preprocessing(
        cls,
        afm: AFMPreprocessing,
        *,
        period_lpermm: int | None = None,
        **grating_kwargs: Any,
    ) -> "AFMGrating":
        """Build an AFM grating from a preprocessed AFM period.

        Args:
            afm: Preprocessing pipeline with a completed period profile.
            period_lpermm: Optional grating period in lines/mm. When omitted,
                inferred from the processed profile span.
            **grating_kwargs: Remaining :class:`ProfileGrating` and
                :class:`~grax.gratings.BaseGrating` keyword arguments.

        Returns:
            An :class:`AFMGrating` instance.
        """

        try:
            x_points_nm, z_points_nm = afm.get_profile()
        except RuntimeError as error:
            raise RuntimeError(
                "AFM profile is not ready. Call rescale_period() first, or pass "
                "period_lpermm explicitly after preparing the period."
            ) from error

        resolved_period_lpermm = period_lpermm
        if resolved_period_lpermm is None:
            span_nm = float(np.max(x_points_nm) - np.min(x_points_nm))
            if not np.isfinite(span_nm) or span_nm <= 0.0:
                raise ValueError(
                    "Cannot infer period_lpermm from AFM profile span. Provide "
                    "period_lpermm explicitly."
                )
            resolved_period_lpermm = int(round(1e6 / span_nm))
            if resolved_period_lpermm <= 0:
                raise ValueError("Inferred period_lpermm is invalid. Provide it explicitly.")

        return cls(
            period_lpermm=resolved_period_lpermm,
            x_points_nm=np.asarray(x_points_nm, dtype=float),
            z_points_nm=np.asarray(z_points_nm, dtype=float),
            **grating_kwargs,
        )
