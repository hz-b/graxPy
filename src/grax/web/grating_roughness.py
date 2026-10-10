"""Saved grating roughness settings and deterministic preview realization."""
from copy import copy
from dataclasses import asdict, replace
import math

from grax.roughness import RoughnessSpec


def optional_nonnegative(value, label):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite and at least 0 nm.") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and at least 0 nm.")
    return number


def roughness_from_form(form):
    kind = form.get("roughness_kind", "none")
    if kind == "none":
        return None
    options = dict(kind=kind, sigma_nm=0.0)
    if kind == "random-interface":
        for field, default, minimum in (("seed", 0, 0), ("num_supercells", 1, 1), ("num_realizations", 1, 1)):
            try:
                value = int(str(form.get("roughness_" + field, default)))
            except (TypeError, ValueError):
                raise ValueError(f"Roughness {field.replace('_', ' ')} must be a whole number of at least {minimum}.") from None
            if value < minimum:
                raise ValueError(f"Roughness {field.replace('_', ' ')} must be a whole number of at least {minimum}.")
            options[field] = value
    return asdict(RoughnessSpec(**options))


def realization_grating(grating, seed=None):
    result = copy(grating)
    if seed is None:
        seed = grating.roughness.realization_seeds()[0] if grating.roughness.num_realizations > 1 else grating.roughness.seed
    result.roughness = replace(grating.roughness, seed=seed, num_realizations=1)
    return result


def validate_roughness_geometry(grating):
    if not grating._random_interface_active():
        return
    seeds = grating.roughness.realization_seeds() if grating.roughness.num_realizations > 1 else [grating.roughness.seed]
    for seed in seeds:
        sample = realization_grating(grating, seed)
        count = sample._roughness_num_supercells()
        x = sample._build_x_grid(num_periods=count)
        surface = sample._surface_profile_on_grid(x, num_periods=count)
        sample._rough_geometry(x, surface, sample.resolved_stack())
