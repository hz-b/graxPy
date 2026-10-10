"""Mixed single-layer and multilayer block editing."""
from pathlib import Path

import pytest
from werkzeug.datastructures import MultiDict

from grax.web.app import _default_form_values, create_app
from grax.web.persistence import GratingStore, build_grating_from_spec
from grax.web.plane_mirror import custom_stack_from_form, split_plane_mirror_form


def mixed_form():
    form = MultiDict(_default_form_values())
    for key, value in dict(name="Mixed blocks", grating_type="sinusoidal", stack_type="custom").items():
        form.setlist(key, [value])
    columns = {
        "kind": ["single", "multilayer", "single", "single"],
        "material": ["C", "Pt", "Si", "Au"],
        "density_g_cm3": [""] * 4,
        "thickness_nm": ["1", "2", "4", "5"],
        "roughness_sigma_nm": ["0.1", "0.2", "0.4", "0.5"],
        "material_b": ["", "Cr", "", ""],
        "density_b_g_cm3": [""] * 4,
        "thickness_b_nm": ["", "3", "", ""],
        "roughness_b_sigma_nm": ["", "0.3", "", ""],
        "repeats": ["1", "2", "1", "1"],
    }
    for key, values in columns.items():
        form.setlist("cl_" + key, values)
    return form


def test_mixed_blocks_expand_and_round_trip(tmp_path: Path):
    form = mixed_form()
    stack = custom_stack_from_form(form)
    assert [(layer.material.name, layer.thickness_nm) for layer in stack.layers_bottom_up] == [
        ("Au", 5), ("Si", 4), ("Cr", 3), ("Pt", 2), ("Cr", 3), ("Pt", 2), ("C", 1)
    ]
    assert [layer.roughness_sigma_nm for layer in stack.layers_bottom_up] == [0.5, 0.4, 0.3, 0.2, 0.3, 0.2, 0.1]
    client = create_app(data_dir=tmp_path).test_client()
    assert client.post("/_preview/grating", data=form).json["ok"]
    saved_response = client.post("/gratings", data=form)
    assert saved_response.status_code == 302
    store = GratingStore(tmp_path / "saved_gratings")
    saved = store.list()[0]
    assert len(saved["stack"]["blocks_top_down"]) == 4
    assert len(build_grating_from_spec(saved).resolved_stack().layers_bottom_up) == 7
    edit = client.get(saved_response.location + "/edit")
    assert b'Add multilayer' in edit.data
    assert b'value="multilayer" selected' in edit.data
    assert b'name="cl_repeats"' in edit.data
    form.setlist("cl_repeats", ["1", "3", "1", "1"])
    assert client.post(saved_response.location, data=form).status_code == 302
    assert len(store.load(saved["id"])["stack"]["layers_bottom_up"]) == 9
    mirror_form, _ = split_plane_mirror_form(form)
    assert mirror_form["cl_kind"] == ["single", "multilayer", "single", "single"]


@pytest.mark.parametrize("repeats", ["0", "251", "2.5", "nan"])
def test_invalid_repeat_count(repeats):
    form = mixed_form()
    form.setlist("cl_repeats", ["1", repeats, "1", "1"])
    with pytest.raises(ValueError, match="whole number from 1 to 250"):
        custom_stack_from_form(form)
