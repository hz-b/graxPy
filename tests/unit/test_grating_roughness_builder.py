"""Saved roughness, shared geometry and builder previews."""
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from grax import RoughnessSpec
from grax.gratings import LaminarGrating, SinusoidalGrating, BlazedGrating
from grax.stacks import BareStack, SingleLayerStack, MultilayerStack, CustomStack, LayerSpec
from grax.simulation import run_simulation
from grax.solvers import NeviereOptions
from grax.web.app import create_app, _default_form_values, _attach_roughness
from grax.web.grating_roughness import realization_grating, validate_roughness_geometry
from grax.web.persistence import grating_to_spec, build_grating_from_spec, GratingStore
from tests.optical_constants import load_optical_constants_table

TABLES = Path(__file__).resolve().parents[2] / 'examples' / 'optical_constants'
SI = load_optical_constants_table(TABLES / 'n_Si_cxro.txt', 'Si')
PT = load_optical_constants_table(TABLES / 'n_Pt_cxro.txt', 'Pt')
CR = load_optical_constants_table(TABLES / 'n_Cr_cxro.txt', 'Cr')


def make_grating(kind, stack_kind='single', sigma=0.2):
    common = dict(substrate_material=SI, substrate_roughness_sigma_nm=sigma,
                  substrate_correlation_length_nm=15)
    if stack_kind == 'single':
        stack = SingleLayerStack(**common, layer_material=PT, layer_thickness_nm=8,
                                 layer_roughness_sigma_nm=sigma, layer_correlation_length_nm=60)
    elif stack_kind == 'multi':
        stack = MultilayerStack(**common, material_a=PT, material_b=CR, top_material=CR,
                                d_period_nm=12, n_bilayers=2, gamma=0.5,
                                material_a_roughness_sigma_nm=sigma, material_b_roughness_sigma_nm=sigma,
                                material_a_correlation_length_nm=20, material_b_correlation_length_nm=80)
    elif stack_kind == 'custom':
        stack = CustomStack(**common, layers_bottom_up=[LayerSpec(CR, 6, sigma, 20), LayerSpec(PT, 8, sigma, 80)])
    else:
        stack = BareStack(**common)
    kwargs = dict(coating_stack=stack, period_lpermm=2500, x_resolution_nm=5,
                  z_resolution_nm=1, roughness=RoughnessSpec('random-interface', 0, seed=14, num_realizations=1))
    if kind == 'laminar':
        return LaminarGrating(**kwargs, depth_nm=5, left_wall_angle_deg=45, right_wall_angle_deg=45)
    if kind == 'blazed':
        return BlazedGrating(**kwargs, blaze_angle_deg=1, anti_blaze_angle_deg=20)
    return SinusoidalGrating(**kwargs, depth_nm=5)


@pytest.mark.parametrize('kind', ['laminar', 'blazed', 'sinusoidal'])
@pytest.mark.parametrize('stack_kind', ['single', 'multi', 'custom', 'bare'])
def test_random_geometry_dense_low_memory_and_zero_limit(kind, stack_kind):
    grating = make_grating(kind, stack_kind)
    dense, _ = grating.build_textures(300, _memory_mode='legacy_dense')
    low, (thicknesses, indices) = grating.build_textures(300, _memory_mode='low_memory')
    expanded = []
    for thickness, index in zip(thicknesses[1:-1], indices[1:-1]):
        expanded.extend([low[int(index)]] * round(thickness / grating.z_resolution_nm))
    signature = grating._texture_descriptor_signature
    assert [signature(row) for row in expanded] == [signature(row) for row in dense[1:-1]]
    zero = make_grating(kind, stack_kind, sigma=0)
    x, z, rough, labels = zero._material_plot_data(num_periods=1)
    smooth = copy(zero)
    smooth.roughness = None
    codes = {label: i for i, label in enumerate(labels)}
    clean = smooth._build_material_code_grid(x_grid=x, z_grid=z,
        surface=smooth._surface_profile_on_grid(x, num_periods=1), coating_stack=smooth.resolved_stack(),
        material_codes=codes, include_incident_medium=False)
    np.testing.assert_array_equal(rough, clean)


def test_xi_seed_and_vertical_bounds():
    grating = make_grating('sinusoidal', 'bare', sigma=2)
    grating.roughness = replace(grating.roughness, num_supercells=2, num_realizations=2)
    sample = realization_grating(grating)
    assert sample.roughness.seed == grating.roughness.realization_seeds()[0]
    x = sample._build_x_grid(num_periods=2)
    surface = sample._surface_profile_on_grid(x, num_periods=2)
    substrate, layers, top = sample._rough_geometry(x, surface, sample.resolved_stack())
    assert layers == []
    np.testing.assert_array_equal(substrate, top)
    z = sample._build_solver_z_grid(sample.resolved_stack())
    assert z[-1] <= substrate.min() and z[0] >= top.max()
    assert not np.allclose(substrate, surface)
    same = realization_grating(grating)
    np.testing.assert_array_equal(same._rough_geometry(x, surface, same.resolved_stack())[0], substrate)
    changed = copy(sample)
    changed.coating_stack = replace(sample.coating_stack, substrate_correlation_length_nm=0)
    noisy = changed._rough_geometry(x, surface, changed.resolved_stack())[0]
    assert np.mean(np.diff(noisy)**2) > np.mean(np.diff(substrate)**2)


def test_crossings_are_rejected():
    grating = make_grating('laminar', sigma=20)
    with pytest.raises(ValueError, match='interfaces .* cross.*seed'):
        validate_roughness_geometry(grating)


@pytest.mark.parametrize('kind', ['laminar', 'blazed', 'sinusoidal'])
def test_both_solvers_and_continuous_roughness(kind):
    grating = make_grating(kind)
    settings = dict(grating=grating, energy_ev=300, grazing_angle_deg=3, fourier_orders=1,
                    polarization='s', validate_physical_results=False)
    rcwa = run_simulation(**settings, solver='rcwa')
    neviere = run_simulation(**settings, solver='neviere')
    continuous = run_simulation(**settings, solver='neviere', solver_options=NeviereOptions(z_sampling='continuous'))
    assert np.all(np.isfinite(continuous.efficiency_all))
    np.testing.assert_allclose(rcwa.efficiency_all, neviere.efficiency_all, atol=0.03)


@pytest.mark.parametrize('kind', ['laminar', 'blazed', 'sinusoidal'])
def test_builder_save_preview_reload_and_run_settings(tmp_path, kind):
    client = create_app(data_dir=tmp_path).test_client()
    form = _default_form_values()
    form.update(grating_type=kind, roughness_kind='random-interface', layer_roughness_sigma_nm='0.2',
                layer_correlation_length_nm='60', roughness_num_realizations='2')
    preview = client.post('/_preview/grating', data=form).json
    assert preview['ok'], preview
    assert preview['roughness_preview_url']
    assert client.get(preview['roughness_preview_url']).status_code == 200
    response = client.post('/gratings', data=form)
    assert response.status_code == 302
    store = GratingStore(tmp_path / 'saved_gratings')
    saved = store.list()[0]
    assert saved['schema_version'] == 4
    grating = build_grating_from_spec(saved)
    assert grating.resolved_stack().layer_correlation_length_nm == 60
    original = grating.roughness
    _attach_roughness(grating, {})
    assert grating.roughness is original
    _attach_roughness(grating, {'roughness_kind': 'random-interface'})
    assert grating.roughness is original
    _attach_roughness(grating, {'roughness_kind': 'debye-waller'})
    assert grating.roughness.kind == 'debye-waller'
    assert grating.roughness.num_realizations == 1
    assert grating.resolved_stack().layer_correlation_length_nm == 60
    _attach_roughness(grating, {'roughness_kind': 'none'})
    assert grating.roughness is None
    assert build_grating_from_spec(store.list()[0]).roughness == original
    detail = client.get(response.location)
    assert b'name="roughness_kind"' in detail.data
    assert b'value="random-interface" selected' in detail.data
    assert b'random-interface' in detail.data
    edit = client.get(response.location + '/edit')
    assert b'value="random-interface" selected' in edit.data
    form['roughness_kind'] = 'debye-waller'
    assert client.post('/_preview/grating', data=form).json['roughness_preview_url'] is None
    form['layer_correlation_length_nm'] = '-1'
    invalid = client.post(response.location, data=form)
    assert invalid.status_code == 422
    assert b'role="alert"' in invalid.data
    assert b'at least 0 nm' in invalid.data


def test_grating_preview_reports_laminar_geometry_fields(tmp_path):
    client = create_app(data_dir=tmp_path).test_client()
    form = _default_form_values()
    form.update(
        grating_type="laminar",
        period_lpermm="3600",
        width_to_period_ratio="0.67",
        depth_nm="12",
        left_wall_angle_deg="5",
        right_wall_angle_deg="5",
    )

    payload = client.post("/_preview/grating", data=form).json

    assert payload["ok"] is False
    assert payload["error_category"] == "geometry"
    assert payload["error_fields"] == [
        "depth_nm",
        "left_wall_angle_deg",
        "right_wall_angle_deg",
        "width_to_period_ratio",
    ]
    assert "wall footprint" in payload["error"]
    assert "depth=12.000 nm" in payload["error"]


def test_legacy_no_model_and_python_round_trip():
    grating = make_grating('sinusoidal', 'custom')
    # Use catalog symbols for JSON persistence.
    grating.coating_stack = CustomStack('Si', layers_bottom_up=[LayerSpec('Cr', 5, 0.3, 25)])
    spec = grating_to_spec(grating, name='round trip')
    restored = build_grating_from_spec(spec)
    assert restored.roughness == grating.roughness
    assert restored.coating_stack.layers_bottom_up[0].correlation_length_nm == 25
    del spec['roughness']
    spec['schema_version'] = 3
    assert build_grating_from_spec(spec).roughness is None


def test_custom_blocks_keep_a_b_correlation_and_order():
    from tests.unit.test_web_custom_blocks import mixed_form
    from grax.web.app import _spec_from_form, _grating_custom_layer_rows
    form = mixed_form()
    form.setlist('roughness_kind', ['random-interface'])
    form.setlist('cl_correlation_length_nm', ['10', '20', '30', '40'])
    form.setlist('cl_correlation_b_length_nm', ['', '50', '', ''])
    spec = _spec_from_form(form)
    grating = build_grating_from_spec(spec)
    assert [layer.correlation_length_nm for layer in grating.coating_stack.layers_bottom_up] == [40, 30, 50, 20, 50, 20, 10]
    assert _grating_custom_layer_rows(spec)[1]['correlation_b_length_nm'] == '50'


@pytest.mark.parametrize('field,value', [('roughness_seed', '-1'), ('roughness_num_supercells', '0'),
    ('roughness_num_realizations', '1.5'), ('substrate_roughness_sigma_nm', 'nan'),
    ('layer_correlation_length_nm', 'inf')])
def test_invalid_builder_parameters_keep_form(tmp_path, field, value):
    form = _default_form_values()
    form.update(grating_type='sinusoidal', roughness_kind='random-interface')
    form[field] = value
    response = create_app(data_dir=tmp_path).test_client().post('/gratings', data=form)
    assert response.status_code == 422
    assert b'role="alert"' in response.data
    assert b'Untitled grating' in response.data


def test_preview_is_stable_and_debug_uses_simulation_span(tmp_path):
    client = create_app(data_dir=tmp_path).test_client()
    form = _default_form_values()
    form.update(roughness_kind='random-interface', layer_roughness_sigma_nm='0.5', roughness_num_realizations='2')
    first = client.post('/_preview/grating', data=form).json
    form['name'] = 'Name change only'
    second = client.post('/_preview/grating', data=form).json
    assert client.get(first['roughness_preview_url']).data == client.get(second['roughness_preview_url']).data
    grating = make_grating('sinusoidal', 'bare', sigma=1)
    grating.roughness = replace(grating.roughness, num_supercells=2)
    grating.save_structure_debug_data(300, tmp_path / 'debug')
    x = np.loadtxt(tmp_path / 'debug' / 'x.csv', delimiter=',')
    assert x[-1] == 2 * grating.period_nm
    z = np.loadtxt(tmp_path / 'debug' / 'z.csv', delimiter=',')
    assert z.min() < 0


@pytest.mark.parametrize('periods', [1, 2])
def test_detail_separates_nominal_and_simulated_plot_spans(tmp_path, monkeypatch, periods):
    from grax.gratings import BaseGrating
    from bs4 import BeautifulSoup
    spans = []
    original = BaseGrating._material_plot_data

    def record(grating, *args, **kwargs):
        result = original(grating, *args, **kwargs)
        spans.append((grating._random_interface_active(), result[0][-1] - result[0][0]))
        return result

    monkeypatch.setattr(BaseGrating, '_material_plot_data', record)
    client = create_app(data_dir=tmp_path).test_client()
    form = _default_form_values()
    form.update(grating_type='sinusoidal', stack_type='bare', roughness_kind='random-interface',
                roughness_num_supercells=str(periods), roughness_num_realizations='1')
    saved = client.post('/gratings', data=form)
    page = client.get(saved.location)
    assert page.status_code == 200
    assert spans == [(False, 7500), (True, periods * 2500)]
    images = BeautifulSoup(page.data, 'html.parser').select('img.preview')
    assert len(images) == 2
    assert all(client.get(img['src']).status_code == 200 for img in images)
    assert b'three aligned periods' in page.data
    form['roughness_kind'] = 'debye-waller'
    client.post(saved.location, data=form)
    smooth_page = client.get(saved.location)
    assert len(BeautifulSoup(smooth_page.data, 'html.parser').select('img.preview')) == 1
