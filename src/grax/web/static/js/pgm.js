const PGM_SIDE_VIEW_COLORS = {
  mirror: "#2ca02c", grating: "#d62728", beam: "#f59e0b", hit: "#1f77b4",
  gratingAxis: "#000", mirrorAxis: "#c026d3",
};
const PGM_CFF_LABEL = "c<sub>ff</sub> = cos β / cos α";
const PGM_MAPS = {
  geometry: {
    title: "Geometric throughput", colorbar: "Geometrically transmitted beam height [%]",
    fixedMax: 100, critical: true, legend: true,
  },
  mirror: { title: "Mirror reflectivity", colorbar: "R<sub>mirror</sub> [%]" },
  grating: { title: "Grating efficiency", colorbar: "η<sub>grating</sub> [%]" },
  optical: { title: "Mirror × grating", colorbar: "R<sub>mirror</sub> η<sub>grating</sub> [%]" },
  total: {
    title: "Mirror × grating × geometry", colorbar: "Total throughput [%]",
    critical: true, angleCut: true,
  },
};
const PGM_LINE_SAMPLES = { solid: "——", dash: "– –", dot: "···", dashdot: "–·–" };

function pgmMaxPercent(grid) {
  let top = 0;
  for (const row of grid) {
    for (const value of row) {
      if (value !== null && value > top) top = value;
    }
  }
  return Math.max(Math.min(top, 100), 1e-6);
}

// The colorbar ticks of the SX700 dashboard: a compact step, plus the maximum itself.
function pgmColorbarTicks(top, fixed) {
  if (fixed) return [0, 20, 40, 60, 80, 100];
  let step = 20;
  if (top <= 10) step = 2;
  else if (top <= 25) step = 5;
  else if (top <= 50) step = 10;
  const ticks = [];
  for (let value = 0; value < top; value += step) ticks.push(value);
  ticks.push(top);
  return ticks;
}

function pgmEnergyTicks(minimum, maximum) {
  const rough = (maximum - minimum) / 5;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 5, 10].map((factor) => factor * power).find((value) => value >= rough);
  const ticks = [];
  for (let value = Math.ceil(minimum / step) * step; value <= maximum + 1e-9; value += step) {
    ticks.push(value);
  }
  return { step, ticks };
}

function pgmNearestIndex(values, target) {
  let best = 0;
  values.forEach((value, index) => {
    if (Math.abs(value - target) < Math.abs(values[best] - target)) best = index;
  });
  return best;
}

function pgmOverlayTraces(config, overlays) {
  if (!overlays) return [];
  const skipped = { showscale: false, hoverinfo: "skip", zmin: 0, zmax: 1 };
  const traces = [];
  if (config.angleCut && overlays.angle_cut) {
    traces.push({
      type: "heatmap", x: overlays.energies, y: overlays.cff, z: overlays.angle_cut,
      colorscale: [[0, "#000"], [1, "#000"]], ...skipped,
    });
  }
  if (config.critical && overlays.hatch) {
    traces.push({
      type: "heatmap", x: overlays.hatch.energies, y: overlays.hatch.cff, z: overlays.hatch.z,
      colorscale: [[0, "rgba(0,0,0,0.75)"], [1, "rgba(0,0,0,0.75)"]], ...skipped,
    });
  }
  const contour = (z, level, color, width, dash) => ({
    type: "contour", x: overlays.energies, y: overlays.cff, z,
    autocontour: false, showscale: false, hoverinfo: "skip",
    contours: { start: level, end: level, size: 1, coloring: "none" },
    line: { color, width, dash },
  });
  if (config.critical && overlays.distance_limit.drawn) {
    traces.push(contour(overlays.distance, overlays.distance_limit.level, "black", 1.7, "dash"));
  }
  overlays.limits.forEach((limit) => {
    traces.push(contour(overlays.fields[limit.field], limit.level, "red", 1.4, limit.dash));
  });
  return traces;
}

function pgmLegendText(overlays) {
  const lines = ["Mechanical limits"];
  overlays.limits.forEach((limit) => {
    lines.push(`<span style="color:red">${PGM_LINE_SAMPLES[limit.dash]}</span> ${limit.label}`);
  });
  lines.push(`${PGM_LINE_SAMPLES.dash} ${overlays.distance_limit.label}`);
  return lines.join("<br>");
}

function pgmGridShapes(energies, cff, energyTicks, cffTicks) {
  const white = (alpha) => `rgba(255,255,255,${alpha})`;
  const shapes = [];
  const add = (shape, alpha, width, dash) => {
    shapes.push({ type: "line", layer: "above", line: { color: white(alpha), width, dash }, ...shape });
  };
  energyTicks.forEach((value) => add({ xref: "x", yref: "paper", x0: value, x1: value, y0: 0, y1: 1 }, 0.62, 0.85));
  cffTicks.forEach((value) => add({ xref: "paper", yref: "y", x0: 0, x1: 1, y0: value, y1: value }, 0.62, 0.85));
  const first = energies[0];
  const span = energies[energies.length - 1] - first;
  [0.2, 0.4, 0.6, 0.8].forEach((fraction) => {
    const value = first + fraction * span;
    add({ xref: "x", yref: "paper", x0: value, x1: value, y0: 0, y1: 1 }, 0.4, 0.65, "dot");
  });
  return shapes;
}

function pgmMarkerAnnotation(point) {
  return {
    name: "pgm-selected", xref: "x", yref: "y", x: point.energy, y: Math.log10(point.cff),
    text: "&nbsp;", showarrow: false, font: { size: 5 },
    bgcolor: "#fff", bordercolor: "#000", borderwidth: 1, borderpad: 1,
  };
}

function initPgmDashboard(root) {
  const message = root.querySelector("[data-pgm-dashboard-message]");
  const sideView = root.querySelector("[data-pgm-side-view]");
  const charts = new Map(
    [...root.querySelectorAll("[data-pgm-chart]")].map((element) => [element.dataset.pgmChart, element]),
  );
  const active = root.hasAttribute("data-pgm-active");
  const plotConfig = { responsive: true, displaylogo: false };
  let data = null;
  let token = "";
  let selected = null;
  let selectedUnsolved = false;
  const shown = {};
  const clickBound = new Set();

  function renderMap(name) {
    const config = PGM_MAPS[name];
    const element = charts.get(name);
    const current = shown[name];
    const real = current.z;
    const top = config.fixedMax || pgmMaxPercent(real);
    const ticks = pgmColorbarTicks(top, Boolean(config.fixedMax));
    const energies = data.energies;
    const energyTicks = pgmEnergyTicks(energies[0], energies[energies.length - 1]);
    const cffMin = data.cff[0];
    const cffMax = data.cff[data.cff.length - 1];
    const cffTicks = [];
    for (let value = Math.ceil(cffMin); value <= cffMax; value += 1) cffTicks.push(value);
    const annotations = [];
    if (!config.fixedMax) {
      annotations.push({
        xref: "paper", yref: "paper", x: 0.985, y: 0.985, xanchor: "right", yanchor: "top",
        text: `max ${top.toFixed(1)} %`, showarrow: false, font: { size: 11 },
        bgcolor: "rgba(255,255,255,0.85)", bordercolor: "#b3b3b3", borderpad: 3,
      });
    }
    if (config.legend && data.overlays) {
      annotations.push({
        xref: "paper", yref: "paper", x: 0.01, y: 0.02, xanchor: "left", yanchor: "bottom", align: "left",
        text: pgmLegendText(data.overlays), showarrow: false, font: { size: 10 },
        bgcolor: "rgba(255,255,255,0.9)", bordercolor: "#b3b3b3", borderpad: 4,
      });
    }
    if (selected) annotations.push(pgmMarkerAnnotation(selected));
    return Plotly.react(element, [{
      type: "heatmap", x: current.x, y: current.y,
      z: real.map((row) => row.map((value) => (value === null ? null : Math.sqrt(Math.max(value, 0))))),
      customdata: real, zmin: 0, zmax: Math.sqrt(top), colorscale: data.colorscale,
      hoverongaps: false,
      hovertemplate: "E %{x:.1f} eV<br>c<sub>ff</sub> %{y:.3f}<br>%{customdata:.2f} %<extra></extra>",
      colorbar: {
        title: { text: config.colorbar, side: "right", font: { size: 11 } },
        tickvals: ticks.map(Math.sqrt),
        ticktext: ticks.map((value) => (config.fixedMax ? String(value) : value.toFixed(2))),
        tickfont: { size: 10 }, thickness: 14,
      },
    }, ...pgmOverlayTraces(config, data.overlays)], {
      title: { text: config.title, font: { size: 13 }, y: 0.98 },
      uirevision: "pgm",
      showlegend: false,
      margin: { t: 30, r: 10, b: 45, l: 55 },
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fff",
      hovermode: "closest",
      xaxis: {
        title: { text: "Photon energy E [eV]", font: { size: 12 } },
        range: [energies[0], energies[energies.length - 1]],
        tick0: 0, dtick: energyTicks.step, showgrid: false, zeroline: false,
        showline: true, linecolor: "#000", mirror: true, ticks: "outside",
      },
      yaxis: {
        title: { text: PGM_CFF_LABEL, font: { size: 12 } }, type: "log",
        range: [Math.log10(cffMin), Math.log10(cffMax)],
        tickvals: cffTicks, ticktext: cffTicks.map(String), showgrid: false, zeroline: false,
        showline: true, linecolor: "#000", mirror: true, ticks: "outside",
      },
      shapes: pgmGridShapes(energies, data.cff, energyTicks.ticks, cffTicks),
      annotations,
    }, plotConfig).then(() => {
      if (clickBound.has(name)) return;
      clickBound.add(name);
      element.on("plotly_click", (event) => {
        const point = event.points?.[0];
        if (!point) return;
        selectPoint(pgmNearestIndex(data.energies, point.x), pgmNearestIndex(data.cff, point.y));
      });
    });
  }

  function renderSideView(point) {
    const percent = (value) => (value === null ? "…" : `${value.toFixed(1)} %`);
    const g = point.geometry;
    const view = point.side_view;
    const e = point.efficiencies;
    const info = g ? [
      `Geometry: ${g.throughput_percent.toFixed(1)} %`,
      `Mirror R = ${percent(e.mirror)}`,
      `Grating η = ${percent(e.grating)}`,
      `R·η = ${percent(e.optical)}`,
      `Total = ${percent(e.total)}`,
      `E = ${point.energy_eV.toFixed(0)} eV`,
      `c<sub>ff</sub> = ${point.cff.toFixed(3)}`,
      `α = ${g.alpha_deg.toFixed(2)}°`,
      `β = ${g.beta_deg.toFixed(2)}°`,
      `Θ = ${g.theta_deg.toFixed(2)}°`,
    ] : [`E = ${point.energy_eV.toFixed(0)} eV`, `c<sub>ff</sub> = ${point.cff.toFixed(3)}`, "No optical solution"];
    const traces = [];
    if (view) {
      const colors = PGM_SIDE_VIEW_COLORS;
      const line = (segment, color, width, name) => ({
        type: "scatter", mode: "lines", x: segment.x, y: segment.y, name: name || "",
        showlegend: Boolean(name), hoverinfo: "skip",
        line: { color, width },
      });
      traces.push(line(view.mirror, colors.mirror, 4, "Premirror"), line(view.grating, colors.grating, 4, "Grating"));
      view.central_ray.forEach((segment, index) => {
        traces.push(line(segment, colors.beam, 1.8, index === 0 ? "Central ray" : ""));
      });
      view.edge_rays.forEach((segment) => traces.push(line(segment, colors.beam, 0.7)));
      const cross = (at, color, name) => ({
        type: "scatter", mode: "markers", x: [at.x], y: [at.y], name, hoverinfo: "skip",
        marker: { symbol: "cross-thin", size: 14, line: { color, width: 2.5 } },
      });
      // The crosses come last so a hit point on an axis cannot hide them.
      traces.push({
        type: "scatter", mode: "markers", x: view.hit_points.x, y: view.hit_points.y,
        name: "Hit points", hoverinfo: "skip", marker: { color: colors.hit, size: 7 },
      }, cross(view.grating_axis, colors.gratingAxis, "Grating axis"), cross(view.mirror_axis, colors.mirrorAxis, "Mirror axis"));
    }
    return Plotly.react(sideView, traces, {
      title: { text: "Side view", font: { size: 13 }, y: 0.98 },
      margin: { t: 30, r: 10, b: 45, l: 50 },
      paper_bgcolor: "#fff", plot_bgcolor: "#fff",
      xaxis: {
        title: { text: "x [mm]", font: { size: 12 } }, zeroline: false, gridcolor: "rgba(0,0,0,0.12)",
        showline: true, linecolor: "#000", mirror: true, ticks: "outside",
      },
      yaxis: {
        title: { text: "y [mm]", font: { size: 12 } }, range: view ? view.y_range : undefined,
        zeroline: false, gridcolor: "rgba(0,0,0,0.12)",
        showline: true, linecolor: "#000", mirror: true, ticks: "outside",
      },
      legend: {
        x: 0.99, y: 0.01, xanchor: "right", yanchor: "bottom", font: { size: 10 },
        bgcolor: "rgba(255,255,255,0.85)", bordercolor: "#b3b3b3", borderwidth: 1,
      },
      annotations: [{
        xref: "paper", yref: "paper", x: 0.02, y: 0.98, xanchor: "left", yanchor: "top", align: "left",
        text: info.join("<br>"), showarrow: false, font: { size: 11 },
        bgcolor: "rgba(255,255,255,0.88)", bordercolor: "#b3b3b3", borderpad: 4,
      }],
    }, plotConfig);
  }

  async function selectPoint(energyIndex, cffIndex) {
    const url = new URL(root.dataset.pointUrl, window.location.href);
    url.searchParams.set("energy", energyIndex);
    url.searchParams.set("cff", cffIndex);
    const response = await fetch(url);
    if (!response.ok) return;
    const point = await response.json();
    await renderSideView(point);
    selected = { energy: point.energy_eV, cff: point.cff, energyIndex, cffIndex };
    selectedUnsolved = point.efficiencies.grating === null;
    charts.forEach((element) => {
      const annotations = element.layout?.annotations;
      if (!annotations) return;
      const others = annotations.filter((annotation) => annotation.name !== "pgm-selected");
      Plotly.relayout(element, { annotations: [...others, pgmMarkerAnnotation(selected)] });
    });
  }

  async function pollLive() {
    try {
      const url = new URL(root.dataset.liveUrl, window.location.href);
      url.searchParams.set("v", token);
      const response = await fetch(url);
      if (response.ok) {
        const live = await response.json();
        if (!live.unchanged) {
          token = live.token;
          for (const [name, z] of Object.entries(live.maps)) {
            shown[name] = { x: live.energies, y: live.cff, z };
            await renderMap(name);
          }
          // The side view's efficiencies appear once the grating reaches its point.
          if (selected && selectedUnsolved) await selectPoint(selected.energyIndex, selected.cffIndex);
        }
      }
    } catch { /* Retry while the local server is temporarily unavailable. */ }
    window.setTimeout(pollLive, 2000);
  }

  async function loadDashboard() {
    let response = null;
    try {
      response = await fetch(root.dataset.mapsUrl);
    } catch { /* Retried below while the run is active. */ }
    if (!response || !response.ok) {
      if (active) {
        window.setTimeout(loadDashboard, 1000);
      } else {
        message.textContent = "This run has no maps.";
      }
      return;
    }
    data = await response.json();
    token = data.token;
    message.textContent = "";
    for (const name of Object.keys(PGM_MAPS)) {
      shown[name] = { x: data.energies, y: data.cff, z: data.maps[name] };
      await renderMap(name);
    }
    await selectPoint(data.start.energy_index, data.start.cff_index);
    if (active) window.setTimeout(pollLive, 2000);
  }

  loadDashboard();
}

const pgmDashboardRoot = document.querySelector("[data-pgm-dashboard]");
if (pgmDashboardRoot && window.Plotly) initPgmDashboard(pgmDashboardRoot);

function bindPgmEditorFields(fields) {
  fields.querySelectorAll("[data-grating-type]").forEach((select) => {
    syncGratingSections(select);
    select.addEventListener("change", () => {
      syncGratingSections(select);
      defaultStackForGrating(select);
    });
  });
  fields.querySelectorAll("[data-stack-type]").forEach((select) => {
    syncStackSections(select);
    select.addEventListener("change", () => syncStackSections(select));
  });
  fields.querySelectorAll("[data-material-select]").forEach((field) => {
    field.addEventListener("input", () => syncMaterialDensity(field));
    field.addEventListener("change", () => syncMaterialDensity(field));
  });
  fields.querySelectorAll("[data-material-density]").forEach((densityInput) => {
    densityInput.addEventListener("input", () => {
      densityInput.dataset.autoFilled = "false";
    });
  });
}

function initPgmPicker(optic, form) {
  const picker = optic.querySelector("[data-pgm-picker]");
  const fields = form.querySelector("[data-pgm-fields]");
  let requestId = 0;
  picker.addEventListener("change", async () => {
    requestId += 1;
    const currentRequest = requestId;
    const url = new URL(picker.dataset.fieldsUrl, window.location.href);
    url.searchParams.set("id", picker.value);
    const response = await fetch(url);
    if (currentRequest !== requestId || !response.ok) return;
    fields.innerHTML = await response.text();
    bindPgmEditorFields(fields);
    const editor = fields.querySelector("[data-layer-editor]");
    if (editor) initLayerEditor(editor, form);
    form.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

function initPgmMirrorPreview(optic, form) {
  const image = optic.querySelector("[data-pgm-mirror-image]");
  const status = optic.querySelector("[data-pgm-mirror-status]");
  let requestId = 0;
  const update = debounce(async () => {
    requestId += 1;
    const currentRequest = requestId;
    try {
      const response = await fetch(form.dataset.stackPreviewUrl, { method: "POST", body: new FormData(form) });
      const payload = await response.json();
      if (currentRequest !== requestId) return;
      if (payload.ok) {
        image.src = payload.image;
        image.classList.remove("is-hidden");
        status.textContent = "Ready";
      } else {
        status.textContent = payload.error || "Schematic unavailable.";
      }
    } catch {
      if (currentRequest === requestId) status.textContent = "Schematic request failed.";
    }
  }, 400);
  form.addEventListener("input", update);
  form.addEventListener("change", update);
  update();
}

function initPgmWorkspace(setupForm) {
  const gratingForm = document.querySelector("[data-pgm-grating-form]");
  const mirrorForm = document.querySelector("[data-pgm-mirror-form]");
  const errorBox = setupForm.querySelector("[data-pgm-error]");
  const savedMessage = setupForm.querySelector("[data-pgm-saved-message]");
  const startPicker = document.querySelector("[data-pgm-start]");
  const saveButton = setupForm.querySelector("[data-pgm-save]");
  const runButton = setupForm.querySelector('button[type="submit"]');

  startPicker.addEventListener("change", () => { window.location.assign(startPicker.value); });
  initPgmPicker(gratingForm.closest("[data-pgm-optic]"), gratingForm);
  initPgmPicker(mirrorForm.closest("[data-pgm-optic]"), mirrorForm);
  const mirrorEditor = mirrorForm.querySelector("[data-layer-editor]");
  if (mirrorEditor) initLayerEditor(mirrorEditor, mirrorForm);
  initPgmMirrorPreview(mirrorForm.closest("[data-pgm-optic]"), mirrorForm);
  gratingForm.dispatchEvent(new Event("change", { bubbles: true }));
  [gratingForm, mirrorForm].forEach((form) => {
    form.addEventListener("submit", (event) => event.preventDefault());
  });

  const mergedForm = () => {
    const merged = new FormData(setupForm);
    for (const [form, prefix] of [[gratingForm, "grating."], [mirrorForm, "mirror."]]) {
      for (const [name, value] of new FormData(form)) merged.append(prefix + name, value);
    }
    return merged;
  };
  const send = async (url) => {
    errorBox.classList.add("is-hidden");
    savedMessage.classList.add("is-hidden");
    runButton.disabled = true;
    saveButton.disabled = true;
    try {
      const response = await fetch(url, { method: "POST", body: mergedForm() });
      const payload = await response.json();
      if (response.ok) return payload;
      errorBox.textContent = payload.error || "This setup could not be used.";
    } catch {
      errorBox.textContent = "The request failed. Check that the app is still running.";
    } finally {
      runButton.disabled = false;
      saveButton.disabled = false;
    }
    errorBox.classList.remove("is-hidden");
    errorBox.focus();
    return null;
  };

  setupForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = await send(setupForm.action);
    if (payload) window.location.assign(payload.redirect);
  });
  saveButton.addEventListener("click", async () => {
    const payload = await send(setupForm.dataset.saveUrl);
    if (!payload) return;
    const option = new Option(payload.name, payload.url, true, true);
    startPicker.querySelector("[data-pgm-saved-options]").append(option);
    setupForm.querySelector("[data-pgm-id]").value = payload.id;
    savedMessage.textContent = `Saved as ${payload.name}.`;
    savedMessage.classList.remove("is-hidden");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const setupForm = document.querySelector("[data-pgm-setup-form]");
  if (setupForm) initPgmWorkspace(setupForm);
});
