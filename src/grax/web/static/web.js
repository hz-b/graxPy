function toggleSectionFields(section, enabled) {
  section.classList.toggle("is-hidden", !enabled);
  section.querySelectorAll("input, select, textarea").forEach((field) => {
    field.disabled = !enabled;
  });
}

function syncGratingSections(select) {
  const selected = select.value;
  document.querySelectorAll("[data-grating-section]").forEach((section) => {
    toggleSectionFields(section, section.dataset.gratingSection === selected);
  });
}

function defaultStackForGrating(select) {
  const stackSelect = select.closest("form")?.querySelector("[data-stack-type]");
  if (!stackSelect) return;
  syncStackSections(stackSelect);
}

function syncStackSections(select) {
  const mode = select.value;
  // The custom layer table needs room: let its fieldset span the whole form grid.
  select.closest("fieldset")?.classList.toggle("is-wide", mode === "custom");
  document.querySelectorAll("[data-stack-controls]").forEach((section) => {
    toggleSectionFields(section, mode === "multilayer");
  });
  document.querySelectorAll("[data-single-layer-controls]").forEach((section) => {
    toggleSectionFields(section, mode === "single_layer");
  });
  document.querySelectorAll("[data-custom-layer-controls]").forEach((section) => {
    toggleSectionFields(section, mode === "custom");
  });
  document.querySelectorAll("[data-top-cap-controls]").forEach((section) => {
    toggleSectionFields(section, mode !== "bare");
  });
}

function initLayerEditor(editor, form) {
  const rows = editor.querySelector("[data-layer-rows]");
  const template = editor.querySelector("[data-layer-template]");
  let nextKey = rows.querySelectorAll("[data-layer-row]").length;

  const notify = () => form.dispatchEvent(new Event("change", { bubbles: true }));
  const renumber = () => {
    const all = rows.querySelectorAll("[data-layer-row]");
    all.forEach((row, index) => {
      row.querySelector("[data-layer-index]").textContent = String(index + 1);
      row.querySelector("[data-layer-remove]").disabled = all.length === 1;
      row.querySelector("[data-layer-up]").disabled = index === 0;
      row.querySelector("[data-layer-down]").disabled = index === all.length - 1;
    });
  };

  editor.querySelectorAll("[data-layer-add]").forEach((button) => {
    button.addEventListener("click", () => {
      const html = template.innerHTML.replaceAll("__KEY__", String(nextKey));
      nextKey += 1;
      rows.insertAdjacentHTML("beforeend", html);
      const row = rows.lastElementChild;
      row.querySelector("[data-layer-kind]").value = button.dataset.layerAdd || "single";
      row.querySelector("[data-bilayer-fields]").classList.toggle("is-hidden", button.dataset.layerAdd !== "multilayer");
      renumber();
      notify();
    });
  });
  rows.addEventListener("change", (event) => {
    if (event.target.matches("[data-layer-kind]")) {
      event.target.closest("[data-layer-row]").querySelector("[data-bilayer-fields]")
        .classList.toggle("is-hidden", event.target.value !== "multilayer");
    }
  });

  rows.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    const row = event.target.closest("[data-layer-row]");
    if (!button || !row) {
      return;
    }
    if (button.matches("[data-layer-remove]") && rows.children.length > 1) {
      row.remove();
    } else if (button.matches("[data-layer-up]") && row.previousElementSibling) {
      rows.insertBefore(row, row.previousElementSibling);
    } else if (button.matches("[data-layer-down]") && row.nextElementSibling) {
      rows.insertBefore(row.nextElementSibling, row);
    } else {
      return;
    }
    renumber();
    notify();
  });

  // Rows added later are not covered by initMaterialDensitySync, so delegate.
  rows.addEventListener("input", (event) => {
    if (event.target.matches("[data-material-select]")) {
      syncMaterialDensity(event.target);
    } else if (event.target.matches("[data-material-density]")) {
      event.target.dataset.autoFilled = "false";
    }
  });
  renumber();
}

function syncMaterialDensity(field) {
  const fieldKey = field.dataset.materialSelect;
  if (!fieldKey) {
    return;
  }
  const densityInput = document.querySelector(`[data-material-density="${fieldKey}"]`);
  if (!densityInput) {
    return;
  }
  const datalistId = field.getAttribute("list");
  const datalist = datalistId ? document.getElementById(datalistId) : null;
  const matchedOption = datalist
    ? Array.from(datalist.options).find((option) => option.value === field.value.trim())
    : null;
  const density = matchedOption ? matchedOption.dataset.density || "" : "";
  const previousAutoDensity = densityInput.dataset.autoDensity || "";

  if (density !== "") {
    densityInput.value = density;
    densityInput.placeholder = density;
    densityInput.dataset.autoDensity = density;
    densityInput.dataset.autoFilled = "true";
    return;
  }

  if (
    densityInput.dataset.autoFilled === "true" &&
    densityInput.value === previousAutoDensity
  ) {
    densityInput.value = "";
  }
  densityInput.placeholder = "";
  densityInput.dataset.autoDensity = "";
  densityInput.dataset.autoFilled = "false";
}

function syncWorkerFields(select) {
  const isManual = select.value === "manual";
  document.querySelectorAll("[data-manual-workers]").forEach((field) => {
    field.classList.toggle("is-hidden", !isManual);
    field.querySelectorAll("input").forEach((input) => {
      input.disabled = !isManual;
    });
  });
}

function syncRunWorkflow(select) {
  const workflow = select.value;
  const descriptions = {
    fixed_angle: {
      title: "Fixed angle",
      summary: "Scans an energy range while keeping the grazing angle fixed. Use it to calculate the spectral response at one incidence angle.",
    },
    monochromator: {
      title: "Monochromator",
      summary: "Scans energy while adjusting the incidence and diffraction geometry for a constant CFF. Use it to model a grating operated as a monochromator.",
    },
    multilayer_theta_search: {
      title: "Multilayer theta search",
      summary: "Searches for the best grazing angle at each energy for a multilayer grating. Use it to follow the multilayer efficiency peak across an energy range.",
    },
    parameter_study: {
      title: "Parameter study",
      summary: "Varies Fourier orders and spatial resolutions at one energy and angle. Use it to check numerical convergence and choose stable simulation settings.",
    },
    custom_grid: {
      title: "Custom grid",
      summary: "Scans any supported grating or simulation parameters, independently or in coupled pairs. Use it for design exploration and nonstandard parameter combinations.",
    },
  };
  const description = descriptions[workflow];
  const panel = select.closest("form")?.querySelector("[data-run-workflow-description]");
  if (panel && description) {
    panel.querySelector("[data-run-workflow-title]").textContent = description.title;
    panel.querySelector("[data-run-workflow-summary]").textContent = description.summary;
  }
  if (workflow === "parameter_study") {
    const form = select.closest("form");
    const energy = form?.querySelector('[name="energy_start_ev"]');
    const studyEnergy = form?.querySelector('[name="study_energy_ev"]');
    const angle = form?.querySelector('[name="grazing_angle_deg"]');
    const studyAngle = form?.querySelector('[name="study_grazing_angle_deg"]');
    if (energy && studyEnergy && energy.value && studyEnergy.dataset.userEdited !== "true") {
      studyEnergy.value = energy.value;
    }
    if (angle && studyAngle && angle.value && studyAngle.dataset.userEdited !== "true") {
      studyAngle.value = angle.value;
    }
  }
  document.querySelectorAll("[data-workflow-fields]").forEach((section) => {
    const allowed = (section.dataset.workflowFields || "").split(/\s+/).filter(Boolean);
    toggleSectionFields(section, allowed.includes(workflow));
  });
  select.closest("form")?.querySelectorAll("[data-custom-axis]").forEach(syncCustomGridAxisInputs);
  select.closest("form")?.querySelector("[data-custom-grid-builder]")?.dispatchEvent(new Event("workflowchange"));
  syncFourierRangeWarning(select.closest("form"));
}

function syncCustomGridAxisInputs(row) {
  const isRange = row.querySelector("[data-axis-mode]").value === "range";
  const active = row.closest("form").querySelector("[data-run-workflow]").value === "custom_grid";
  const list = row.querySelector("[data-axis-list]");
  const range = row.querySelector("[data-axis-range]");
  list.classList.toggle("is-hidden", isRange);
  range.classList.toggle("is-hidden", !isRange);
  list.querySelector("textarea").disabled = !active || isRange;
  range.querySelectorAll("input").forEach((input) => { input.disabled = !active || !isRange; });
  const generated = row.querySelector("[data-axis-generated]");
  generated.classList.toggle("is-hidden", !isRange);
  if (!isRange) generated.textContent = "";
}

function syncFourierRangeWarning(form) {
  if (!form) return;
  const minimum = Number(form.querySelector('[name="study_fourier_min"]')?.value);
  const maximum = Number(form.querySelector('[name="study_fourier_max"]')?.value);
  const requested = Number(form.querySelector('[name="study_fourier_points"]')?.value);
  const warning = form.querySelector("[data-fourier-range-warning]");
  if (!warning || !Number.isFinite(minimum) || !Number.isFinite(maximum) || !Number.isFinite(requested)) return;
  const available = Math.max(0, Math.floor(maximum) - Math.ceil(minimum) + 1);
  if (maximum < minimum || available === 0) {
    warning.textContent = "The Fourier-order maximum must be at least the minimum, with integer bounds.";
    return;
  }
  const lower = Math.ceil(minimum);
  const upper = Math.floor(maximum);
  const roundLikeBackend = (value) => {
    const floor = Math.floor(value);
    const fraction = value - floor;
    return fraction > 0.5 || (fraction === 0.5 && floor % 2 !== 0) ? floor + 1 : floor;
  };
  const effectiveCount = Math.min(Math.max(1, Math.floor(requested)), available);
  const values = [];
  if (effectiveCount === available) {
    for (let value = lower; value <= upper; value += 1) values.push(value);
  } else if (effectiveCount > 1) {
    for (let index = 0; index < effectiveCount; index += 1) {
      values.push(roundLikeBackend(lower + (upper - lower) * index / (effectiveCount - 1)));
    }
  } else if (effectiveCount === 1) {
    values.push(lower);
  }
  const distinctValues = [...new Set(values)];
  const fractionalSpacing = requested > 1 && !Number.isInteger((maximum - minimum) / (requested - 1));
  const messages = [];
  if (requested > available) {
    messages.push(`Only ${available} distinct integer values are available; the requested ${requested} values will be reduced to ${available}.`);
  }
  if (fractionalSpacing) {
    messages.push(`The requested spacing is fractional; Fourier orders will be rounded to: ${distinctValues.join(", ")}.`);
  }
  warning.textContent = messages.join(" ");
}

function debounce(fn, delayMs) {
  let timerId = null;
  return (...args) => {
    if (timerId !== null) {
      window.clearTimeout(timerId);
    }
    timerId = window.setTimeout(() => fn(...args), delayMs);
  };
}

function syncGratingRoughness(form) {
  const model = form.querySelector("[data-roughness-model]");
  if (!model) return;
  form.querySelectorAll("[data-interface-sigma]").forEach((label) => label.classList.toggle("is-hidden", model.value === "none"));
  form.querySelectorAll("[data-interface-correlation], [data-random-settings]").forEach((label) => label.classList.toggle("is-hidden", model.value !== "random-interface"));
  form.querySelectorAll("[data-debye-explanation]").forEach((label) => label.classList.toggle("is-hidden", model.value !== "debye-waller"));
}

function initGratingPreview(form) {
  syncGratingRoughness(form);
  form.addEventListener("change", () => syncGratingRoughness(form));
  const roughImage = document.querySelector("[data-roughness-preview-image]");
  const warnings = document.querySelector("[data-roughness-warnings]");
  const previewUrl = form.dataset.previewUrl;
  const image = document.querySelector("[data-grating-preview-image]");
  const status = document.querySelector("[data-grating-preview-status]");
  const loading = document.querySelector("[data-grating-preview-loading]");
  const errorBox = document.querySelector("[data-grating-preview-error]");
  let previewValid = true;
  let requestId = 0;

  function clearInvalidFields() {
    form.querySelectorAll(".is-invalid").forEach((field) => {
      field.classList.remove("is-invalid");
      field.removeAttribute("aria-invalid");
    });
  }

  function showPreviewError(payload) {
    previewValid = false;
    const fields = payload.error_fields || [];
    clearInvalidFields();
    fields.forEach((name) => {
      const field = form.querySelector(`[name="${name}"]`);
      if (field) {
        field.classList.add("is-invalid");
        field.setAttribute("aria-invalid", "true");
      }
    });
    if (errorBox) {
      errorBox.textContent = payload.error || "Preview inputs are invalid.";
      errorBox.classList.remove("is-hidden");
    }
    status.textContent = payload.error_category === "geometry" ? "Invalid geometry" : "Validation error";
  }

  function clearPreviewError() {
    previewValid = true;
    clearInvalidFields();
    if (errorBox) {
      errorBox.textContent = "";
      errorBox.classList.add("is-hidden");
    }
  }

  const updatePreview = debounce(async () => {
    if (materialPauseTimer !== null) return;
    requestId += 1;
    const currentRequest = requestId;
    const afmPending = form.querySelector('[name="grating_type"]')?.value === "afm" &&
      !form.querySelector("[data-afm-profile-path]")?.value;
    if (afmPending) {
      previewValid = false;
      clearInvalidFields();
      const afmError = form.dataset.afmProcessingError || "";
      if (errorBox) {
        errorBox.textContent = afmError;
        errorBox.classList.toggle("is-hidden", !afmError);
      }
      image.classList.add("is-hidden");
      if (roughImage) roughImage.classList.add("is-hidden");
      if (warnings) warnings.textContent = "";
      loading.textContent = "AFM preview";
      status.textContent = afmError ? "AFM processing failed" :
        "Review the AFM period. Save will process the current settings.";
      return;
    }
    loading.textContent = "Rendering";
    try {
      const response = await fetch(previewUrl, {
        method: "POST",
        body: new FormData(form),
      });
      const payload = await response.json();
      if (currentRequest !== requestId) {
        return;
      }
      loading.textContent = payload.ok ? "Ready" : "Invalid inputs";
      if (roughImage) {
        roughImage.classList.toggle("is-hidden", !payload.ok || !payload.roughness_preview_url);
        if (payload.roughness_preview_url) roughImage.src = payload.roughness_preview_url;
      }
      if (warnings) warnings.textContent = (payload.warnings || []).join(" ");
      if (payload.ok) {
        clearPreviewError();
        image.src = payload.preview_url;
        image.classList.remove("is-hidden");
        status.textContent = "Ready";
      } else {
        showPreviewError(payload);
      }
    } catch (error) {
      if (currentRequest !== requestId) {
        return;
      }
      loading.textContent = "Error";
      status.textContent = "Preview request failed.";
    }
  }, 250);

  let materialPauseTimer = null;
  function validateMaterialNow() {
    if (materialPauseTimer !== null) window.clearTimeout(materialPauseTimer);
    materialPauseTimer = null;
    updatePreview();
  }
  form.addEventListener("input", (event) => {
    if (event.target.matches("[data-material-select]")) {
      // Material formulas can be invalid midway through typing ("A" before "Au").
      // Invalidate an in-flight preview and wait until the user pauses.
      requestId += 1;
      if (materialPauseTimer !== null) window.clearTimeout(materialPauseTimer);
      materialPauseTimer = window.setTimeout(validateMaterialNow, 2000);
    } else {
      validateMaterialNow();
    }
  });
  form.addEventListener("change", validateMaterialNow);
  form.addEventListener("focusout", (event) => {
    if (event.target.matches("[data-material-select]")) validateMaterialNow();
  });
  form.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches("[data-material-select]")) {
      event.preventDefault();
      validateMaterialNow();
    }
  });
  form.addEventListener("submit", (event) => {
    if (form.querySelector('[name="grating_type"]')?.value === "afm") {
      if (!form.querySelector("[data-afm-profile-path]")?.value) {
        event.preventDefault();
        form.dispatchEvent(new Event("grax:afm-save-requested"));
      }
      // The server validates the processed AFM profile and coating fields.
      // A pending/previous live-preview request must not block a valid save.
      return;
    }
    if (!previewValid) {
      event.preventDefault();
      if (errorBox) errorBox.focus();
    }
  });
  if (form.querySelector('[name="grating_type"]')?.value === "afm" &&
      !form.querySelector("[data-afm-profile-path]")?.value) {
    updatePreview();
  }
}

function initMaterialDensitySync() {
  document.querySelectorAll("[data-material-select]").forEach((field) => {
    syncMaterialDensity(field);
    field.addEventListener("input", () => {
      syncMaterialDensity(field);
    });
    field.addEventListener("change", () => {
      syncMaterialDensity(field);
    });
  });
  document.querySelectorAll("[data-material-density]").forEach((densityInput) => {
    densityInput.addEventListener("input", () => {
      densityInput.dataset.autoFilled = "false";
    });
  });
}

function createOrderCheckbox(runId, order) {
  const label = document.createElement("label");
  label.className = "inline-check";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.name = `orders_${runId}`;
  input.value = String(order);
  const span = document.createElement("span");
  span.textContent = `Order ${order}`;
  label.append(input, span);
  return label;
}

function renderPlotlyFigure(container, figureJson) {
  if (!container || !window.Plotly || !figureJson) {
    return;
  }
  const parsed = typeof figureJson === "string" ? JSON.parse(figureJson) : figureJson;
  window.Plotly.react(container, parsed.data || [], parsed.layout || {}, {
    responsive: true,
    displaylogo: false,
  });
}

// The survey figures and the (d, blaze) picker live in different sections of the
// design detail page, so they talk through document-level events instead of
// reaching into each other: a click on the map asks the picker to toggle a case,
// and the picker announces its selection so the map can mark it.
const SURVEY_CELL_PICKED = "grax:survey-cell-picked";
const DESIGN_SELECTION_CHANGED = "grax:design-selection-changed";

function formatDesignValue(value) {
  return Number(value).toFixed(3);
}

function designPairKey(d, blaze) {
  return `${formatDesignValue(d)},${formatDesignValue(blaze)}`;
}

function nearestGridValue(values, target) {
  let best = null;
  let bestDistance = Infinity;
  values.forEach((value) => {
    const distance = Math.abs(Number(value) - Number(target));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = Number(value);
    }
  });
  return best;
}

function surveyFigureLayout(meta, xTitle, yTitle, title, options) {
  // The survey energy is a property of the survey, so it belongs on the survey
  // plots only -- naming it on an energy scan would be actively misleading.
  const parts = (options || {}).withEnergy === false
    ? [meta.coating_label, `order ${meta.diffraction_order}`]
    : [meta.coating_label, `${meta.target_energy_ev} eV`, `order ${meta.diffraction_order}`];
  const subtitle = parts
    .filter((part) => part !== undefined && part !== null && part !== "")
    .join(" · ");
  return {
    template: "plotly_white",
    title: {text: subtitle ? `${title}<br><sub>${subtitle}</sub>` : title, x: 0.02},
    margin: {l: 64, r: 24, t: 72, b: 56},
    xaxis: {title: {text: xTitle}},
    yaxis: {title: {text: yTitle}},
    showlegend: false,
  };
}

function buildSurveyFigures(options, meta, selectedKeys) {
  const dValues = (options.d_values || []).map(Number);
  const blazeValues = (options.blaze_values || []).map(Number);
  const efficiency = options.efficiency || {};
  const perD = options.per_d || [];

  const ridgeD = perD.map((pair) => Number(pair[0]));
  const ridgeBlaze = perD.map((pair) => Number(pair[1]));
  const ridgeEfficiency = perD.map((pair) => efficiency[designPairKey(pair[0], pair[1])] ?? null);

  const optimalBlaze = {
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x: ridgeD,
        y: ridgeBlaze,
        line: {color: "#b0b7bd", width: 1.4},
        marker: {
          size: 11,
          color: ridgeEfficiency,
          colorscale: "Viridis",
          colorbar: {title: {text: "peak efficiency"}},
        },
        customdata: ridgeEfficiency,
        hovertemplate:
          "d = %{x:.3f} nm<br>blaze = %{y:.3f} deg<br>efficiency = %{customdata:.4f}<extra></extra>",
      },
    ],
    layout: surveyFigureLayout(
      meta,
      "Bilayer d-spacing (nm)",
      "Optimal blaze angle (deg)",
      "Optimal blaze angle versus d-spacing",
    ),
  };

  const maxEfficiency = {
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x: ridgeD,
        y: ridgeEfficiency,
        line: {color: "#b0b7bd", width: 1.4},
        marker: {
          size: 11,
          color: ridgeBlaze,
          colorscale: "Plasma",
          colorbar: {title: {text: "optimal blaze (deg)"}},
        },
        customdata: ridgeBlaze,
        hovertemplate:
          "d = %{x:.3f} nm<br>efficiency = %{y:.4f}<br>blaze = %{customdata:.3f} deg<extra></extra>",
      },
    ],
    layout: surveyFigureLayout(
      meta,
      "Bilayer d-spacing (nm)",
      "Max selected-order efficiency",
      "Max efficiency versus d-spacing",
    ),
  };

  // z is indexed [blaze][d]; a cell the survey never solved stays null so Plotly
  // leaves a gap instead of drawing a misleading zero.
  const z = blazeValues.map((blaze) =>
    dValues.map((d) => {
      const value = efficiency[designPairKey(d, blaze)];
      return value === undefined ? null : Number(value);
    }),
  );
  const selected = Array.from(selectedKeys || []).map((key) => key.split(",").map(Number));
  const heatmapLayout = surveyFigureLayout(
    meta,
    "Bilayer d-spacing (nm)",
    "Blaze angle (deg)",
    "Peak efficiency over (d, blaze) — click a cell to scan it",
  );
  heatmapLayout.showlegend = true;
  // Inside the axes, like the matplotlib version: a legend above the plot would
  // land on the title's subtitle line.
  heatmapLayout.legend = {
    orientation: "h",
    x: 0.02,
    xanchor: "left",
    y: 0.98,
    yanchor: "top",
    bgcolor: "rgba(255, 255, 255, 0.78)",
  };
  const heatmap = {
    data: [
      {
        type: "heatmap",
        x: dValues,
        y: blazeValues,
        z,
        colorscale: "Viridis",
        colorbar: {title: {text: "peak efficiency"}},
        hovertemplate:
          "d = %{x:.3f} nm<br>blaze = %{y:.3f} deg<br>efficiency = %{z:.4f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "optimal blaze per d",
        x: ridgeD,
        y: ridgeBlaze,
        line: {color: "#ffffff", width: 1.4},
        marker: {color: "#ffffff", size: 6},
        hoverinfo: "skip",
      },
      {
        type: "scatter",
        mode: "markers",
        name: "selected designs",
        x: selected.map((pair) => pair[0]),
        y: selected.map((pair) => pair[1]),
        // square-open takes its stroke from marker.color, not marker.line.
        marker: {symbol: "square-open", size: 16, color: "#ff3b30", line: {width: 3}},
        hoverinfo: "skip",
      },
    ],
    layout: heatmapLayout,
  };

  return {optimal_blaze: optimalBlaze, max_efficiency: maxEfficiency, heatmap};
}

function initEnergyScanFigure(root) {
  const stage = root.querySelector("[data-energy-scan-figure]");
  const liveUrl = root.dataset.energyScanLiveUrl;
  if (!stage || !liveUrl || !window.Plotly) {
    return;
  }
  let meta;
  try {
    meta = JSON.parse(root.dataset.surveyPlotMeta || "{}");
  } catch (error) {
    meta = {};
  }

  function draw(designs) {
    const layout = surveyFigureLayout(
      meta,
      "Photon energy (eV)",
      "Selected-order efficiency",
      "Efficiency versus energy",
      {withEnergy: false},
    );
    // Below the axes: one entry per design, and the curves themselves leave no
    // reliable free corner.
    layout.showlegend = true;
    layout.legend = {orientation: "h", x: 0, y: -0.22, yanchor: "top"};
    layout.margin = {l: 64, r: 24, t: 72, b: 96};
    renderPlotlyFigure(stage, {
      data: designs.map((design) => ({
        type: "scatter",
        mode: "lines+markers",
        name: `d = ${formatDesignValue(design.d_spacing_nm)} nm, blaze = ${formatDesignValue(design.blaze_angle_deg)} deg`,
        x: design.energies_ev || [],
        y: design.efficiencies || [],
        marker: {size: 5},
        line: {width: 1.6},
        hovertemplate: "%{x:.1f} eV<br>efficiency = %{y:.4f}<extra>%{fullData.name}</extra>",
      })),
      layout,
    });
  }

  async function poll() {
    try {
      const response = await window.fetch(liveUrl, {cache: "no-store"});
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      if ((payload.designs || []).some((design) => (design.energies_ev || []).length)) {
        draw(payload.designs);
      }
    } catch (error) {
      // A dropped poll is not worth surfacing; the next one will catch up.
    }
  }

  poll();
  window.setInterval(poll, 4000);
}

function initSurveyFigures(root) {
  let options;
  let meta;
  try {
    options = JSON.parse(root.dataset.designOptions || "{}");
    meta = JSON.parse(root.dataset.surveyPlotMeta || "{}");
  } catch (error) {
    return;
  }
  const stages = new Map();
  root.querySelectorAll("[data-survey-figure]").forEach((node) => {
    stages.set(node.dataset.surveyFigure, node);
  });
  if (stages.size === 0 || !window.Plotly) {
    return;
  }
  const dValues = (options.d_values || []).map(Number);
  const blazeValues = (options.blaze_values || []).map(Number);
  let selectedKeys = new Set();

  function draw() {
    const figures = buildSurveyFigures(options, meta, selectedKeys);
    stages.forEach((node, name) => {
      if (figures[name]) {
        renderPlotlyFigure(node, figures[name]);
      }
    });
  }

  draw();

  // While the survey runs its table is rewritten after every cell, so re-reading
  // it shows the grid filling in. The monitor reloads the page when the stage
  // finishes, which is what ends this polling.
  const liveUrl = root.dataset.surveyLiveUrl;
  if (liveUrl) {
    window.setInterval(async () => {
      try {
        const response = await window.fetch(liveUrl, {cache: "no-store"});
        if (!response.ok) {
          return;
        }
        const fresh = await response.json();
        if ((fresh.per_d || []).length === 0) {
          return;
        }
        options = fresh;
        dValues.splice(0, dValues.length, ...(fresh.d_values || []).map(Number));
        blazeValues.splice(0, blazeValues.length, ...(fresh.blaze_values || []).map(Number));
        draw();
      } catch (error) {
        // A dropped poll is not worth surfacing; the next one will catch up.
      }
    }, 4000);
  }

  const heatmapNode = stages.get("heatmap");
  if (heatmapNode && typeof heatmapNode.on === "function") {
    heatmapNode.on("plotly_click", (event) => {
      const point = (event.points || [])[0];
      if (!point) {
        return;
      }
      const d = nearestGridValue(dValues, point.x);
      const blaze = nearestGridValue(blazeValues, point.y);
      if (d === null || blaze === null) {
        return;
      }
      if ((options.efficiency || {})[designPairKey(d, blaze)] === undefined) {
        return; // an unsolved cell is not a design anyone can scan
      }
      document.dispatchEvent(
        new CustomEvent(SURVEY_CELL_PICKED, {detail: {d, blaze}}),
      );
    });
  }

  document.addEventListener(DESIGN_SELECTION_CHANGED, (event) => {
    selectedKeys = new Set((event.detail && event.detail.pairs) || []);
    draw();
  });
}

function initPlotWorkspace(form) {
  const previewUrl = form.dataset.previewUrl;
  const picker = form.querySelector("[data-run-picker]");
  const runList = form.querySelector("[data-plot-run-list]");
  const template = document.querySelector("[data-plot-run-template]");
  const seriesTemplate = document.querySelector("[data-plot-series-style-template]");
  const previewContainer = document.querySelector("[data-plot-preview]");
  const previewStatus = document.querySelector("[data-plot-preview-status]");
  const loading = document.querySelector("[data-plot-preview-loading]");
  const saveButton = document.querySelector("[data-save-plot]");
  const selectedRunSummary = document.querySelector("[data-selected-run-summary]");
  const seriesStyleList = document.querySelector("[data-series-style-list]");
  let requestId = 0;

  function setPreviewFeedback(state, message) {
    const labels = {
      idle: "Idle",
      rendering: "Rendering",
      ready: "Ready",
      error: "Error",
    };
    loading.textContent = labels[state];
    loading.dataset.state = state;
    loading.classList.toggle("is-rendering", state === "rendering");
    loading.classList.toggle("is-ready", state === "ready");
    loading.classList.toggle("is-error", state === "error");
    previewStatus.classList.toggle("is-error", state === "error");
    previewStatus.textContent = message;
  }

  function summarizeRuns(selectedRuns) {
    selectedRunSummary.replaceChildren();
    selectedRuns.forEach((run) => {
      const row = document.createElement("div");
      row.className = "row";
      const title = document.createElement("strong");
      title.textContent = run.name;
      const orders = document.createElement("span");
      orders.textContent = `Orders: ${run.orders.join(", ")}`;
      row.append(title, orders);
      selectedRunSummary.append(row);
    });
  }

  function attachStyleInputHandlers(item) {
    item.querySelectorAll("input, select").forEach((field) => {
      field.addEventListener("input", refreshPreview);
      field.addEventListener("change", refreshPreview);
    });
  }

  function syncSeriesControls(seriesControls) {
    seriesStyleList.replaceChildren();
    seriesControls.forEach((series) => {
      const item = seriesTemplate.content.firstElementChild.cloneNode(true);
      item.querySelector("[data-series-style-title]").textContent = series.label;

      const legendLabelInput = item.querySelector("[data-series-legend-label]");
      legendLabelInput.name = `legend_label_${series.series_token}`;
      legendLabelInput.value = series.legend_label;

      const colorInput = item.querySelector("[data-series-color]");
      colorInput.name = `color_${series.series_token}`;
      colorInput.value = series.color;

      const markerSelect = item.querySelector("[data-series-marker-symbol]");
      markerSelect.name = `marker_symbol_${series.series_token}`;
      markerSelect.value = series.marker_symbol;

      const markerSizeInput = item.querySelector("[data-series-marker-size]");
      markerSizeInput.name = `marker_size_${series.series_token}`;
      markerSizeInput.value = String(series.marker_size);

      attachStyleInputHandlers(item);
      seriesStyleList.append(item);
    });
  }

  const refreshPreview = debounce(async () => {
    requestId += 1;
    const currentRequest = requestId;
    setPreviewFeedback("rendering", "Updating the live preview…");
    saveButton.disabled = true;
    try {
      const response = await fetch(previewUrl, {
        method: "POST",
        body: new FormData(form),
      });
      const payload = await response.json();
      if (currentRequest !== requestId) {
        return;
      }
      if (payload.ok) {
        renderPlotlyFigure(previewContainer, payload.figure_json);
        setPreviewFeedback("ready", "Preview is up to date.");
        saveButton.disabled = false;
        summarizeRuns(payload.selected_runs || []);
        syncSeriesControls(payload.series_controls || []);
      } else {
        setPreviewFeedback("error", payload.error || "Select runs and orders.");
        saveButton.disabled = true;
        seriesStyleList.replaceChildren();
      }
    } catch (error) {
      if (currentRequest !== requestId) {
        return;
      }
      setPreviewFeedback("error", "The preview could not be updated. Try again.");
      saveButton.disabled = true;
      seriesStyleList.replaceChildren();
    }
  }, 250);

  function bindRunItem(item) {
    item.querySelectorAll("input").forEach((field) => {
      field.addEventListener("change", refreshPreview);
    });
    item.querySelector("[data-remove-run]").addEventListener("click", () => {
      item.remove();
      refreshPreview();
    });
  }

  picker?.addEventListener("change", () => {
    const option = picker.selectedOptions[0];
    if (!option || option.value === "") {
      return;
    }
    if (runList.querySelector(`[data-plot-run-item][data-run-id="${option.value}"]`)) {
      picker.value = "";
      return;
    }
    const item = template.content.firstElementChild.cloneNode(true);
    item.dataset.runId = option.value;
    item.querySelector("[data-run-title]").textContent = option.dataset.runName;
    item.querySelector("[data-run-id-field]").value = option.value;
    const orderGrid = item.querySelector("[data-order-grid]");
    const orders = (option.dataset.runOrders || "")
      .split(",")
      .filter((value) => value !== "")
      .map((value) => Number.parseInt(value, 10));
    orders.forEach((order, index) => {
      const checkbox = createOrderCheckbox(option.value, order);
      checkbox.querySelector("input").checked = index === 0;
      orderGrid.append(checkbox);
    });
    runList.append(item);
    bindRunItem(item);
    picker.value = "";
    refreshPreview();
  });

  form.querySelectorAll('input[name="title"], select[name="x_axis_type"], select[name="y_axis_type"]').forEach((field) => {
    field.addEventListener("input", refreshPreview);
    field.addEventListener("change", refreshPreview);
  });
}

function initSavedPlotFigure(container) {
  const figureJson = container.dataset.figureJson;
  if (!figureJson) {
    return;
  }
  renderPlotlyFigure(container, JSON.parse(figureJson));
}

function initCustomGridBuilder(root) {
  const catalogNode = root.querySelector("[data-custom-grid-catalog]");
  const defaultNode = root.querySelector("[data-custom-grid-default]");
  const configField = root.querySelector("[data-custom-grid-config]");
  const axesRoot = root.querySelector("[data-custom-axes]");
  const groupsRoot = root.querySelector("[data-custom-groups]");
  if (!catalogNode || !defaultNode || !configField || !axesRoot || !groupsRoot) return;

  const baseCatalog = JSON.parse(catalogNode.textContent || "[]");
  const initial = JSON.parse(defaultNode.textContent || "{}");
  const threshold = Number(root.dataset.warningThreshold || 1000);
  const countNode = root.querySelector("[data-custom-grid-count]");
  const warningNode = root.querySelector("[data-custom-grid-warning]");
  const previewNode = root.querySelector("[data-custom-grid-preview]");
  const coupledEditors = new Map();
  const materialAxes = new WeakMap();
  let casePreview = null;

  function appendPreviewCases() {
    if (!casePreview) return;
    const end = Math.min(casePreview.rendered + 100, casePreview.count);
    const lines = [];
    for (let index = casePreview.rendered; index < end; index += 1) {
      let remainder = index;
      const selected = new Array(casePreview.axes.length);
      for (let axis = casePreview.axes.length - 1; axis >= 0; axis -= 1) {
        const values = casePreview.axes[axis];
        selected[axis] = values[remainder % values.length];
        remainder = Math.floor(remainder / values.length);
      }
      const item = Object.assign({}, ...selected);
      lines.push(`${index + 1}. ${Object.entries(item).map(([key, value]) => `${descriptor(key)?.label || key}=${materialText(value)}`).join("; ") || "Default settings"}`);
    }
    if (lines.length) previewNode.append(document.createTextNode((casePreview.rendered ? "\n" : "") + lines.join("\n")));
    casePreview.rendered = end;
  }

  previewNode.addEventListener("scroll", () => {
    if (previewNode.scrollTop + previewNode.clientHeight >= previewNode.scrollHeight - 100) appendPreviewCases();
  });

  function descriptorOptions(selected, blocked = new Set()) {
    const groups = new Map();
    baseCatalog.forEach((item) => {
      if (blocked.has(item.id) && item.id !== selected) return;
      if (!groups.has(item.category)) groups.set(item.category, []);
      groups.get(item.category).push(item);
    });
    const select = document.createElement("select");
    select.className = "custom-grid-parameter";
    groups.forEach((items, category) => {
      const group = document.createElement("optgroup");
      group.label = category;
      items.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.label;
        option.selected = item.id === selected;
        group.append(option);
      });
      select.append(group);
    });
    return select;
  }

  function descriptor(id) {
    return baseCatalog.find((item) => item.id === id);
  }

  function availableParameters() {
    const used = new Set([...root.querySelectorAll(".custom-grid-parameter")].map((select) => select.value));
    return baseCatalog.filter((item) => !used.has(item.id));
  }

  function sourceAxes() {
    return [...axesRoot.querySelectorAll("[data-custom-axis]")];
  }

  function syncParameterOptions() {
    const all = [...root.querySelectorAll(".custom-grid-parameter")];
    all.forEach((select) => {
      const used = new Set(all.filter((other) => other !== select).map((other) => other.value));
      [...select.options].forEach((option) => { option.disabled = used.has(option.value); });
    });
    const sources = [...groupsRoot.querySelectorAll("[data-couple-source]")];
    const axes = sourceAxes().map((axis) => axis.querySelector(".custom-grid-parameter").value);
    sources.forEach((select) => {
      const previous = select.value;
      const choices = axes;
      const signature = JSON.stringify(choices);
      if (select.dataset.choices !== signature) {
        select.replaceChildren(new Option("Choose an existing parameter", ""));
        choices.forEach((id) => select.append(new Option(descriptor(id).label, id)));
        select.value = choices.includes(previous) ? previous : "";
        select.dataset.choices = signature;
      }
    });
    const coupledSources = new Set(sources.map((select) => select.value));
    sourceAxes().forEach((axis) => {
      axis.querySelector("[data-axis-coupling-hint]").textContent = coupledSources.has(axis.querySelector(".custom-grid-parameter").value)
        ? "Used by coupled parameters below. Edit these values here to update their tables."
        : "";
    });
    root.querySelector("[data-custom-add-axis]").disabled = !availableParameters().length;
    root.querySelector("[data-custom-add-group]").disabled = !availableParameters().length || !axes.length;
  }

  function splitValues(text) {
    return text.split(/[\n,;]/).map((value) => value.trim()).filter(Boolean);
  }

  function roundHalfEven(value) {
    const lower = Math.floor(value);
    const fraction = value - lower;
    if (Math.abs(fraction - 0.5) < Number.EPSILON * Math.max(1, Math.abs(value))) {
      return lower % 2 === 0 ? lower : lower + 1;
    }
    return Math.round(value);
  }

  function castPreview(value, item) {
    if (!item) throw new Error("Choose a valid parameter.");
    if (item.kind === "int") {
      const number = Number(value);
      if (!Number.isInteger(number)) throw new Error(`${item.label} only accepts integers.`);
      if (item.minimum !== null && item.minimum !== undefined && number < item.minimum) throw new Error(`${item.label} must be at least ${item.minimum}.`);
      return number;
    }
    if (["float", "optional_float"].includes(item.kind)) {
      if (item.kind === "optional_float" && value === "") return null;
      const number = Number(value);
      if (!Number.isFinite(number)) throw new Error(`${item.label} needs numeric values.`);
      if (item.minimum !== null && item.minimum !== undefined && number < item.minimum) throw new Error(`${item.label} must be at least ${item.minimum}.`);
      return number;
    }
    if (item.kind === "material") {
      const material = typeof value === "object" ? {...value} : {name: String(value)};
      material.name = String(material.name || "").trim();
      if (!material.name) throw new Error(`${item.label} needs a material name.`);
      for (const key of ["density_g_cm3", "roughness_sigma_nm"]) {
        const raw = material[key];
        if (raw === "" || raw === null || raw === undefined) { material[key] = null; continue; }
        const number = Number(raw);
        if (!Number.isFinite(number) || (key === "density_g_cm3" ? number <= 0 : number < 0)) throw new Error(`${item.label}: density must be positive and roughness must be non-negative.`);
        material[key] = number;
      }
      return material;
    }
    if (item.choices?.length && !item.choices.includes(value)) throw new Error(`Invalid value for ${item.label}.`);
    return value;
  }

  function materialFields(value, item, onChange, includeName = true) {
    const wrapper = document.createElement("div"); wrapper.className = "custom-grid-material-fields";
    const current = typeof value === "object" && value !== null ? {...value} : {name: value || ""};
    const fields = [["name", "Material", ""], ["density_g_cm3", "Density, g/cm³", "Automatic"]];
    if (item.roughness_path?.length) fields.push(["roughness_sigma_nm", "Roughness σ, nm", "Saved grating"]);
    fields.forEach(([key, title, placeholder]) => {
      if (key === "name" && !includeName) return;
      const label = document.createElement("label"); label.append(title);
      const input = document.createElement("input"); input.type = key === "name" ? "text" : "number";
      if (key !== "name") { input.step = "any"; input.min = key === "density_g_cm3" ? "0.000000000001" : "0"; }
      input.placeholder = placeholder; input.value = current[key] ?? "";
      input.addEventListener("input", () => { current[key] = input.value; onChange({...current}); });
      label.append(input); wrapper.append(label);
    });
    if (item.roughness_path?.length) {
      const hint = document.createElement("small");
      hint.className = "muted";
      hint.textContent = "Roughness uses the model selected in the grating settings.";
      wrapper.append(hint);
    }
    return wrapper;
  }

  function materialText(value) {
    if (!value || typeof value !== "object") return String(value);
    return `${value.name}${value.density_g_cm3 == null ? "" : ` (${value.density_g_cm3} g/cm³)`}${value.roughness_sigma_nm == null ? "" : `; σ=${value.roughness_sigma_nm} nm`}`;
  }

  function makeAxis(axis = {}) {
    const row = document.createElement("article");
    row.className = "custom-grid-card";
    row.dataset.customAxis = "";
    const parameter = descriptorOptions(axis.parameter || availableParameters()[0]?.id);
    const mode = document.createElement("select");
    mode.dataset.axisMode = "";
    mode.innerHTML = '<option value="list">Explicit values</option><option value="range">Linear range</option>';
    mode.value = axis.mode || "list";
    row.innerHTML = '<div class="custom-grid-card-head"><strong>Independent axis</strong><button type="button" class="button danger" data-remove>Remove</button></div>';
    const controls = document.createElement("div");
    controls.className = "custom-grid-axis-controls";
    const parameterLabel = document.createElement("label"); parameterLabel.append("Parameter", parameter);
    const modeLabel = document.createElement("label"); modeLabel.append("Input", mode);
    controls.append(parameterLabel, modeLabel);
    const values = document.createElement("label"); values.dataset.axisList = ""; values.append("Values");
    const textarea = document.createElement("textarea"); textarea.rows = 2; textarea.placeholder = "100, 200, 300";
    textarea.value = (axis.values || []).map((value) => typeof value === "object" ? value.name : value).join(", "); values.append(textarea);
    const materials = document.createElement("div"); materials.dataset.axisMaterials = "";
    let materialSignature = "";
    let materialValues = (axis.values || []).map((value) => typeof value === "object" ? {...value} : {name: String(value)});
    function syncMaterials() {
      const item = descriptor(parameter.value);
      const isMaterial = item?.kind === "material";
      textarea.placeholder = isMaterial ? "Au, Pt, Si" : "100, 200, 300";
      materials.classList.toggle("is-hidden", !isMaterial);
      if (isMaterial) {
        const names = splitValues(textarea.value);
        const signature = JSON.stringify([parameter.value, names]);
        if (signature !== materialSignature) {
          materialValues = names.map((name, index) => materialValues[index]?.name === name ? materialValues[index] : {name});
          materials.replaceChildren();
          const hint = document.createElement("p"); hint.className = "subtle";
          hint.textContent = "Set density and interface roughness for each material. Blank density uses the material default; blank roughness keeps the saved grating setting. A separate scan of either property overrides these values.";
          materials.append(hint);
          materialValues.forEach((value, index) => {
            const line = document.createElement("div"); line.className = "custom-grid-material-row";
            const name = document.createElement("strong"); name.textContent = value.name;
            line.append(name, materialFields(value, item, (updated) => { materialValues[index] = updated; refresh(); }, false));
            materials.append(line);
          });
          materialSignature = signature;
        }
      }
      const active = root.closest("form").querySelector("[data-run-workflow]").value === "custom_grid";
      materials.querySelectorAll("input").forEach((field) => { field.disabled = !active || !isMaterial; });
    }
    materialAxes.set(row, {sync: syncMaterials, read: () => materialValues.map((value) => castPreview(value, descriptor(parameter.value)))});
    const range = document.createElement("div"); range.className = "custom-grid-range"; range.dataset.axisRange = "";
    [["Minimum", "minimum"], ["Maximum", "maximum"], ["Count", "count"]].forEach(([label, key]) => {
      const fieldLabel = document.createElement("label"); const input = document.createElement("input");
      input.type = "number"; input.step = key === "count" ? "1" : "any"; input.dataset[key] = "";
      input.value = axis[key] ?? (key === "count" ? 2 : key === "minimum" ? 0 : 1);
      fieldLabel.append(label, input); range.append(fieldLabel);
    });
    const generated = document.createElement("p"); generated.className = "subtle"; generated.dataset.axisGenerated = "";
    const couplingHint = document.createElement("p");
    couplingHint.className = "subtle";
    couplingHint.dataset.axisCouplingHint = "";
    row.append(controls, values, materials, range, generated, couplingHint);
    function syncMode() {
      const rangeAllowed = ["int", "float", "optional_float"].includes(descriptor(parameter.value)?.kind);
      if (!rangeAllowed) mode.value = "list";
      mode.querySelector('[value="range"]').disabled = !rangeAllowed;
      syncCustomGridAxisInputs(row);
      syncMaterials();
    }
    parameter.addEventListener("change", () => { syncMode(); refresh(); }); mode.addEventListener("change", () => { syncMode(); refresh(); });
    row.addEventListener("input", refresh); row.querySelector("[data-remove]").addEventListener("click", () => { row.remove(); refresh(); });
    axesRoot.append(row); syncMode();
  }

  function makeGroup(group = {}) {
    const row = document.createElement("article");
    row.className = "custom-grid-card";
    row.dataset.customGroup = "";
    row.innerHTML = `<div class="custom-grid-card-head"><strong>Coupled parameter</strong><button type="button" class="button danger" data-remove>Remove</button></div>
      <div class="custom-grid-axis-controls">
        <label>Couple to <select data-couple-source></select></label>
        <label>Parameter to couple <span data-couple-target></span></label>
      </div>
      <label>Enter values <select data-couple-mode><option value="manual">Manually</option><option value="range">As a range</option></select></label>
      <div class="custom-grid-range" data-couple-range>
        <label>Start <input type="number" step="any" data-couple-start value="0"></label>
        <label>Stop <input type="number" step="any" data-couple-stop value="1"></label>
        <label>Number of points <input type="number" min="1" step="1" data-couple-count></label>
      </div>
      <p class="subtle" data-couple-hint></p>
      <p class="notice-error" data-couple-error role="status"></p>
      <div class="custom-grid-table-wrap"><table class="custom-grid-coupling-table"><thead><tr><th scope="col" data-source-heading></th><th scope="col" data-target-heading></th></tr></thead><tbody></tbody></table></div>`;
    const source = row.querySelector("[data-couple-source]");
    const target = descriptorOptions(group.parameters?.[1] || availableParameters()[0]?.id);
    row.querySelector("[data-couple-target]").replaceWith(target);
    const mode = row.querySelector("[data-couple-mode]");
    const range = row.querySelector("[data-couple-range]");
    const start = row.querySelector("[data-couple-start]");
    const stop = row.querySelector("[data-couple-stop]");
    const count = row.querySelector("[data-couple-count]");
    const body = row.querySelector("tbody");
    const state = {manual: (group.rows || []).map((values) => values[group.parameters[1]] ?? ""), values: [], error: "", signature: "", source: "", target: target.value};
    groupsRoot.append(row);
    syncParameterOptions();
    source.value = group.parameters?.[0] || sourceAxes()[0]?.querySelector(".custom-grid-parameter").value || "";

    function update() {
      const active = root.closest("form").querySelector("[data-run-workflow]").value === "custom_grid";
      const item = descriptor(target.value);
      const rangeAllowed = ["int", "float", "optional_float"].includes(item?.kind);
      mode.querySelector('[value="range"]').disabled = !rangeAllowed;
      if (!rangeAllowed) mode.value = "manual";
      const isRange = mode.value === "range";
      range.classList.toggle("is-hidden", !isRange);
      range.querySelectorAll("input").forEach((input) => { input.disabled = !active || !isRange; });
      row.querySelector("[data-source-heading]").textContent = descriptor(source.value)?.label || "Existing values";
      row.querySelector("[data-target-heading]").textContent = item?.label || "Coupled values";
      state.error = "";
      state.values = [];
      let generated = [];
      try {
        const axis = sourceAxes().find((axis) => axis.querySelector(".custom-grid-parameter").value === source.value);
        if (!axis) throw new Error("Choose an existing independent parameter to couple to.");
        state.values = axisValues(axis);
        if (source.value !== state.source) {
          if (state.source) state.manual = [];
          state.source = source.value;
          count.value = state.values.length;
        }
        if (target.value !== state.target) {
          state.manual = [];
          state.target = target.value;
        }
        if (isRange) {
          const size = Number(count.value);
          if (!Number.isInteger(size) || size !== state.values.length) throw new Error(`Number of points must match ${descriptor(source.value).label}: ${state.values.length} values.`);
          if (!start.value.trim() || !stop.value.trim()) throw new Error("Enter both Start and Stop for the range.");
          const lower = Number(start.value), upper = Number(stop.value);
          if (!Number.isFinite(lower) || !Number.isFinite(upper)) throw new Error("Start and Stop must be finite numbers.");
          generated = Array.from({length: size}, (_, index) => size === 1 ? lower : lower + (upper - lower) * index / (size - 1));
          // Keep every paired row when integer rounding produces repeated values.
          if (item.kind === "int") generated = generated.map(roundHalfEven);
          generated = generated.map((value) => castPreview(value, item));
        }
      } catch (error) {
        state.error = error.message;
      }
      row.querySelector("[data-couple-hint]").textContent = isRange
        ? `Match the ${state.values.length} existing values. The table is read-only.${item?.kind === "int" ? " Generated values are rounded to integers; paired rows are retained." : ""}`
        : "The first column comes from the selected parameter. Enter one coupled value in each row of the second column.";
      row.querySelector("[data-couple-error]").textContent = state.error;
      const signature = JSON.stringify([source.value, target.value, mode.value, state.values, isRange ? generated : null]);
      if (signature !== state.signature) {
        body.replaceChildren();
        state.values.forEach((value, index) => {
          const tr = document.createElement("tr");
          const left = document.createElement("td"); left.textContent = materialText(value);
          const right = document.createElement("td");
          if (isRange) {
            right.textContent = generated[index] === undefined ? "—" : String(generated[index]);
          } else if (item?.kind === "material") {
            right.append(materialFields(state.manual[index], item, (value) => { state.manual[index] = value; refresh(); }));
          } else {
            const field = document.createElement(item?.choices?.length ? "select" : "input");
            if (item?.choices?.length) {
              field.append(new Option("Choose a value", ""));
              item.choices.forEach((choice) => field.append(new Option(choice, choice)));
            } else {
              field.type = "text";
              if (["float", "int", "optional_float"].includes(item?.kind)) field.inputMode = "decimal";
            }
            field.setAttribute("aria-label", `${item?.label || "Value"}, row ${index + 1}`);
            field.value = state.manual[index] ?? "";
            field.addEventListener("input", () => { state.manual[index] = field.value; refresh(); });
            right.append(field);
          }
          tr.append(left, right); body.append(tr);
        });
        state.signature = signature;
      }
      body.querySelectorAll("input, select").forEach((field) => { field.disabled = !active; });
      state.generated = generated;
    }

    function read() {
      if (state.error) throw new Error(state.error);
      const rows = state.values.map((value, index) => {
        const manual = state.manual[index] ?? "";
        const raw = mode.value === "range" ? state.generated[index] : typeof manual === "object" ? manual : String(manual).trim();
        if (raw === "") throw new Error(`Enter ${descriptor(target.value).label} for row ${index + 1}.`);
        return {[source.value]: value, [target.value]: castPreview(raw, descriptor(target.value))};
      });
      return {name: `Coupled to ${descriptor(source.value).label}`, parameters: [source.value, target.value], rows};
    }

    coupledEditors.set(row, {update, read});
    [source, target, mode].forEach((select) => select.addEventListener("change", refresh));
    [start, stop, count].forEach((input) => input.addEventListener("input", refresh));
    row.querySelector("[data-remove]").addEventListener("click", () => { coupledEditors.delete(row); row.remove(); refresh(); });
  }

  function axisValues(row) {
    const item = descriptor(row.querySelector(".custom-grid-parameter").value);
    const mode = row.querySelector("[data-axis-mode]").value;
    if (mode === "list") {
      const values = splitValues(row.querySelector("[data-axis-list] textarea").value);
      if (!values.length) throw new Error(`${item.label} needs at least one value.`);
      if (item.kind === "material") { materialAxes.get(row).sync(); return materialAxes.get(row).read(); }
      return values.map((value) => castPreview(value, item));
    }
    const minimum = Number(row.querySelector("[data-minimum]").value);
    const maximum = Number(row.querySelector("[data-maximum]").value);
    const count = Number(row.querySelector("[data-count]").value);
    if (!Number.isInteger(count) || count < 1 || !Number.isFinite(minimum) || !Number.isFinite(maximum)) throw new Error("Every range needs finite bounds and a positive integer count.");
    const values = Array.from({length: count}, (_, index) => count === 1 ? minimum : minimum + ((maximum - minimum) * index) / (count - 1));
    const exact = item.kind === "int" ? [...new Set(values.map(roundHalfEven))] : values;
    row.querySelector("[data-axis-generated]").textContent = item.kind === "int" ? `Exact integer values: ${exact.join(", ")}${exact.length !== count ? ` (requested ${count}; reduced to ${exact.length})` : ""}` : `Generated ${exact.length} values.`;
    return exact.map((value) => castPreview(value, item));
  }

  function buildConfig() {
    const bySource = new Map();
    coupledEditors.forEach((editor) => {
      const group = editor.read();
      const existing = bySource.get(group.parameters[0]);
      if (existing) {
        existing.parameters.push(group.parameters[1]);
        existing.rows.forEach((values, index) => Object.assign(values, group.rows[index]));
      } else {
        bySource.set(group.parameters[0], group);
      }
    });
    const coupled_groups = [...bySource.values()];
    const sources = new Set(coupled_groups.map((group) => group.parameters[0]));
    const used = new Set();
    coupled_groups.forEach((group) => group.parameters.forEach((parameter) => {
      if (used.has(parameter)) throw new Error("A parameter can appear only once across coupled groups.");
      used.add(parameter);
    }));
    const independent = sourceAxes().filter((row) => !sources.has(row.querySelector(".custom-grid-parameter").value)).map((row) => {
      const parameter = row.querySelector(".custom-grid-parameter").value;
      if (used.has(parameter)) throw new Error("A parameter can appear only once across all axes and coupled groups."); used.add(parameter);
      const mode = row.querySelector("[data-axis-mode]").value;
      if (mode === "range") return {parameter, mode, minimum: Number(row.querySelector("[data-minimum]").value), maximum: Number(row.querySelector("[data-maximum]").value), count: Number(row.querySelector("[data-count]").value)};
      return {parameter, mode, values: axisValues(row)};
    });
    const display_order = [
      ...independent.map((axis) => ({kind: "independent", parameter: axis.parameter})),
      ...coupled_groups.map((group, index) => ({kind: "coupled_group", index, name: group.name})),
    ];
    return {schema_version: 1, independent, coupled_groups, display_order};
  }

  function gridSummary(config) {
    const axes = config.independent.map((axis) => axisValues(sourceAxes().find((row) => row.querySelector(".custom-grid-parameter").value === axis.parameter)).map((value) => ({[axis.parameter]: value})));
    config.coupled_groups.forEach((group) => axes.push(group.rows));
    const count = axes.reduce((total, axis) => total * axis.length, 1);
    const uniqueCount = axes.reduce((total, axis) => total * new Set(axis.map((item) => JSON.stringify(item))).size, 1);
    return {count, duplicateCount: count - uniqueCount, axes};
  }

  function refresh() {
    try {
      syncParameterOptions();
      sourceAxes().forEach((row) => materialAxes.get(row)?.sync());
      coupledEditors.forEach((editor) => editor.update());
      const config = buildConfig(); const summary = gridSummary(config);
      configField.value = JSON.stringify(config);
      countNode.textContent = `${summary.count.toLocaleString()} ${summary.count === 1 ? "case" : "cases"}`;
      const duplicateCount = summary.duplicateCount;
      const warnings = [];
      if (summary.count > threshold) warnings.push(`Large grid: ${summary.count.toLocaleString()} cases exceeds the ${threshold.toLocaleString()}-case warning threshold.`);
      if (duplicateCount) warnings.push(`${duplicateCount} resolved duplicate ${duplicateCount === 1 ? "case is" : "cases are"} retained.`);
      warningNode.textContent = warnings.join(" "); warningNode.classList.toggle("notice-error", warnings.length > 0);
      casePreview = {...summary, rendered: 0};
      previewNode.textContent = "";
      previewNode.scrollTop = 0;
      appendPreviewCases();
      root.dataset.customGridValid = "true";
    } catch (error) {
      casePreview = null;
      configField.value = "";
      previewNode.textContent = "Enter valid values to preview the grid.";
      countNode.textContent = "Grid configuration needs attention"; warningNode.textContent = error.message; warningNode.classList.add("notice-error"); root.dataset.customGridValid = "false";
    }
  }

  root.addEventListener("workflowchange", refresh);
  (initial.independent || []).forEach(makeAxis);
  (initial.coupled_groups || []).forEach(makeGroup);
  root.querySelector("[data-custom-add-axis]").addEventListener("click", () => { makeAxis(); refresh(); });
  root.querySelector("[data-custom-add-group]").addEventListener("click", () => { makeGroup(); refresh(); });
  root.closest("form")?.addEventListener("submit", (event) => {
    if (root.closest("form").querySelector("[data-run-workflow]")?.value !== "custom_grid") return;
    refresh();
    if (root.dataset.customGridValid !== "true") { event.preventDefault(); window.alert(warningNode.textContent); }
  });
  refresh();
}

document.addEventListener("DOMContentLoaded", () => {
  initMaterialDensitySync();
  document.querySelectorAll("[data-custom-grid-builder]").forEach(initCustomGridBuilder);
});

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Estimating";
  }
  const total = Math.max(0, Math.round(value));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) {
    return "Unavailable";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function initRunMonitor(container) {
  const statusUrl = container.dataset.runStatusUrl;
  const memoryUrl = container.dataset.memoryUrl;
  const plotImage = container.querySelector("[data-run-plot-image]");
  const stateNode = container.querySelector("[data-run-state]");
  const summaryNode = container.querySelector("[data-run-progress-summary]");
  const completedNode = container.querySelector("[data-run-completed]");
  const remainingNode = container.querySelector("[data-run-remaining]");
  const elapsedNode = container.querySelector("[data-run-elapsed]");
  const etaNode = container.querySelector("[data-run-eta]");
  const workersNode = container.querySelector("[data-run-workers]");
  const memoryNode = container.querySelector("[data-run-memory]");
  const errorNode = container.querySelector("[data-run-error]");
  const progressBar = container.querySelector("[data-run-progress-bar]");
  // Only some monitors render a "what is running right now" line; the shared
  // run-detail monitor does not, so every use below is null-guarded.
  const currentLabelNode = container.querySelector("[data-run-current-label]");
  // Run-detail keeps its action in the page header, outside the monitor. Stage
  // monitors have no abort link, so the page-level lookup is unambiguous.
  const abortButton = document.querySelector("[data-run-abort-action]");
  let latestPlotToken = "";
  let latestPlotUrl = plotImage.getAttribute("src") || "";
  let statusTimerId = null;
  let memoryTimerId = null;

  function preloadAndSwapPlot(url, token) {
    const candidate = new window.Image();
    candidate.addEventListener("load", () => {
      latestPlotToken = token;
      latestPlotUrl = url;
      plotImage.src = url;
      plotImage.classList.remove("is-hidden");
    });
    candidate.addEventListener("error", () => {
      if (latestPlotUrl) {
        plotImage.src = latestPlotUrl;
        plotImage.classList.remove("is-hidden");
      }
    });
    candidate.src = url;
  }

  function updateFromStatus(payload) {
    stateNode.textContent = payload.state;
    summaryNode.textContent = `${payload.completed_points} / ${payload.total_points}`;
    completedNode.textContent = String(payload.completed_points);
    remainingNode.textContent = String(payload.remaining_points);
    elapsedNode.textContent = formatSeconds(payload.elapsed_seconds);
    const terminal = ["completed", "failed", "aborted"].includes(payload.state);
    etaNode.textContent = terminal ? "Complete" : formatSeconds(payload.eta_seconds);
    const requested = payload.worker_mode === "manual" ? String(payload.requested_workers) : "auto";
    const resolved = payload.resolved_workers ? ` -> ${payload.resolved_workers}` : "";
    workersNode.textContent = `${requested}${resolved}`;
    const percent = payload.total_points > 0 ? (payload.completed_points / payload.total_points) * 100 : 0;
    progressBar.style.width = `${percent}%`;
    errorNode.textContent = payload.error_text || "";
    if (currentLabelNode) {
      currentLabelNode.textContent = payload.current_label || "";
    }
    if (abortButton) {
      if (payload.can_abort) {
        abortButton.removeAttribute("aria-disabled");
      } else {
        abortButton.setAttribute("aria-disabled", "true");
      }
    }
    if (payload.plot_url && payload.plot_token !== latestPlotToken) {
      preloadAndSwapPlot(payload.plot_url, payload.plot_token);
    }
    if (terminal) {
      if (statusTimerId !== null) {
        window.clearInterval(statusTimerId);
      }
      if (memoryTimerId !== null) {
        window.clearInterval(memoryTimerId);
      }
      // Results are rendered server-side, so a page loaded while the stage was
      // running has no way to show them. Monitors that opt in reload once; the
      // reloaded page renders the results instead of this monitor, so there is
      // no loop.
      if (container.hasAttribute("data-run-reload-on-finish")) {
        window.location.reload();
      }
    }
  }

  async function pollStatus() {
    const response = await fetch(statusUrl);
    const payload = await response.json();
    updateFromStatus(payload);
  }

  async function pollMemory() {
    const response = await fetch(memoryUrl);
    const payload = await response.json();
    if (!payload.ok) {
      memoryNode.textContent = "RAM unavailable";
      return;
    }
    memoryNode.textContent = `${formatBytes(payload.used_bytes)} used / ${formatBytes(payload.available_bytes)} free`;
  }

  pollStatus();
  pollMemory();
  statusTimerId = window.setInterval(pollStatus, 500);
  memoryTimerId = window.setInterval(pollMemory, 500);
}

function initSurveyCellCounter(form) {
  const dPoints = form.querySelector("[data-survey-d-points]");
  const blazePoints = form.querySelector("[data-survey-blaze-points]");
  const readout = form.querySelector("[data-survey-cell-readout]");
  if (!dPoints || !blazePoints || !readout) {
    return;
  }
  const threshold = Number(form.dataset.cellWarningThreshold || 200);

  function update() {
    const cells = Math.max(0, Number(dPoints.value) || 0) * Math.max(0, Number(blazePoints.value) || 0);
    if (cells > threshold) {
      readout.textContent =
        `This survey will run ${cells} theta searches — well above ${threshold}, so expect it to take a long time. ` +
        `You can abort it from the study page once it starts.`;
      readout.classList.add("notice-error");
    } else {
      readout.textContent = `This survey will run ${cells} theta searches.`;
      readout.classList.remove("notice-error");
    }
  }

  dPoints.addEventListener("input", update);
  blazePoints.addEventListener("input", update);
  update();
}

function initDesignPicker(form) {
  let options;
  try {
    options = JSON.parse(form.dataset.designOptions || "{}");
  } catch (error) {
    return;
  }
  const toggle = form.querySelector("[data-design-toggle]");
  const panel = form.querySelector("[data-design-manual]");
  const rows = form.querySelector("[data-design-rows]");
  const addButton = form.querySelector("[data-design-add]");
  if (!toggle || !panel || !rows || !addButton) {
    return;
  }
  const dValues = options.d_values || [];
  const blazeValues = options.blaze_values || [];
  const efficiency = options.efficiency || {};

  const formatValue = formatDesignValue;

  function announceSelection() {
    const pairs = Array.from(rows.querySelectorAll('input[name="design"]')).map(
      (input) => input.value,
    );
    document.dispatchEvent(
      new CustomEvent(DESIGN_SELECTION_CHANGED, {detail: {pairs}}),
    );
  }

  function buildSelect(values, initial) {
    const select = document.createElement("select");
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = formatValue(value);
      option.textContent = formatValue(value);
      select.appendChild(option);
    });
    if (initial !== undefined) {
      select.value = formatValue(initial);
    }
    return select;
  }

  function addRow(pair) {
    const best = pair || options.best || [dValues[0], blazeValues[0]];
    const row = document.createElement("div");
    row.className = "row";

    const dSelect = buildSelect(dValues, best[0]);
    const blazeSelect = buildSelect(blazeValues, best[1]);
    const hidden = document.createElement("input");
    hidden.type = "hidden";
    hidden.name = "design";

    const hint = document.createElement("span");
    hint.className = "subtle";

    function sync() {
      const pair = `${dSelect.value},${blazeSelect.value}`;
      hidden.value = pair;
      const value = efficiency[pair];
      hint.textContent = value === undefined ? "not surveyed" : `efficiency ${Number(value).toPrecision(4)}`;
      announceSelection();
    }

    dSelect.addEventListener("change", sync);
    blazeSelect.addEventListener("change", sync);
    sync();

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "button danger";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      row.remove();
      announceSelection();
    });

    const dLabel = document.createElement("label");
    dLabel.append("d-spacing, nm", dSelect);
    const blazeLabel = document.createElement("label");
    blazeLabel.append("Blaze, deg", blazeSelect);

    row.append(dLabel, blazeLabel, hint, hidden, remove);
    rows.appendChild(row);
    // Only now is the hidden input part of the list announceSelection reads.
    announceSelection();
  }

  toggle.addEventListener("click", () => {
    panel.classList.toggle("is-hidden");
    if (!panel.classList.contains("is-hidden") && rows.children.length === 0) {
      addRow();
    }
  });
  addButton.addEventListener("click", () => addRow());

  // Clicking a heatmap cell toggles that design: a second click on an already
  // chosen cell removes its row, so the map doubles as the selection list.
  document.addEventListener(SURVEY_CELL_PICKED, (event) => {
    const {d, blaze} = event.detail || {};
    if (d === undefined || blaze === undefined) {
      return;
    }
    const key = designPairKey(d, blaze);
    const existing = Array.from(rows.querySelectorAll('input[name="design"]')).find(
      (input) => input.value === key,
    );
    if (existing) {
      existing.closest(".row").remove();
      announceSelection();
      return;
    }
    panel.classList.remove("is-hidden");
    addRow([d, blaze]);
  });

  form.addEventListener("submit", (event) => {
    const submitter = event.submitter;
    if (!submitter || submitter.value !== "manual") {
      return;
    }
    const chosen = new Set(
      Array.from(rows.querySelectorAll('input[name="design"]')).map((input) => input.value),
    );
    if (chosen.size === 0) {
      event.preventDefault();
      window.alert("Add at least one (d, blaze) design to scan.");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-grating-type]").forEach((select) => {
    syncGratingSections(select);
    select.addEventListener("change", () => {
      syncGratingSections(select);
      defaultStackForGrating(select);
    });
  });

  document.querySelectorAll("[data-stack-type]").forEach((select) => {
    syncStackSections(select);
    select.addEventListener("change", () => syncStackSections(select));
  });

  document.querySelectorAll("[data-worker-mode]").forEach((select) => {
    syncWorkerFields(select);
    select.addEventListener("change", () => syncWorkerFields(select));
  });

  document.querySelectorAll("[data-run-workflow]").forEach((select) => {
    select.closest("form")?.querySelectorAll(
      '[name="study_energy_ev"], [name="study_grazing_angle_deg"]',
    ).forEach((field) => {
      field.addEventListener("input", () => { field.dataset.userEdited = "true"; });
    });
    syncRunWorkflow(select);
    select.addEventListener("change", () => syncRunWorkflow(select));
  });

  document.querySelectorAll('[name="study_fourier_min"], [name="study_fourier_max"], [name="study_fourier_points"]').forEach((field) => {
    field.addEventListener("input", () => syncFourierRangeWarning(field.closest("form")));
  });

  document.querySelectorAll("[data-confirm]").forEach((button) => {
    button.addEventListener("click", (event) => {
      const message = button.dataset.confirm;
      if (message && !window.confirm(message)) {
        event.preventDefault();
      }
    });
  });

  document.querySelectorAll("[data-run-abort-action]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (link.getAttribute("aria-disabled") === "true") event.preventDefault();
    });
  });

  const gratingPreviewForm = document.querySelector("[data-grating-preview-form]");
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target instanceof HTMLInputElement &&
        event.target.form && !["button", "submit", "reset"].includes(event.target.type)) {
      event.preventDefault();
    }
  });
  if (gratingPreviewForm) {
    const editor = gratingPreviewForm.querySelector("[data-layer-editor]");
    if (editor) initLayerEditor(editor, gratingPreviewForm);
    initGratingPreview(gratingPreviewForm);
  }

  const plotWorkspace = document.querySelector("[data-plot-workspace]");
  if (plotWorkspace) {
    initPlotWorkspace(plotWorkspace);
  }

  document.querySelectorAll("[data-saved-plot-figure]").forEach((container) => {
    initSavedPlotFigure(container);
  });

  document.querySelectorAll("[data-survey-cell-counter]").forEach((form) => {
    initSurveyCellCounter(form);
  });

  document.querySelectorAll("[data-design-picker]").forEach((form) => {
    initDesignPicker(form);
  });

  document.querySelectorAll("[data-survey-figures]").forEach((root) => {
    initSurveyFigures(root);
  });

  document.querySelectorAll("[data-energy-scan-figure-root]").forEach((root) => {
    initEnergyScanFigure(root);
  });

  document.querySelectorAll("[data-live-run-monitor]").forEach((runMonitor) => {
    initRunMonitor(runMonitor);
  });
});

function syncGradingFields(select) {
  document.querySelectorAll("[data-grading-field]").forEach((label) => {
    const modes = label.dataset.gradingField.split(" ");
    toggleSectionFields(label, modes.includes(select.value));
  });
}

function syncScanFields(select) {
  document.querySelectorAll("[data-scan-field]").forEach((label) => {
    toggleSectionFields(label, label.dataset.scanField.split(" ").includes(select.value));
  });
}

function initPlaneMirror(form) {
  const computeUrl = form.dataset.computeUrl;
  const figure = document.querySelector("[data-plane-mirror-figure]");
  const status = document.querySelector("[data-plane-mirror-status]");
  const loading = document.querySelector("[data-plane-mirror-loading]");
  const gradingMode = form.querySelector("[data-grading-mode]");
  const stackPreviewUrl = form.dataset.stackPreviewUrl;
  const stackImage = document.querySelector("[data-plane-mirror-stack-image]");
  const stackStatus = document.querySelector("[data-plane-mirror-stack-status]");
  let requestId = 0;
  let stackRequestId = 0;

  const updateStack = debounce(async () => {
    stackRequestId += 1;
    const currentRequest = stackRequestId;
    try {
      const response = await fetch(stackPreviewUrl, { method: "POST", body: new FormData(form) });
      const payload = await response.json();
      if (currentRequest !== stackRequestId) {
        return;
      }
      if (payload.ok) {
        stackImage.src = payload.image;
        stackImage.classList.remove("is-hidden");
        stackStatus.textContent = "Ready";
      } else {
        stackStatus.textContent = payload.error || "Schematic unavailable.";
      }
    } catch (error) {
      if (currentRequest === stackRequestId) {
        stackStatus.textContent = "Schematic request failed.";
      }
    }
  }, 400);

  const draw = (payload) => {
    if (!figure || !window.Plotly) {
      return;
    }
    const config = { responsive: true, displaylogo: false };
    if (payload.mode === "map") {
      window.Plotly.react(figure, [{
        type: "heatmap",
        x: payload.x,
        y: payload.y,
        z: payload.z,
        zmin: 0,
        zmax: 1,
        colorscale: "Viridis",
        colorbar: { title: { text: payload.graded ? "Reflectivity (footprint avg)" : "Reflectivity" } },
      }], {
        xaxis: { title: { text: "Photon energy (eV)" } },
        yaxis: { title: { text: "Grazing angle (deg)" } },
        margin: { t: 20 },
      }, config);
      return;
    }
    const traces = [{
      x: payload.x,
      y: payload.reflectivity,
      mode: "lines",
      name: payload.graded ? "Ungraded (centre)" : "Reflectivity",
    }];
    if (payload.graded) {
      traces.push({ x: payload.x, y: payload.graded, mode: "lines", name: "Footprint average" });
    }
    window.Plotly.react(figure, traces, {
      xaxis: { title: { text: payload.x_label } },
      yaxis: { title: { text: "Reflectivity" }, range: [0, 1] },
      margin: { t: 20 },
      legend: { orientation: "h", y: -0.3 },
    }, config);
  };

  const update = debounce(async () => {
    requestId += 1;
    const currentRequest = requestId;
    loading.textContent = "Computing";
    try {
      const response = await fetch(computeUrl, { method: "POST", body: new FormData(form) });
      const payload = await response.json();
      if (currentRequest !== requestId) {
        return;
      }
      if (payload.ok) {
        loading.textContent = "Ready";
        status.textContent = "Ready";
        draw(payload);
      } else {
        loading.textContent = "Error";
        status.textContent = payload.error || "Calculation failed.";
      }
    } catch (error) {
      if (currentRequest !== requestId) {
        return;
      }
      loading.textContent = "Error";
      status.textContent = "Calculation request failed.";
    }
  }, 300);

  const scanMode = form.querySelector("[data-scan-mode]");
  if (scanMode) {
    syncScanFields(scanMode);
    scanMode.addEventListener("change", () => syncScanFields(scanMode));
  }
  if (gradingMode) {
    syncGradingFields(gradingMode);
    gradingMode.addEventListener("change", () => syncGradingFields(gradingMode));
  }
  const stackFieldChanged = (event) => {
    // Scan, polarization and grading do not change the stack schematic.
    if (!event.target.closest("[data-scan-field], [data-grading-field]") &&
        !["scan_mode", "polarization", "grading_mode"].includes(event.target.name)) {
      updateStack();
    }
  };
  const editor = form.querySelector("[data-layer-editor]");
  if (editor) {
    initLayerEditor(editor, form);
  }
  // Delegated, so layer rows added or removed later are covered too.
  // Names and labels in the save controls never change the calculation.
  const ignoreSaveControls = (handler) => (event) => {
    if (!event.target.closest("[data-save-controls]")) {
      handler(event);
    }
  };
  ["input", "change"].forEach((type) => {
    form.addEventListener(type, ignoreSaveControls(update));
    form.addEventListener(type, ignoreSaveControls(stackFieldChanged));
  });
  update();
  updateStack();
}

function initRunExport() {
  const dialog = document.querySelector("[data-run-export-dialog]");
  const form = dialog?.querySelector("[data-run-export-form]");
  if (!dialog || !form) {
    return;
  }
  const status = form.querySelector("[data-run-export-status]");
  const filenameField = form.querySelector("[data-run-export-filename]");
  let exportUrl = "";
  let defaultFilename = "grax-run-export.zip";

  document.querySelectorAll("[data-run-export-open]").forEach((button) => {
    button.addEventListener("click", () => {
      exportUrl = button.dataset.runExportUrl || "";
      defaultFilename = button.dataset.runExportName || "grax-run-export.zip";
      filenameField.value = defaultFilename;
      status.textContent = "";
      dialog.showModal();
    });
  });

  dialog.querySelectorAll("[data-run-export-cancel]").forEach((button) => {
    button.addEventListener("click", () => dialog.close());
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formats = [...form.querySelectorAll("input[name='formats']:checked")].map(
      (input) => input.value,
    );
    if (!exportUrl || formats.length === 0) {
      status.textContent = "Select at least one file format.";
      return;
    }
    const submitButton = form.querySelector("button[type='submit']");
    submitButton.disabled = true;
    status.textContent = "Preparing archive…";
    try {
      const body = new URLSearchParams();
      formats.forEach((format) => body.append("formats", format));
      const response = await fetch(exportUrl, {method: "POST", body});
      if (!response.ok) {
        throw new Error("The selected run data are not available yet.");
      }
      const blob = await response.blob();
      const filename = filenameField.value.trim() || defaultFilename;
      if (typeof window.showSaveFilePicker === "function") {
        const fileHandle = await window.showSaveFilePicker({
          suggestedName: filename.endsWith(".zip") ? filename : `${filename}.zip`,
          types: [{description: "Grax run export", accept: {"application/zip": [".zip"]}}],
        });
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        status.textContent = "Export saved.";
      } else {
        const objectUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = objectUrl;
        link.download = filename.endsWith(".zip") ? filename : `${filename}.zip`;
        document.body.append(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(objectUrl);
        status.textContent = "Export downloaded. Choose the destination in your browser’s download settings.";
      }
    } catch (error) {
      status.textContent = error.name === "AbortError" ? "Export cancelled." : error.message;
    } finally {
      submitButton.disabled = false;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initRunExport();
  document.querySelectorAll("[data-plane-mirror-form]").forEach((form) => {
    initPlaneMirror(form);
  });
});
