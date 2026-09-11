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

function syncStackSections(select) {
  const isMultilayer = select.value === "multilayer";
  document.querySelectorAll("[data-stack-controls]").forEach((section) => {
    toggleSectionFields(section, isMultilayer);
  });
  document.querySelectorAll("[data-single-layer-controls]").forEach((section) => {
    toggleSectionFields(section, !isMultilayer);
  });
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
  document.querySelectorAll("[data-workflow-fields]").forEach((section) => {
    const allowed = (section.dataset.workflowFields || "").split(/\s+/).filter(Boolean);
    toggleSectionFields(section, allowed.includes(workflow));
  });
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

function initGratingPreview(form) {
  const previewUrl = form.dataset.previewUrl;
  const image = document.querySelector("[data-grating-preview-image]");
  const status = document.querySelector("[data-grating-preview-status]");
  const loading = document.querySelector("[data-grating-preview-loading]");
  let requestId = 0;

  const updatePreview = debounce(async () => {
    requestId += 1;
    const currentRequest = requestId;
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
      loading.textContent = "Ready";
      if (payload.ok) {
        image.src = payload.preview_url;
        image.classList.remove("is-hidden");
        status.textContent = "Ready";
      } else {
        status.textContent = payload.error || "Preview unavailable.";
      }
    } catch (error) {
      if (currentRequest !== requestId) {
        return;
      }
      loading.textContent = "Error";
      status.textContent = "Preview request failed.";
    }
  }, 250);

  form.querySelectorAll("input, select, textarea").forEach((field) => {
    field.addEventListener("input", updatePreview);
    field.addEventListener("change", updatePreview);
  });
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

function surveyFigureLayout(meta, xTitle, yTitle, title) {
  const subtitle = [meta.coating_label, `${meta.target_energy_ev} eV`, `order ${meta.diffraction_order}`]
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
      loading.textContent = payload.ok ? "Ready" : "Idle";
      if (payload.ok) {
        renderPlotlyFigure(previewContainer, payload.figure_json);
        previewStatus.textContent = "Ready";
        saveButton.disabled = false;
        summarizeRuns(payload.selected_runs || []);
        syncSeriesControls(payload.series_controls || []);
      } else {
        previewStatus.textContent = payload.error || "Select runs and orders.";
        saveButton.disabled = true;
        seriesStyleList.replaceChildren();
      }
    } catch (error) {
      if (currentRequest !== requestId) {
        return;
      }
      loading.textContent = "Error";
      previewStatus.textContent = "Preview request failed.";
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

document.addEventListener("DOMContentLoaded", () => {
  initMaterialDensitySync();
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
  // Scoped to this monitor: a page can host several (one per workflow stage),
  // and a document-wide lookup would let every monitor drive the first button.
  const abortButton = container.querySelector("[data-run-abort-action]");
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
    etaNode.textContent = formatSeconds(payload.eta_seconds);
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
    if (["completed", "failed", "aborted"].includes(payload.state)) {
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
    select.addEventListener("change", () => syncGratingSections(select));
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
    syncRunWorkflow(select);
    select.addEventListener("change", () => syncRunWorkflow(select));
  });

  document.querySelectorAll("[data-confirm]").forEach((button) => {
    button.addEventListener("click", (event) => {
      const message = button.dataset.confirm;
      if (message && !window.confirm(message)) {
        event.preventDefault();
      }
    });
  });

  const gratingPreviewForm = document.querySelector("[data-grating-preview-form]");
  if (gratingPreviewForm) {
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

  document.querySelectorAll("[data-live-run-monitor]").forEach((runMonitor) => {
    initRunMonitor(runMonitor);
  });
});
