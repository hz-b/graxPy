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
    etaNode.textContent = formatSeconds(payload.eta_seconds);
    const requested = payload.worker_mode === "manual" ? String(payload.requested_workers) : "auto";
    const resolved = payload.resolved_workers ? ` -> ${payload.resolved_workers}` : "";
    workersNode.textContent = `${requested}${resolved}`;
    const percent = payload.total_points > 0 ? (payload.completed_points / payload.total_points) * 100 : 0;
    progressBar.style.width = `${percent}%`;
    errorNode.textContent = payload.error_text || "";
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

  const runMonitor = document.querySelector("[data-live-run-monitor]");
  if (runMonitor) {
    initRunMonitor(runMonitor);
  }
});
