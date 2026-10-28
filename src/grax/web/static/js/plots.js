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
  const styleDrafts = new Map();
  const editingPlotNode = document.querySelector("[data-editing-plot]");
  const editingPlot = editingPlotNode ? JSON.parse(editingPlotNode.textContent) : null;
  if (editingPlot) {
    Object.entries(editingPlot.plot_config?.series_styles || {}).forEach(([key, style]) => {
      styleDrafts.set(key, {
        legend: style.legend_label || "", color: style.color,
        marker: style.marker_symbol, size: String(style.marker_size), line: style.line_style,
      });
    });
  }
  const energyTimesOrderOption = form.querySelector('select[name="x_axis_value"] option[value="energy_times_order"]');
  let requestId = 0;

  function syncOrderAxisOption() {
    const hasMirror = [...runList.querySelectorAll("[data-plot-run-item]")]
      .some((item) => item.dataset.runKind === "mirror");
    if (energyTimesOrderOption) {
      energyTimesOrderOption.hidden = hasMirror;
      energyTimesOrderOption.disabled = hasMirror;
      if (hasMirror && energyTimesOrderOption.selected) {
        form.querySelector('select[name="x_axis_value"]').value = "energy";
      }
    }
  }

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
      const kind = document.createElement("span");
      kind.textContent = run.is_plane_mirror ? "Plane-mirror reflectivity" : `Orders: ${run.orders.join(", ")}`;
      row.append(title, kind);
      selectedRunSummary.append(row);
    });
  }

  function attachStyleInputHandlers(item) {
    item.querySelectorAll("input, select").forEach((field) => {
      if (field.matches("[data-series-legend-label]")) {
        field.addEventListener("input", () => {
          const index = [...seriesStyleList.querySelectorAll("[data-series-style-item]")].indexOf(item);
          const name = field.value.trim();
          if (name && index >= 0 && previewContainer?.data?.[index] && window.Plotly) {
            window.Plotly.restyle(previewContainer, {name: [name]}, [index]);
          }
        });
      }
      field.addEventListener("input", refreshPreview);
      field.addEventListener("change", refreshPreview);
    });
  }

  function rememberStyles() {
    seriesStyleList.querySelectorAll("[data-series-style-item]").forEach((item) => {
      const key = item.dataset.seriesKey;
      styleDrafts.set(key, {
        legend: item.querySelector("[data-series-legend-label]").value,
        color: item.querySelector("[data-series-color]").value,
        marker: item.querySelector("[data-series-marker-symbol]").value,
        size: item.querySelector("[data-series-marker-size]").value,
        line: item.querySelector("[data-series-line-style]").value,
      });
    });
  }

  function syncSeriesControls(seriesControls) {
    const existing = new Map([...seriesStyleList.querySelectorAll("[data-series-style-item]")]
      .map((item) => [item.dataset.seriesKey, item]));
    rememberStyles();
    seriesControls.forEach((series) => {
      let item = existing.get(series.series_key);
      if (!item) {
        item = seriesTemplate.content.firstElementChild.cloneNode(true);
        item.dataset.seriesKey = series.series_key;
        const draft = styleDrafts.get(series.series_key);
        const legendLabelInput = item.querySelector("[data-series-legend-label]");
        legendLabelInput.name = `legend_label_${series.series_token}`;
        legendLabelInput.value = draft?.legend ?? series.legend_label;
        const colorInput = item.querySelector("[data-series-color]");
        colorInput.name = `color_${series.series_token}`;
        colorInput.value = draft?.color ?? series.color;
        const markerSelect = item.querySelector("[data-series-marker-symbol]");
        markerSelect.name = `marker_symbol_${series.series_token}`;
        markerSelect.value = draft?.marker ?? series.marker_symbol;
        const markerSizeInput = item.querySelector("[data-series-marker-size]");
        markerSizeInput.name = `marker_size_${series.series_token}`;
        markerSizeInput.value = draft?.size ?? String(series.marker_size);
        const lineStyleSelect = item.querySelector("[data-series-line-style]");
        lineStyleSelect.name = `line_style_${series.series_token}`;
        lineStyleSelect.value = draft?.line ?? series.line_style;
        attachStyleInputHandlers(item);
      }
      item.querySelector("[data-series-style-title]").textContent = series.label;
      if (!existing.has(series.series_key)) seriesStyleList.append(item);
    });
    existing.forEach((item, key) => {
      if (!seriesControls.some((series) => series.series_key === key)) item.remove();
    });
    if (seriesControls.some((series) => styleDrafts.has(series.series_key) && !existing.has(series.series_key))) {
      refreshPreview();
    }
  }

  const refreshPreview = debounce(async () => {
    requestId += 1;
    const currentRequest = requestId;
    setPreviewFeedback("rendering", "Updating the preview…");
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
        const hasSeries = (payload.series_controls || []).length > 0;
        setPreviewFeedback(hasSeries ? "ready" : "idle", hasSeries ? "Preview is up to date." : "No plots are available for this loop and selection.");
        saveButton.disabled = !hasSeries;
        summarizeRuns(payload.selected_runs || []);
        syncSeriesControls(payload.series_controls || []);
      } else {
        previewContainer.replaceChildren();
        selectedRunSummary.replaceChildren();
        setPreviewFeedback("error", payload.error || "Select runs and orders.");
        saveButton.disabled = true;
        rememberStyles();
        seriesStyleList.replaceChildren();
      }
    } catch (error) {
      if (currentRequest !== requestId) {
        return;
      }
      previewContainer.replaceChildren();
      selectedRunSummary.replaceChildren();
      setPreviewFeedback("error", "The preview could not be updated. Try again.");
      saveButton.disabled = true;
      rememberStyles();
      seriesStyleList.replaceChildren();
    }
  }, 250);

  function bindRunItem(item) {
    item.querySelectorAll("input").forEach((field) => {
      field.addEventListener("change", refreshPreview);
    });
    item.querySelector("[data-remove-run]").addEventListener("click", () => {
      item.remove();
      syncOrderAxisOption();
      refreshPreview();
    });
  }

  function bindLoopControls(item, options) {
    if (!options.length) return;
    const root = item.querySelector("[data-loop-controls]");
    const list = root.querySelector("[data-loop-list]");
    const add = root.querySelector("[data-add-loop]");
    root.hidden = false;

    function updateChoices() {
      const selected = new Set([...list.querySelectorAll("select")].map((field) => field.value).filter(Boolean));
      list.querySelectorAll("select").forEach((field) => {
        [...field.options].forEach((option) => {
          if (option.value) option.disabled = selected.has(option.value) && option.value !== field.value;
        });
      });
      add.disabled = selected.size === options.length;
    }

    add.addEventListener("click", () => {
      const row = document.createElement("div");
      row.className = "row";
      const label = document.createElement("label");
      label.textContent = "Loop parameter";
      const select = document.createElement("select");
      select.append(new Option("Choose a parameter", ""));
      options.forEach((option) => select.append(new Option(option.label, option.id)));
      select.addEventListener("change", () => {
        select.name = select.value ? `loops_${item.dataset.runId}` : "";
        updateChoices();
        refreshPreview();
      });
      label.append(select);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "button";
      remove.textContent = "Remove loop";
      remove.addEventListener("click", () => {
        row.remove();
        updateChoices();
        refreshPreview();
      });
      row.append(label, remove);
      list.append(row);
      updateChoices();
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
    item.dataset.runKind = option.dataset.runKind || "grating";
    item.querySelector("[data-run-title]").textContent = option.dataset.runName;
    item.querySelector("[data-run-id-field]").value = option.value;
    const orderGrid = item.querySelector("[data-order-grid]");
    const orders = (option.dataset.runOrders || "")
      .split(",")
      .filter((value) => value !== "")
      .map((value) => Number.parseInt(value, 10));
    if (item.dataset.runKind === "mirror") {
      const orderField = document.createElement("input");
      orderField.type = "hidden";
      orderField.name = `orders_${option.value}`;
      orderField.value = "0";
      orderGrid.append(orderField);
    } else {
      orders.forEach((order, index) => {
        const checkbox = createOrderCheckbox(option.value, order);
        checkbox.querySelector("input").checked = option.dataset.runDefaultOrder
          ? order === Number(option.dataset.runDefaultOrder) : index === 0;
        orderGrid.append(checkbox);
      });
    }
    const scans = JSON.parse(option.dataset.runScans || "[]");
    if (scans.length) {
      const scanGrid = item.querySelector("[data-scan-grid]");
      const marker = document.createElement("input"); marker.type = "hidden"; marker.name = `scan_selection_${option.value}`; marker.value = "1"; scanGrid.append(marker);
      scans.forEach((scan, index) => {
        const label = document.createElement("label"); label.className = "inline-check";
        const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.name = `scans_${option.value}`; checkbox.value = scan.id; checkbox.checked = index === 0;
        const text = document.createElement("span"); text.textContent = scan.label; label.append(checkbox, text); scanGrid.append(label);
      });
    }
    bindLoopControls(item, JSON.parse(option.dataset.runLoops || "[]"));
    runList.append(item);
    syncOrderAxisOption();
    bindRunItem(item);
    picker.value = "";
    refreshPreview();
  });

  form.querySelector('input[name="title"]')?.addEventListener("change", refreshPreview);
  form.querySelectorAll('select[name="x_axis_value"], select[name="x_axis_type"], select[name="y_axis_type"]').forEach((field) => {
    field.addEventListener("change", refreshPreview);
  });
  if (editingPlot) {
    (editingPlot.selected_runs || []).forEach((run) => {
      picker.value = run.id;
      picker.dispatchEvent(new Event("change"));
      const item = [...runList.querySelectorAll("[data-plot-run-item]")]
        .find((candidate) => candidate.dataset.runId === run.id);
      if (!item) return;
      item.querySelectorAll(`[name="orders_${run.id}"][type="checkbox"]`).forEach((field) => {
        field.checked = (run.orders || []).includes(Number(field.value));
      });
      const selectedScans = editingPlot.plot_config?.scan_selection?.[run.id] ?? run.scan_ids;
      if (selectedScans) item.querySelectorAll(`[name="scans_${run.id}"]`).forEach((field) => {
        field.checked = selectedScans.includes(field.value);
      });
      (editingPlot.plot_config?.loop_selection?.[run.id] || []).forEach((loop) => {
        item.querySelector("[data-add-loop]")?.click();
        const select = [...item.querySelectorAll("[data-loop-list] select")].at(-1);
        if (select) { select.value = loop; select.dispatchEvent(new Event("change")); }
      });
    });
    refreshPreview();
  }
}

function initSavedPlotFigure(container) {
  const figureJson = container.dataset.figureJson;
  if (!figureJson) {
    return;
  }
  renderPlotlyFigure(container, JSON.parse(figureJson));
}

function initScanViewer(root) {
  const sweep = JSON.parse(root.querySelector("[data-scan-definition]").textContent);
  const axisPicker = root.querySelector("[data-scan-axis]");
  const orderPicker = root.querySelector("[data-scan-order]");
  const filters = root.querySelector("[data-scan-filters]");
  const feedback = root.querySelector("[data-scan-feedback]");
  const plot = root.querySelector("[data-scan-plot]");
  let payload = null, requestId = 0;
  const choices = new Map();
  function draw() {
    if (!payload) return;
    const selected = payload.scans.filter((scan) => [...choices].every(([axis, value]) => scan.axis_indices[axis] === value));
    const colors = ["#184e8b", "#bcd233", "#6cabe9", "#cb5a2e", "#6f8292", "#81768e"];
    renderPlotlyFigure(plot, {data: selected.map((scan, index) => ({
      x: payload.sample_values || payload.energies, y: scan.efficiencies, name: scan.label, mode: "lines+markers", connectgaps: false,
      line: {color: colors[index % colors.length], dash: ["solid", "dot", "dash", "dashdot"][index % 4]}, marker: {size: 3},
    })), layout: {title: `Order ${orderPicker.value} — scans`, template: "plotly_white",
      xaxis: {title: payload.sample_axis?.label || "Photon Energy (eV)"}, yaxis: {title: "Diffraction efficiency"},
      legend: {orientation: "h", y: -0.25}, margin: {t: 70, b: 150}}});
    feedback.textContent = `${selected.length} curves. Failed or missing points are gaps.`;
  }
  function updateFilters() {
    const previous = new Map(choices); choices.clear(); filters.replaceChildren();
    sweep.axes.filter((axis) => axis.id !== axisPicker.value).forEach((axis) => {
      const label = document.createElement("label"); label.append(`Fix ${axis.label}`);
      const select = document.createElement("select");
      axis.rows.forEach((row, index) => select.append(new Option(Object.entries(row).map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`).join("; "), String(index))));
      select.value = String(previous.get(axis.id) || 0); choices.set(axis.id, parseDecimal(select.value));
      select.addEventListener("change", () => {choices.set(axis.id, parseDecimal(select.value)); draw();});
      label.append(select); filters.append(label);
    });
    draw();
  }
  async function load() {
    const id = ++requestId; feedback.textContent = "Loading energy curves…";
    try {
      const response = await fetch(`${root.dataset.seriesUrl}?order=${encodeURIComponent(orderPicker.value)}`);
      if (!response.ok) throw new Error("Unable to load energy scans.");
      const data = await response.json(); if (id !== requestId) return;
      payload = data; draw();
    } catch (error) { if (id === requestId) feedback.textContent = error.message; }
  }
  axisPicker.addEventListener("change", updateFilters); orderPicker.addEventListener("change", load);
  updateFilters(); load();
}
