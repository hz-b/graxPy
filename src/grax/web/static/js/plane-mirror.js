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
  form.querySelectorAll("[data-mirror-variation-axes], [data-mirror-coupled-variations]").forEach((group) => {
    const rows = group.querySelector("[data-variation-rows]");
    const template = group.querySelector("[data-variation-template]");
    group.querySelector("[data-variation-add]")?.addEventListener("click", () => {
      rows.append(template.content.cloneNode(true));
    });
    group.addEventListener("click", (event) => {
      const remove = event.target.closest("[data-variation-remove]");
      if (remove && rows.querySelectorAll("[data-variation-row]").length > 1) {
        remove.closest("[data-variation-row]").remove();
      }
    });
  });
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
    if (!figure || !status || !loading || !computeUrl) {
      return;
    }
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
    scanMode.addEventListener("change", () => {
      syncScanFields(scanMode);
      const nameField = form.querySelector("[data-plane-mirror-run-name]");
      if (nameField && nameField.dataset.autoName !== "false") {
        nameField.value = `${nameField.dataset.runPrefix} · ${scanMode.value} scan`;
      }
    });
    const nameField = form.querySelector("[data-plane-mirror-run-name]");
    if (nameField) {
      const initialName = nameField.value;
      nameField.addEventListener("input", () => {
        nameField.dataset.autoName = String(nameField.value === initialName);
      });
    }
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

function initPlaneMirrorScanForm(form) {
  // The preview panel sits next to the form, not inside it.
  const figure = document.querySelector("[data-plane-mirror-figure]");
  const status = document.querySelector("[data-plane-mirror-preview-status]");
  const workflow = form.querySelector("[data-run-workflow]");
  const nameField = form.querySelector("[data-plane-mirror-run-name]");
  let requestId = 0;
  form.querySelectorAll("[data-mirror-variation-axes], [data-mirror-coupled-variations]").forEach((group) => {
    const rows = group.querySelector("[data-variation-rows]");
    const template = group.querySelector("[data-variation-template]");
    group.querySelector("[data-variation-add]")?.addEventListener("click", () => rows.append(template.content.cloneNode(true)));
    group.addEventListener("click", (event) => {
      const remove = event.target.closest("[data-variation-remove]");
      if (remove && rows.querySelectorAll("[data-variation-row]").length > 1) remove.closest("[data-variation-row]").remove();
    });
  });
  if (workflow && nameField) {
    const presetName = nameField.value;
    nameField.addEventListener("input", () => { nameField.dataset.autoName = String(nameField.value === presetName); });
    workflow.addEventListener("change", () => {
      if (nameField.dataset.autoName !== "false") nameField.value = `${nameField.dataset.runPrefix} · ${workflow.value} scan`;
      updatePreview();
    });
  }
  const draw = (payload) => {
    if (!figure || !window.Plotly) return;
    const traces = [{
      x: payload.x,
      y: payload.reflectivity,
      mode: "lines",
      name: "Reflectivity",
    }];
    if (payload.graded) traces.push({x: payload.x, y: payload.graded, mode: "lines", name: "Footprint average"});
    window.Plotly.react(figure, traces, {
      xaxis: {title: {text: payload.x_label}},
      yaxis: {title: {text: "Reflectivity"}, range: [0, 1]},
      margin: {t: 20},
      legend: {orientation: "h", y: -0.3},
    }, {responsive: true, displaylogo: false});
  };
  const updatePreview = debounce(async () => {
    requestId += 1;
    const current = requestId;
    status.textContent = "Updating preview…";
    try {
      const response = await fetch(form.dataset.previewUrl, {method: "POST", body: new FormData(form)});
      const payload = await response.json();
      if (current !== requestId) return;
      if (!response.ok || !payload.ok) throw new Error(payload.error || "Preview calculation failed.");
      draw(payload);
      if (payload.mode === "monochromator" && payload.grazing_angles_deg?.length) {
        const values = payload.grazing_angles_deg;
        status.textContent = `Preview ready · grazing angle ${formatRoundedNumber(values[0])}°–${formatRoundedNumber(values.at(-1))}°`;
      } else {
        status.textContent = "Preview ready.";
      }
    } catch (error) {
      if (current === requestId) status.textContent = error.message || "Preview calculation failed.";
    }
  }, 350);
  form.addEventListener("input", updatePreview);
  form.addEventListener("change", updatePreview);
  updatePreview();
}
