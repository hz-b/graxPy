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
    energy_sweep: {
      title: "Energy sweep",
      summary: "Scans energy at a fixed grazing incidence angle. Optional independent and coupled parameters create a scan for every combination.",
    },
    angle_sweep: {
      title: "Angle sweep",
      summary: "Scans a grazing-angle range at one photon energy.",
    },
    monochromator: {
      title: "Monochromator",
      summary: "Scans energy while adjusting the incidence and diffraction geometry for a constant CFF. Use it to model a grating operated as a monochromator.",
    },
    map: {
      title: "Legacy angle vs energy map",
      summary: "This saved scan preset uses the previous mirror map workflow. New runs can use Energy sweep, Angle sweep, or Monochromator.",
    },
    constant_deviation: {
      title: "Constant deviation (2θ)",
      summary: "Scans energy while adjusting incidence and diffraction angles to keep the total beam deviation fixed.",
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
  if (select.dataset.previousWorkflow && select.dataset.previousWorkflow !== workflow) {
    select.closest("form")?.querySelectorAll("[data-monochromator-default]").forEach((field) => {
      if (field.dataset.userEdited === "true") return;
      field.value = workflow === "monochromator" ? field.dataset.monochromatorDefault : field.dataset.defaultValue;
      field.dataset.applyingDefaults = "true";
      field.dispatchEvent(new Event("input", {bubbles: true}));
      delete field.dataset.applyingDefaults;
    });
  }
  select.dataset.previousWorkflow = workflow;
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
  select.closest("form")?.querySelectorAll("[data-custom-grid-builder]").forEach((root) => root.dispatchEvent(new Event("workflowchange")));
  syncFourierRangeWarning(select.closest("form"));
}

let linkedStepHelpId = 0;

function formatRoundedNumber(value) {
  return Number(value.toFixed(3)).toString();
}

function syncRunSweepButton(form) {
  const button = form?.querySelector('button[type="submit"]');
  if (!button) return;
  const workflow = form.querySelector("[data-run-workflow]")?.value;
  const gridInvalid = [...form.querySelectorAll("[data-custom-grid-builder]")].some((root) =>
    gridIsActive(root) && root.dataset.customGridValid !== "true",
  );
  const cffVaried = [...form.querySelectorAll("[data-custom-grid-builder]")].some((root) =>
    gridIsActive(root) && root.dataset.cffIsVaried === "true",
  );
  const directCffInvalid = workflow === "monochromator" && !cffVaried
    && parseDecimal(form.querySelector('[name="cff"]')?.value) === 1;
  form.querySelector("[data-cff-warning]")?.classList.toggle("is-hidden", !directCffInvalid);
  button.disabled = gridInvalid || directCffInvalid;
}

function initLinkedRange(minimum, maximum, count, fixedCount = false) {
  if (!minimum || !maximum || !count || count.dataset.linkedRange) return;
  count.dataset.linkedRange = "true";
  const label = document.createElement("label");
  const labelText = document.createElement("span");
  labelText.textContent = "Step size (fits endpoints)";
  for (const attr of ["data-scan-field", "data-grading-field", "data-workflow-fields"]) {
    if (count.closest("label")?.hasAttribute(attr)) label.setAttribute(attr, count.closest("label").getAttribute(attr));
  }
  const step = document.createElement("input");
  step.type = "number"; step.step = count.closest("[data-custom-axis]")?.querySelector(".custom-grid-parameter")?.value === "blaze_angle_deg" ? "0.1" : "any"; step.min = "0"; step.dataset.linkedStep = "";
  const help = document.createElement("span"); help.className = "roughness-help-wrap";
  const helpButton = document.createElement("button");
  helpButton.type = "button"; helpButton.className = "roughness-help"; helpButton.textContent = "?";
  helpButton.setAttribute("aria-label", "Help: step size");
  const tooltip = document.createElement("span");
  tooltip.className = "roughness-tip"; tooltip.setAttribute("role", "tooltip");
  tooltip.id = `linked-step-help-${++linkedStepHelpId}`;
  helpButton.setAttribute("aria-describedby", tooltip.id);
  help.append(helpButton, tooltip);
  const labelHead = document.createElement("span"); labelHead.className = "linked-step-label-head";
  labelHead.append(labelText, help);
  const hint = document.createElement("span"); hint.className = "subtle"; hint.setAttribute("aria-live", "polite");
  label.append(labelHead, step, hint); count.closest("label").after(label);
  let lastNonSingleCount = parseDecimal(count.value) > 1 ? parseDecimal(count.value) : 2;
  let collapsedRange = minimum.value !== "" && maximum.value !== "" && parseDecimal(minimum.value) === parseDecimal(maximum.value);
  const sync = (force = false) => {
    const lower = parseDecimal(minimum.value), upper = parseDecimal(maximum.value), size = parseDecimal(count.value);
    const actual = Number.isFinite(lower + upper) && Number.isInteger(size) && size >= 1
      ? (size === 1 ? 0 : Math.abs(upper - lower) / (size - 1)) : null;
    const exact = actual === null ? "" : String(actual);
    const display = actual === null ? "" : formatRoundedNumber(actual);
    if (force || document.activeElement !== step) { step.value = display; step.setCustomValidity(""); }
    tooltip.textContent = exact
      ? `Full calculated value: ${exact}. The visible value is rounded to at most three decimal places.`
      : "Enter a positive step size, or zero for a single-point range.";
    hint.textContent = step.value !== display && display ? `Actual spacing: ${display}` : "";
    step.disabled = count.disabled;
  };
  count._syncLinkedRange = sync;
  count.addEventListener("input", () => {
    if (!fixedCount) {
      const value = parseDecimal(count.value);
      if (Number.isInteger(value) && value > 1) lastNonSingleCount = value;
    }
    sync();
  });
  count.addEventListener("change", () => {
    if (!fixedCount && count.value !== "" && Number.isInteger(parseDecimal(count.value)) && parseDecimal(count.value) >= 1) {
      if (parseDecimal(count.value) === 1) maximum.value = minimum.value;
      else if (parseDecimal(minimum.value) === parseDecimal(maximum.value)) maximum.value = String(parseDecimal(minimum.value) + 1);
      collapsedRange = parseDecimal(minimum.value) === parseDecimal(maximum.value);
    }
    sync(true);
  });
  for (const field of [minimum, maximum]) field.addEventListener("change", () => {
    if (!fixedCount) {
      const equalBounds = minimum.value !== "" && maximum.value !== "" && parseDecimal(minimum.value) === parseDecimal(maximum.value);
      if (equalBounds) {
        if (!collapsedRange && parseDecimal(count.value) > 1) lastNonSingleCount = parseDecimal(count.value);
        count.value = "1";
        collapsedRange = true;
      } else {
        if (collapsedRange && parseDecimal(count.value) === 1) count.value = String(lastNonSingleCount);
        else if (parseDecimal(count.value) < 2) count.value = String(lastNonSingleCount);
        if (parseDecimal(count.value) > 1) lastNonSingleCount = parseDecimal(count.value);
        collapsedRange = false;
      }
    }
    sync();
  });
  step.addEventListener("input", () => {
    const requested = parseDecimal(step.value), lower = parseDecimal(minimum.value), upper = parseDecimal(maximum.value);
    step.setCustomValidity("");
    if (!Number.isFinite(requested) || requested < 0 || (requested === 0 && lower !== upper)) {
      step.setCustomValidity("Enter a positive step size, or zero for a single-point range."); return;
    }
    if (fixedCount) {
      maximum.value = String(lower + (upper < lower ? -1 : 1) * requested * Math.max(0, parseDecimal(count.value) - 1));
      maximum.dispatchEvent(new Event("input", {bubbles: true}));
    } else {
      const intervals = lower === upper ? 0 : Math.max(1, Math.round(Math.abs(upper - lower) / requested));
      count.value = String(Math.min(intervals + 1, parseDecimal(count.max) || Infinity));
      count.dispatchEvent(new Event("input", {bubbles: true}));
    }
    sync();
  });
  step.addEventListener("change", () => sync(true));
  sync();
}

function initNamedRanges() {
  document.querySelectorAll("form").forEach((form) => {
    const field = (name) => form.querySelector(`[name="${name}"]`);
    const mappings = [
      ["energy_start_ev", "energy_stop_ev", "energy_points"],
      ["grazing_angle_start_deg", "grazing_angle_stop_deg", "grazing_angle_points"],
      ["study_fourier_min", "study_fourier_max", "study_fourier_points"],
      ["study_x_resolution_min", "study_x_resolution_max", "study_x_resolution_points"],
      ["study_z_resolution_min", "study_z_resolution_max", "study_z_resolution_points"],
      ["energy_min", "energy_max", "energy_points"], ["angle_min", "angle_max", "angle_points"],
      ["energy_min", "energy_max", "map_energy_points"], ["angle_min", "angle_max", "map_angle_points"],
      ["d_min_nm", "d_max_nm", "d_points"], ["blaze_min_deg", "blaze_max_deg", "blaze_points"],
      ["energy_scan_min_ev", "energy_scan_max_ev", "energy_scan_points"],
    ];
    mappings.forEach(([lower, upper, size]) => initLinkedRange(field(lower), field(upper), field(size)));
    form.querySelectorAll('[name$="_scan_points"], [name="footprint_points"]').forEach((count) => {
      if (count.dataset.linkedRange) return;
      const width = field(count.name === "footprint_points" ? "footprint_length_mm" : count.name.replace("_points", "_half_width_deg"));
      if (!width) return;
      const minimum = document.createElement("input"), maximum = document.createElement("input");
      const update = () => { minimum.value = "0"; maximum.value = String(parseDecimal(width.value) * (count.name === "footprint_points" ? 1 : 2)); count._syncLinkedRange?.(); };
      update(); initLinkedRange(minimum, maximum, count); width.addEventListener("input", update); count.addEventListener("input", update);
    });
  });
}

function gridIsActive(root) {
  const mode = root.closest("form").querySelector("[data-run-workflow]").value;
  return root.hasAttribute("data-parameter-scans") ? ["energy_sweep", "angle_sweep", "monochromator", "constant_deviation", "fixed_angle"].includes(mode) : mode === "custom_grid";
}

function syncCustomGridAxisInputs(row) {
  const isRange = row.querySelector("[data-axis-mode]").value === "range";
  const active = gridIsActive(row.closest("[data-custom-grid-builder]"));
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
  const minimum = parseDecimal(form.querySelector('[name="study_fourier_min"]')?.value);
  const maximum = parseDecimal(form.querySelector('[name="study_fourier_max"]')?.value);
  const requested = parseDecimal(form.querySelector('[name="study_fourier_points"]')?.value);
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
  const rawTitle = parsed.layout?.title;
  const title = typeof rawTitle === "string" ? rawTitle : rawTitle?.text || "plot";
  const filename = title.replace(/<[^>]*>/g, "").trim().replace(/[^\p{L}\p{N}._-]+/gu, "_").replace(/^_+|_+$/g, "").slice(0, 80) || "plot";
  window.Plotly.react(container, parsed.data || [], parsed.layout || {}, {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: {format: "png", filename},
  });
}
