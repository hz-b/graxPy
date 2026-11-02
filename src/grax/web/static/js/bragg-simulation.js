function initBraggPreview(form) {
  const image = document.querySelector("[data-bragg-preview]");
  const crossSection = document.querySelector("[data-bragg-cross-section]");
  const note = document.querySelector("[data-bragg-preview-status]");
  let requestId = 0;
  let timer = null;

  const update = async () => {
    requestId += 1;
    const current = requestId;
    try {
      const response = await fetch(form.dataset.previewUrl, { method: "POST", body: new FormData(form) });
      const payload = await response.json();
      if (current !== requestId) {
        return;
      }
      if (payload.ok) {
        image.src = payload.image;
        image.hidden = false;
        crossSection.hidden = !payload.cross_section;
        if (payload.cross_section) {
          crossSection.src = payload.cross_section;
        }
        note.textContent = "";
      } else {
        note.textContent = payload.error || "Preview unavailable.";
      }
    } catch (error) {
      if (current === requestId) {
        note.textContent = "Preview request failed.";
      }
    }
  };

  const schedule = () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(update, 350);
  };
  form.addEventListener("input", schedule);
  form.addEventListener("change", schedule);

  const editor = form.querySelector("[data-bragg-coating]");
  const coated = form.querySelector("[data-bragg-coated]");
  if (editor && coated) {
    initLayerEditor(editor, form);
    coated.addEventListener("change", () => { editor.hidden = !coated.checked; });
  }
}

function initBraggScan(form) {
  const status = document.querySelector("[data-bragg-status]");
  const runButton = form.querySelector("[data-bragg-run]");
  const scan = form.querySelector("[data-bragg-scan]");

  const solver = form.querySelector("[data-bragg-solver]");
  const amplitudeMode = form.querySelector("[data-bragg-amplitude-mode]");
  const energyMode = form.querySelector("[data-bragg-energy-mode]");
  // A field shows (and is sent) only when the scan type, solver and value modes all match it.
  const conditions = [
    ["braggScanField", () => scan.value],
    ["braggSolverField", () => solver.value],
    ["braggAmplitudeField", () => amplitudeMode.value],
    ["braggEnergyField", () => energyMode.value],
  ];
  const syncFields = () => {
    const option = scan.selectedOptions[0];
    document.querySelector("[data-bragg-scan-title]").textContent = option.textContent;
    document.querySelector("[data-bragg-scan-summary]").textContent = option.dataset.summary;
    const selector = conditions.map(([key]) => `[data-${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}]`).join(",");
    form.querySelectorAll(selector).forEach((field) => {
      const active = conditions.every(([key, current]) => (
        field.dataset[key] === undefined || field.dataset[key].split(" ").includes(current())
      ));
      field.hidden = !active;
      field.querySelectorAll("input, select").forEach((input) => { input.disabled = !active; });
    });
  };
  [scan, solver, amplitudeMode, energyMode].forEach((select) => select.addEventListener("change", syncFields));
  syncFields();

  // Suggested grazing-angle window for the photon energy; applied until the user edits the angles.
  const energy = form.querySelector("[data-bragg-energy]");
  // The photon energy, or for an energy scan the middle of its range (or of its list).
  const suggestionEnergy = () => {
    if (scan.value !== "energy_rocking_curve") {
      return energy.value;
    }
    const values = energyMode.value === "range"
      ? [form.elements.energy_start_ev.value, form.elements.energy_stop_ev.value]
      : form.elements.energies_ev.value.split(/[\s,]+/);
    const numbers = values.map(Number).filter((value) => value > 0);
    return numbers.length ? String((Math.min(...numbers) + Math.max(...numbers)) / 2) : energy.value;
  };
  const suggestion = form.querySelector("[data-bragg-theta-suggestion]");
  const useSuggestion = form.querySelector("[data-bragg-use-suggestion]");
  const thetaInputs = [...form.querySelectorAll("[data-bragg-theta]")];
  let window_ = null;
  let thetaEdited = false;
  let windowRequest = 0;
  let windowTimer = null;
  const applyWindow = () => {
    if (!window_ || window_.theta_center_deg === undefined) {
      return;
    }
    form.elements.theta_center_deg.value = window_.theta_center_deg;
    form.elements.theta_half_range_deg.value = window_.theta_half_range_deg;
    form.elements.theta_step_deg.value = window_.theta_step_deg;
    thetaEdited = false;
  };
  const fetchWindow = async () => {
    windowRequest += 1;
    const current = windowRequest;
    try {
      const url = `${form.dataset.thetaWindowUrl}?energy_ev=${encodeURIComponent(suggestionEnergy())}`;
      const payload = await (await fetch(url)).json();
      if (current !== windowRequest) {
        return;
      }
      if (!payload.ok) {
        suggestion.textContent = payload.error || "";
        window_ = null;
        return;
      }
      window_ = payload;
      const at = `${Math.round(payload.energy_ev)} eV`;
      if (payload.multilayer_peak_deg !== undefined && payload.multilayer_peak_deg !== null) {
        suggestion.textContent = `Multilayer Bragg peak at ${payload.multilayer_peak_deg}° for ${at}.`;
      } else if (payload.critical_angle_deg !== undefined) {
        suggestion.textContent = `No multilayer Bragg peak; critical angle ${payload.critical_angle_deg}° at ${at}.`;
      } else {
        suggestion.textContent = "";
      }
      if (!thetaEdited) {
        applyWindow();
      }
    } catch (error) {
      if (current === windowRequest) {
        suggestion.textContent = "The suggested window could not be computed.";
      }
    }
  };
  if (energy && suggestion) {
    thetaInputs.forEach((input) => input.addEventListener("input", () => { thetaEdited = true; }));
    const refetch = () => {
      window.clearTimeout(windowTimer);
      windowTimer = window.setTimeout(fetchWindow, 400);
    };
    [energy, form.elements.energy_start_ev, form.elements.energy_stop_ev, form.elements.energies_ev]
      .forEach((input) => input.addEventListener("input", refetch));
    [scan, energyMode].forEach((select) => select.addEventListener("change", refetch));
    useSuggestion.addEventListener("click", () => {
      if (window_) {
        applyWindow();
      } else {
        thetaEdited = false;
        fetchWindow();
      }
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    runButton.disabled = true;
    status.textContent = "Starting…";
    try {
      const response = await fetch(form.action, { method: "POST", body: new FormData(form) });
      const payload = await response.json();
      if (!payload.ok) {
        status.textContent = payload.error || "The run could not start.";
        runButton.disabled = false;
        return;
      }
      window.location.href = payload.url;
    } catch (error) {
      status.textContent = "The run request failed.";
      runButton.disabled = false;
    }
  });
}

const braggGratingForm = document.querySelector("[data-bragg-grating-form]");
if (braggGratingForm) {
  initBraggPreview(braggGratingForm);
}
const braggScanForm = document.querySelector("[data-bragg-form]");
if (braggScanForm) {
  initBraggScan(braggScanForm);
}
