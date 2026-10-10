(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  function svgElement(name, attributes) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function renderCurve(figure, curve) {
    const svg = figure.querySelector("svg");
    const traces = [...(curve.individuals || []), curve.before, curve].filter(Boolean);
    const xValues = traces.flatMap((trace) => trace.x);
    const zValues = traces.flatMap((trace) => trace.z);
    if (!xValues.length || !zValues.length) return;
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const zMin = Math.min(...zValues);
    const zMax = Math.max(...zValues);
    const xSpan = xMax - xMin || 1;
    const zSpan = zMax - zMin || 1;
    const px = (x) => 54 + ((x - xMin) / xSpan) * 516;
    const py = (z) => 162 - ((z - zMin) / zSpan) * 136;
    svg.setAttribute("viewBox", "0 0 600 200");
    svg.replaceChildren();
    svg.append(
      svgElement("line", {x1: 54, y1: 162, x2: 570, y2: 162, stroke: "#667078"}),
      svgElement("line", {x1: 54, y1: 26, x2: 54, y2: 162, stroke: "#667078"}),
    );
    if (curve.selection) {
      const [start, stop] = curve.selection;
      svg.append(svgElement("rect", {
        x: px(start), y: 26, width: Math.max(0, px(stop) - px(start)), height: 136,
        fill: "#f59e0b", opacity: 0.18,
      }));
    }
    traces.forEach((trace) => {
      const points = trace.x.map((x, point) => `${px(x).toFixed(2)},${py(trace.z[point]).toFixed(2)}`).join(" ");
      svg.append(svgElement("polyline", {
        points, fill: "none", stroke: trace === curve ? "#0f766e" : "#94a3b8",
        "stroke-width": 2, "vector-effect": "non-scaling-stroke",
        "stroke-dasharray": trace === curve ? "none" : "5 3",
      }));
    });
    if (curve.markers) {
      curve.markers.x.forEach((x, index) => {
        svg.append(svgElement("line", {
          x1: px(x), y1: 26, x2: px(x), y2: 162, stroke: "#b42318",
          "stroke-dasharray": "4 3", opacity: 0.65,
        }));
        svg.append(svgElement("circle", {
          cx: px(x), cy: py(curve.markers.z[index]), r: 3.5, fill: "#b42318",
        }));
      });
    }
    if (curve.individuals && curve.individuals.length) {
      const label = svgElement("text", {x: 375, y: 16, "font-size": 11, fill: "#64748b"});
      label.textContent = `${curve.individuals.length} periods + average`;
      svg.append(label);
    }
    if (curve.before) {
      const before = svgElement("text", {x: 410, y: 16, "font-size": 11, fill: "#64748b"});
      before.textContent = "before";
      const after = svgElement("text", {x: 500, y: 16, "font-size": 11, fill: "#0f766e"});
      after.textContent = "after";
      svg.append(before, after);
    }
    const labels = [
      [xMin.toPrecision(4), 54, 181, "start"],
      [xMax.toPrecision(4), 570, 181, "end"],
      [zMin.toPrecision(4), 48, 164, "end"],
      [zMax.toPrecision(4), 48, 29, "end"],
      ["x (nm)", 310, 196, "middle"],
    ];
    labels.forEach(([value, x, y, anchor]) => {
      const label = svgElement("text", {x, y, "text-anchor": anchor, "font-size": 11, fill: "#475569"});
      label.textContent = value;
      svg.append(label);
    });
    const zLabel = svgElement("text", {
      x: 12, y: 104, transform: "rotate(-90 12 104)", "font-size": 11, fill: "#475569",
      "text-anchor": "middle",
    });
    zLabel.textContent = "z (nm)";
    svg.append(zLabel);
  }

  function repeatPeriod(curve) {
    const span = curve.x[curve.x.length - 1] - curve.x[0];
    if (!(span > 0)) return curve;
    const repeated = {x: [], z: []};
    for (let period = 0; period < 3; period += 1) {
      curve.x.forEach((x, index) => {
        if (period && index === 0) return;
        repeated.x.push(x - curve.x[0] + period * span);
        repeated.z.push(curve.z[index]);
      });
    }
    return repeated;
  }

  function init() {
    const processForm = document.querySelector("[data-afm-process-form]");
    const gratingForm = document.querySelector("[data-grating-preview-form]");
    if (!processForm || !gratingForm) return;
    const fileInput = processForm.querySelector('[name="afm_file"]');
    const targetPeriod = processForm.querySelector("[data-afm-target-period]");
    const mainPeriodLabel = gratingForm.querySelector("[data-afm-main-period]");
    const unitsInput = processForm.querySelector('[name="afm_units"]');
    const unitsHint = processForm.querySelector("[data-afm-units-hint]");
    const periodInput = gratingForm.querySelector('[name="period_lpermm"]');
    const periodHint = processForm.querySelector("[data-afm-period-hint]");
    const previewMessage = processForm.querySelector("[data-afm-preview-message]");
    const result = processForm.querySelector("[data-afm-result]");
    const gratingError = document.querySelector("[data-grating-preview-error]");
    const gratingStatus = document.querySelector("[data-grating-preview-status]");
    const liveFigure = document.querySelector("[data-afm-live-preview]");
    const liveCaption = liveFigure?.querySelector("[data-afm-live-caption]");
    const gratingType = gratingForm.querySelector('[name="grating_type"]');
    const profilePath = gratingForm.querySelector("[data-afm-profile-path]");
    let previewRequest = 0;
    let previewTimer = null;
    let autoEstimateAllowed = true;
    let allowUnitSuggestion = true;
    let liveCurve = null;
    let settingsRevision = 0;
    let processingPromise = null;

    function updateLivePreview() {
      if (!liveFigure) return;
      const visible = gratingType.value === "afm" && !profilePath.value && !!liveCurve;
      liveFigure.classList.toggle("is-hidden", !visible);
      if (visible) {
        if (liveCaption) liveCaption.textContent = processForm.querySelector('[name="afm_average"]').checked
          ? "AFM grating — three repeats of the averaged period"
          : "AFM grating — three repeats of the selected period";
        renderCurve(liveFigure, repeatPeriod(liveCurve));
      }
    }

    function periodNm() {
      const density = Number(periodInput.value);
      return Number.isFinite(density) && density > 0 ? 1e6 / density : null;
    }

    function syncTargetPeriod() {
      const period = periodNm();
      if (targetPeriod && periodInput.value !== targetPeriod.value) targetPeriod.value = periodInput.value;
    }

    function afmFormData() {
      const data = new FormData(processForm);
      const period = periodNm();
      if (period != null) data.set("afm_period_nm", String(period));
      return data;
    }

    targetPeriod.addEventListener("input", () => {
      periodInput.value = targetPeriod.value;
      syncTargetPeriod();
      schedulePreview(false);
    });

    function clearProcessed() {
      const profilePath = gratingForm.querySelector("[data-afm-profile-path]");
      if (!profilePath.value) return;
      profilePath.value = "";
      gratingForm.querySelector("[data-afm-source-filename]").value = "";
      gratingForm.dispatchEvent(new Event("change", {bubbles: true}));
      result.textContent = "Settings changed. Process the AFM profile again before saving.";
      result.classList.remove("is-hidden");
    }

    function showTraces(traces) {
      processForm.querySelectorAll("[data-afm-plot]").forEach((figure) => {
        const curve = traces[figure.dataset.afmPlot];
        figure.classList.toggle("is-hidden", !curve);
        if (curve) renderCurve(figure, curve);
      });
      liveCurve = traces.rescaled || null;
      updateLivePreview();
    }

    async function preview() {
      const currentRequest = ++previewRequest;
      if (!fileInput.files.length) {
        showTraces({});
        previewMessage.classList.add("is-hidden");
        return;
      }
      previewMessage.textContent = "Updating AFM previews…";
      previewMessage.classList.remove("is-hidden");
      try {
        const response = await fetch(processForm.dataset.afmPreviewUrl, {
          method: "POST", body: afmFormData(),
        });
        const payload = await response.json();
        if (currentRequest !== previewRequest) return;
        if (!response.ok || !payload.ok) throw new Error(payload.error || "AFM preview failed.");
        showTraces(payload.traces || {});
        if (payload.suggested_units && unitsInput.value !== payload.suggested_units) {
          if (allowUnitSuggestion) {
            unitsInput.value = payload.suggested_units;
            periodInput.value = "";
            densityInput.value = "";
            autoEstimateAllowed = true;
            allowUnitSuggestion = false;
            unitsHint.textContent = "The scan values look like metres; selected m. Please confirm the file's coordinate units.";
            unitsHint.classList.remove("is-hidden");
            preview();
            return;
          }
          unitsHint.textContent = "The scan values look like metres. Check the selected coordinate units.";
          unitsHint.classList.remove("is-hidden");
        }
        previewMessage.textContent = payload.stage_message || "";
        previewMessage.classList.toggle("is-hidden", !payload.stage_message);
        if (autoEstimateAllowed && periodNm() == null) {
          autoEstimateAllowed = false;
          const estimate = Number(payload.estimated_period_nm);
          if (payload.estimated_period_nm != null && Number.isFinite(estimate) && estimate > 0) {
            const label = Number(estimate.toPrecision(5));
            if (estimate < 0.1 || estimate > 1e7) {
              periodHint.textContent = `Estimated spacing: ${label} nm. Check the coordinate units before using it; the value is outside the usual range.`;
            } else {
              periodHint.textContent = `The scan suggests ${label} nm; enter the equivalent lines/mm in the Profile section above if needed.`;
            }
            periodHint.classList.remove("is-hidden");
          } else {
            periodHint.textContent = "No reliable period estimate from this scan. Enter a value manually.";
            periodHint.classList.remove("is-hidden");
          }
        }
        // Once the averaged/selected period is valid, build the same real
        // grating preview used after saving. This includes the material stack.
        if (payload.traces?.rescaled && !profilePath.value && !processingPromise) {
          await processCurrentProfile();
        }
      } catch (error) {
        if (currentRequest !== previewRequest) return;
        showTraces({});
        previewMessage.textContent = error.message || "AFM preview failed.";
        previewMessage.classList.remove("is-hidden");
      }
    }

    function schedulePreview(immediate) {
      settingsRevision += 1;
      delete gratingForm.dataset.afmProcessingError;
      clearProcessed();
      liveCurve = null;
      updateLivePreview();
      if (previewTimer) window.clearTimeout(previewTimer);
      if (immediate) preview();
      else previewTimer = window.setTimeout(preview, 250);
    }

    processForm.addEventListener("change", (event) => {
      if (event.target === fileInput || event.target === unitsInput) {
        periodHint.textContent = "The AFM analysis uses the period from the Profile section below.";
        allowUnitSuggestion = event.target === fileInput;
        unitsHint.classList.add("is-hidden");
      }
      if (event.target === processForm.querySelector("[data-afm-profile-type]")) {
        gratingForm.querySelector("[data-afm-profile-type]").value = event.target.value;
      }
      schedulePreview(event.target === fileInput || event.target === unitsInput || event.target.type === "checkbox");
    });
    // File selection must show the raw scan even when no period analysis can
    // run yet. Keep this explicit so it does not depend on form bubbling.
    fileInput.addEventListener("change", () => {
      settingsRevision += 1;
      delete gratingForm.dataset.afmProcessingError;
      clearProcessed();
      if (previewTimer) window.clearTimeout(previewTimer);
      preview();
    });
    processForm.addEventListener("input", (event) => {
      if (event.target.type === "number") schedulePreview(false);
    });
    gratingForm.addEventListener("change", (event) => {
      updateLivePreview();
      if (event.target === gratingType && mainPeriodLabel) {
        mainPeriodLabel.classList.toggle("is-hidden", gratingType.value === "afm");
      }
      if (event.target === periodInput) {
        syncTargetPeriod();
        schedulePreview(false);
      }
    });
    gratingForm.addEventListener("input", (event) => {
      if (event.target === periodInput) {
        syncTargetPeriod();
        schedulePreview(false);
      }
    });
    syncTargetPeriod();
    if (mainPeriodLabel && gratingType.value === "afm") mainPeriodLabel.classList.add("is-hidden");

    async function processCurrentProfile() {
      if (!processForm.reportValidity()) {
        gratingForm.dataset.afmProcessingError = "Complete the AFM upload and period fields before saving.";
        if (gratingError) {
          gratingError.textContent = gratingForm.dataset.afmProcessingError;
          gratingError.classList.remove("is-hidden");
        }
        return false;
      }
      if (processingPromise) return processingPromise;
      const startedRevision = settingsRevision;
      result.classList.remove("is-hidden");
      result.textContent = "Processing AFM profile…";
      processingPromise = (async () => {
        try {
          const response = await fetch(processForm.action, {method: "POST", body: afmFormData()});
          const payload = await response.json();
          if (!response.ok || !payload.ok) throw new Error(payload.error || "AFM processing failed.");
          if (startedRevision !== settingsRevision) {
            throw new Error("AFM settings changed during processing. Try again with the current settings.");
          }
          gratingForm.querySelector("[data-afm-profile-path]").value = payload.profile_path;
          updateLivePreview();
          gratingForm.querySelector("[data-afm-source-filename]").value = payload.source_filename;
          gratingForm.querySelector("[data-afm-units]").value = payload.units;
          gratingForm.querySelector("[data-afm-profile-type]").value = payload.profile_type;
          gratingForm.querySelector("[data-afm-period-nm]").value = String(payload.period_nm);
          gratingForm.querySelector('[name="period_lpermm"]').value = String(Math.round(1e6 / payload.period_nm));
          gratingForm.querySelector("[name=grating_type]").value = "afm";
          gratingForm.querySelector("[name=grating_type]").dispatchEvent(new Event("change", {bubbles: true}));
          delete gratingForm.dataset.afmProcessingError;
          if (gratingError) gratingError.classList.add("is-hidden");
          result.textContent = `Processed ${payload.points} points. Ready to save.`;
          return true;
        } catch (error) {
          result.textContent = error.message || "AFM processing failed.";
          gratingForm.dataset.afmProcessingError = result.textContent;
          if (gratingError) {
            gratingError.textContent = result.textContent;
            gratingError.classList.remove("is-hidden");
          }
          if (gratingStatus) gratingStatus.textContent = "AFM processing failed";
          return false;
        }
      })();
      try {
        return await processingPromise;
      } finally {
        processingPromise = null;
      }
    }

    processForm.addEventListener("submit", async function (event) {
      event.preventDefault();
      await processCurrentProfile();
    });

    gratingForm.addEventListener("grax:afm-save-requested", async function () {
      if (await processCurrentProfile()) gratingForm.requestSubmit();
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
