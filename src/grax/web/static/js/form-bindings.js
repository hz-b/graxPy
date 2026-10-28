document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll('[name="cff"]').forEach((field) => {
    const warning = field.closest("fieldset")?.querySelector("[data-cff-warning]");
    const syncWarning = () => syncRunSweepButton(field.form);
    warning?.classList.add("is-hidden");
    syncWarning();
    field.addEventListener("input", syncWarning);
    field.addEventListener("change", syncWarning);
  });

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
    select.closest("form")?.querySelectorAll("[data-monochromator-default]").forEach((field) => {
      field.dataset.defaultValue = field.value;
      field.addEventListener("input", () => { if (field.dataset.applyingDefaults !== "true") field.dataset.userEdited = "true"; });
    });
    select.closest("form")?.querySelectorAll(
      '[name="study_energy_ev"], [name="study_grazing_angle_deg"]',
    ).forEach((field) => {
      field.addEventListener("input", () => { field.dataset.userEdited = "true"; });
    });
    syncRunWorkflow(select);
    syncRunSweepButton(select.closest("form"));
    select.addEventListener("change", () => {
      syncRunWorkflow(select);
      syncRunSweepButton(select.closest("form"));
    });
  });

  document.querySelectorAll('[name="study_fourier_min"], [name="study_fourier_max"], [name="study_fourier_points"]').forEach((field) => {
    field.addEventListener("input", () => syncFourierRangeWarning(field.closest("form")));
  });

  document.querySelectorAll("[data-dialog-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const dialog = document.querySelector(button.dataset.dialogTarget);
      if (dialog instanceof HTMLDialogElement) dialog.showModal();
    });
  });

  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => {
      const dialog = button.closest("dialog");
      if (dialog instanceof HTMLDialogElement) dialog.close();
    });
  });

  const gratingManager = document.querySelector("[data-grating-manager]");
  if (gratingManager) {
    const selectors = [...gratingManager.querySelectorAll("[data-grating-select]")];
    const selectAll = gratingManager.querySelector("[data-select-all-gratings]");
    const selectionCount = gratingManager.querySelector("[data-grating-selection-count]");
    const deleteSummary = gratingManager.querySelector("[data-grating-delete-summary]");
    const selectionActions = [...gratingManager.querySelectorAll("[data-selection-action]")];
    const updateSelection = () => {
      const selected = selectors.filter((input) => input.checked);
      const count = selected.length;
      if (selectionCount) selectionCount.textContent = `${count} selected`;
      selectionActions.forEach((button) => { button.disabled = count === 0; });
      if (selectAll) {
        selectAll.checked = count > 0 && count === selectors.length;
        selectAll.indeterminate = count > 0 && count < selectors.length;
      }
      if (deleteSummary) {
        const linkedRuns = selected.reduce((sum, input) => {
          const card = input.closest("[data-grating-card]");
          return sum + Number(card?.dataset.linkedRuns || 0);
        }, 0);
        const itemLabel = count === 1 ? "grating" : "gratings";
        const runLabel = linkedRuns === 1 ? "run" : "runs";
        deleteSummary.textContent = `You selected ${count} ${itemLabel}, with ${linkedRuns} linked ${runLabel}.`;
      }
    };
    selectors.forEach((input) => input.addEventListener("change", updateSelection));
    selectAll?.addEventListener("change", () => {
      selectors.forEach((input) => { input.checked = selectAll.checked; });
      updateSelection();
    });
    updateSelection();
  }

  const runGratingFilter = document.querySelector("[data-run-grating-filter]");
  if (runGratingFilter) {
    const runCards = [...document.querySelectorAll("[data-run-grating]")];
    const emptyNote = document.querySelector("[data-run-filter-empty]");
    runGratingFilter.addEventListener("change", () => {
      const wanted = runGratingFilter.value;
      let visible = 0;
      runCards.forEach((card) => {
        const show = !wanted || card.dataset.runGrating === wanted;
        card.hidden = !show;
        if (show) visible += 1;
        else card.querySelectorAll("input[name='selected_run_id']").forEach((box) => { box.checked = false; });
      });
      if (emptyNote) emptyNote.hidden = visible > 0;
    });
  }

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
