function initSurveyCellCounter(form) {
  const dPoints = form.querySelector("[data-survey-d-points]");
  const blazePoints = form.querySelector("[data-survey-blaze-points]");
  const readout = form.querySelector("[data-survey-cell-readout]");
  if (!dPoints || !blazePoints || !readout) {
    return;
  }
  const threshold = Number(form.dataset.cellWarningThreshold || 200);

  function update() {
    const cells = Math.max(0, parseDecimal(dPoints.value) || 0) * Math.max(0, parseDecimal(blazePoints.value) || 0);
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

  // With "Choose manually" open, clicking a heatmap cell toggles that design: a
  // second click on an already chosen cell removes its row, so the map doubles as
  // the selection list. With it closed, a click only shows the cell's scans.
  document.addEventListener(SURVEY_CELL_PICKED, (event) => {
    const {d, blaze} = event.detail || {};
    if (d === undefined || blaze === undefined || panel.classList.contains("is-hidden")) {
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
