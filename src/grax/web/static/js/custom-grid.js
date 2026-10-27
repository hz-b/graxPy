function initCustomGridBuilder(root) {
  const catalogNode = root.querySelector("[data-custom-grid-catalog]");
  const defaultNode = root.querySelector("[data-custom-grid-default]");
  const configField = root.querySelector("[data-custom-grid-config]");
  const axesRoot = root.querySelector("[data-custom-axes]");
  const groupsRoot = root.querySelector("[data-custom-groups]");
  if (!catalogNode || !defaultNode || !configField || !axesRoot || !groupsRoot) return;

  let baseCatalog = JSON.parse(catalogNode.textContent || "[]");
  const scanCatalogs = root.querySelector("[data-scan-catalogs]");
  const catalogsByMode = scanCatalogs ? JSON.parse(scanCatalogs.textContent) : null;
  if (catalogsByMode) baseCatalog = catalogsByMode[root.closest("form").querySelector("[data-run-workflow]").value] || baseCatalog;
  const initial = JSON.parse(defaultNode.textContent || "{}");
  const countNode = root.querySelector("[data-custom-grid-count]");
  const warningNode = root.querySelector("[data-custom-grid-warning]");
  const previewNode = root.querySelector("[data-custom-grid-preview]");
  function sizePreview() {
    if (!root.hasAttribute("data-parameter-scans")) return;
    const panels = [...root.querySelectorAll(":scope > .custom-grid-section")];
    const summary = root.querySelector("[data-custom-grid-summary]");
    if (panels.length < 2 || !summary) return;
    const sideBySide = Math.abs(panels[0].getBoundingClientRect().top - summary.getBoundingClientRect().top) < 5;
    if (!sideBySide) { previewNode.style.maxHeight = "65vh"; return; }
    const panelHeight = Math.max(...panels.map((panel) => {
      const head = panel.querySelector(".section-head");
      const list = panel.querySelector(".list");
      return (head?.scrollHeight || 0) + (list?.scrollHeight || 0) + 70;
    }));
    const summaryChrome = summary.scrollHeight - previewNode.clientHeight;
    previewNode.style.maxHeight = `${Math.max(192, panelHeight - summaryChrome)}px`;
  }
  window.addEventListener("resize", () => requestAnimationFrame(sizePreview));
  const coupledEditors = new Map();
  const materialAxes = new WeakMap();
  let casePreview = null;
  let cffRangeAutoRemoved = false;
  let cffExplicitInvalid = false;

  function appendPreviewCases() {
    if (!casePreview) return;
    const end = Math.min(casePreview.rendered + 100, casePreview.count);
    for (let index = casePreview.rendered; index < end; index += 1) {
      let remainder = index;
      const selected = new Array(casePreview.axes.length);
      for (let axis = casePreview.axes.length - 1; axis >= 0; axis -= 1) {
        const values = casePreview.axes[axis];
        selected[axis] = values[remainder % values.length];
        remainder = Math.floor(remainder / values.length);
      }
      const item = Object.assign({}, ...selected);
      if (casePreview.rendered || index > 0) previewNode.append(document.createTextNode("\n"));
      previewNode.append(document.createTextNode(`${index + 1}. `));
      const entries = Object.entries(item);
      if (!entries.length) {
        previewNode.append(document.createTextNode("Default settings"));
        continue;
      }
      entries.forEach(([key, value], valueIndex) => {
        if (valueIndex) previewNode.append(document.createTextNode("; "));
        previewNode.append(document.createTextNode(`${descriptor(key)?.label || key}=`));
        if (typeof value === "number" && Number.isFinite(value)) {
          const rounded = document.createElement("span");
          rounded.textContent = displayDecimal(formatRoundedNumber(value));
          rounded.title = `Full value: ${String(value)}. The visible value is rounded to at most three decimal places.`;
          previewNode.append(rounded);
        } else {
          previewNode.append(document.createTextNode(materialText(value)));
        }
      });
    }
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

  function splitValues(text, numeric = false) {
    const separator = numeric && decimalSeparator() === "comma" ? /[\n;]/ : /[\n,;]/;
    return text.split(separator).map((value) => value.trim()).filter(Boolean);
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
      const number = parseDecimal(value);
      if (!Number.isInteger(number)) throw new Error(`${item.label} only accepts integers.`);
      if (item.minimum !== null && item.minimum !== undefined && number < item.minimum) throw new Error(`${item.label} must be at least ${item.minimum}.`);
      return number;
    }
    if (["float", "optional_float"].includes(item.kind)) {
      if (item.kind === "optional_float" && value === "") return null;
      const number = parseDecimal(value);
      if (!Number.isFinite(number)) throw new Error(`${item.label} needs numeric values.`);
      if (item.minimum !== null && item.minimum !== undefined && number < item.minimum) throw new Error(`${item.label} must be at least ${item.minimum}.`);
      return number;
    }
    if (item.kind === "material") {
      if (value === null || value === undefined) return null;
      const material = typeof value === "object" ? {...value} : {name: String(value)};
      material.name = String(material.name || "").trim();
      if (material.name.toLowerCase() === "none") {
        if (["density_g_cm3", "roughness_sigma_nm"].some((key) => material[key] !== null && material[key] !== undefined && material[key] !== "")) {
          throw new Error(`${item.label}: None cannot have density or roughness values.`);
        }
        return null;
      }
      if (!material.name) throw new Error(`${item.label} needs a material name.`);
      for (const key of ["density_g_cm3", "roughness_sigma_nm"]) {
        const raw = material[key];
        if (raw === "" || raw === null || raw === undefined) { material[key] = null; continue; }
        const number = parseDecimal(raw);
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
    if (value === null || value === undefined) return "None";
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
    textarea.value = (axis.values || []).map((value) => value && typeof value === "object"
      ? value.name : value == null ? "None" : value).join(", "); values.append(textarea);
    const valuesHint = document.createElement("small"); valuesHint.className = "muted"; values.append(valuesHint);
    const materials = document.createElement("div"); materials.dataset.axisMaterials = "";
    let materialSignature = "";
    let materialValues = (axis.values || []).map((value) => value && typeof value === "object"
      ? {...value} : {name: value == null ? "None" : String(value)});
    function syncMaterials() {
      const item = descriptor(parameter.value);
      const isMaterial = item?.kind === "material";
      textarea.placeholder = isMaterial ? "None, Au, Pt (None means no coating)" : item?.id === "polarization" ? "S (TE), P (TM); enter s, p" : decimalSeparator() === "comma" ? "0,2; 0,4; 0,6" : "100, 200, 300";
      valuesHint.textContent = isMaterial ? "Example: None, Au, Pt. None means no coating." : item?.id === "polarization" ? "Enter s for S (TE) and p for P (TM); for example s, p." : decimalSeparator() === "comma" && ["float", "optional_float"].includes(item?.kind) ? "Use semicolons or new lines between decimal-comma values." : "";
      materials.classList.toggle("is-hidden", !isMaterial);
      if (isMaterial) {
        const names = splitValues(textarea.value);
        const signature = JSON.stringify([parameter.value, names]);
        if (signature !== materialSignature) {
          materialValues = names.map((name, index) => materialValues[index]?.name === name ? materialValues[index] : {name});
          materials.replaceChildren();
          const hint = document.createElement("p"); hint.className = "subtle";
          hint.textContent = "Set density and interface roughness for each material. None has neither. Blank density uses the material default; blank roughness keeps the saved grating setting. A separate scan of either property overrides these values.";
          materials.append(hint);
          materialValues.forEach((value, index) => {
            const line = document.createElement("div"); line.className = "custom-grid-material-row";
            const name = document.createElement("strong"); name.textContent = value.name;
            line.append(name);
            if (String(value.name || "").trim().toLowerCase() !== "none") {
              line.append(materialFields(value, item, (updated) => { materialValues[index] = updated; refresh(); }, false));
            }
            materials.append(line);
          });
          materialSignature = signature;
        }
      }
      const active = gridIsActive(root);
      materials.querySelectorAll("input").forEach((field) => { field.disabled = !active || !isMaterial; });
    }
    materialAxes.set(row, {sync: syncMaterials, read: () => materialValues.map((value) => castPreview(value, descriptor(parameter.value)))});
    const range = document.createElement("div"); range.className = "custom-grid-range"; range.dataset.axisRange = "";
    [["Minimum", "minimum"], ["Maximum", "maximum"], ["Point count", "count"]].forEach(([label, key]) => {
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
      const numericStep = "any";
      range.querySelectorAll("[data-minimum], [data-maximum]").forEach((input) => { input.step = numericStep; });
      range.querySelector("[data-linked-step]")?.setAttribute("step", numericStep);
      if (!rangeAllowed) mode.value = "list";
      mode.querySelector('[value="range"]').disabled = !rangeAllowed;
      syncCustomGridAxisInputs(row);
      syncMaterials();
    }
    parameter.addEventListener("change", () => { syncMode(); refresh(); }); mode.addEventListener("change", () => { syncMode(); refresh(); });
    row.addEventListener("input", refresh); row.querySelector("[data-remove]").addEventListener("click", () => { row.remove(); refresh(); });
    axesRoot.append(row);
    initLinkedRange(range.querySelector("[data-minimum]"), range.querySelector("[data-maximum]"), range.querySelector("[data-count]"));
    syncMode();
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
    const savedInput = group.input_settings?.[target.value];
    if (savedInput) { mode.value = savedInput.mode; start.value = savedInput.start; stop.value = savedInput.stop; }
    count.readOnly = true;
    initLinkedRange(start, stop, count, true);
    const body = row.querySelector("tbody");
    const state = {manual: (group.rows || []).map((values) => {
      const value = values[group.parameters[1]];
      return value == null && descriptor(group.parameters[1])?.kind === "material" ? "None" : value ?? "";
    }), values: [], error: "", signature: "", source: "", target: target.value};
    groupsRoot.append(row);
    syncParameterOptions();
    source.value = group.parameters?.[0] || sourceAxes()[0]?.querySelector(".custom-grid-parameter").value || "";

    function update() {
      const active = gridIsActive(root);
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
        count.value = state.values.length;
        if (state.values.length === 1) stop.value = start.value;
        count._syncLinkedRange?.();
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
          const size = parseDecimal(count.value);
          if (!Number.isInteger(size) || size !== state.values.length) throw new Error(`Number of points must match ${descriptor(source.value).label}: ${state.values.length} values.`);
          if (!start.value.trim() || !stop.value.trim()) throw new Error("Enter both Start and Stop for the range.");
          const lower = parseDecimal(start.value), upper = parseDecimal(stop.value);
          if (!Number.isFinite(lower) || !Number.isFinite(upper)) throw new Error("Start and Stop must be finite numbers.");
          generated = Array.from({length: size}, (_, index) => size === 1 ? lower : lower + (upper - lower) * index / (size - 1));
          // Keep every paired row when integer rounding produces repeated values.
          if (item.kind === "int") generated = generated.map(roundHalfEven);
          generated = generated.map((value) => castPreview(value, item));
          if (item.id === "cff" && root.closest("form")?.querySelector("[data-run-workflow]")?.value === "monochromator") {
            const retained = generated.map((value, index) => ({value, index}))
              .filter(({value}) => Math.abs(Number(value) - 1) > 1e-12);
            if (retained.length !== generated.length) {
              cffRangeAutoRemoved = true;
              state.values = retained.map(({index}) => state.values[index]);
              generated = retained.map(({value}) => value);
              count.value = String(state.values.length);
              if (!generated.length) state.error = "The CFF linear range has no usable values after CFF = 1 is removed.";
            }
          }
        }
      } catch (error) {
        state.error = error.message;
      }
      row.querySelector("[data-couple-hint]").textContent = isRange
        ? `Match the ${state.values.length} existing values. The table is read-only.${item?.kind === "int" ? " Generated values are rounded to integers; paired rows are retained." : ""}${item?.id === "cff" && cffRangeAutoRemoved ? " CFF = 1 was automatically removed." : ""}`
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
              item.choices.forEach((choice) => field.append(new Option(
                item.id === "polarization" ? polarizationLabel(choice) : choice,
                choice,
              )));
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
      if (target.value === "cff" && root.closest("form")?.querySelector("[data-run-workflow]")?.value === "monochromator") {
        if (mode.value !== "range" && rows.some((item) => item.cff === 1)) {
          cffExplicitInvalid = true;
        }
      }
      return {name: `Coupled to ${descriptor(source.value).label}`, parameters: [source.value, target.value], rows,
        input_settings: {[target.value]: {mode: mode.value, start: start.value, stop: stop.value}}};
    }

    coupledEditors.set(row, {update, read});
    [source, target, mode].forEach((select) => select.addEventListener("change", refresh));
    [start, stop, count].forEach((input) => input.addEventListener("input", refresh));
    row.querySelector("[data-remove]").addEventListener("click", () => { coupledEditors.delete(row); row.remove(); refresh(); });
  }

  function axisValues(row) {
    const item = descriptor(row.querySelector(".custom-grid-parameter").value);
    const mode = row.querySelector("[data-axis-mode]").value;
    const monochromatorCff = item.id === "cff"
      && root.closest("form")?.querySelector("[data-run-workflow]")?.value === "monochromator";
    if (mode === "list") {
      const values = splitValues(row.querySelector("[data-axis-list] textarea").value, ["float", "optional_float", "int"].includes(item.kind));
      if (!values.length) throw new Error(`${item.label} needs at least one value.`);
      if (item.kind === "material") { materialAxes.get(row).sync(); return materialAxes.get(row).read(); }
      const castValues = values.map((value) => castPreview(value, item));
      if (monochromatorCff && castValues.some((value) => value === 1)) cffExplicitInvalid = true;
      row.dataset.cffRangeFiltered = "false";
      return castValues;
    }
    const minimum = parseDecimal(row.querySelector("[data-minimum]").value);
    const maximum = parseDecimal(row.querySelector("[data-maximum]").value);
    const count = parseDecimal(row.querySelector("[data-count]").value);
    if (!Number.isInteger(count) || count < 1 || !Number.isFinite(minimum) || !Number.isFinite(maximum)) throw new Error("Every range needs finite bounds and a positive integer count.");
    const values = Array.from({length: count}, (_, index) => count === 1 ? minimum : minimum + ((maximum - minimum) * index) / (count - 1));
    let exact = item.kind === "int" ? [...new Set(values.map(roundHalfEven))] : values;
    if (monochromatorCff) {
      const usable = exact.filter((value) => Math.abs(Number(value) - 1) > 1e-12);
      if (usable.length !== exact.length) {
        cffRangeAutoRemoved = true;
        row.dataset.cffRangeFiltered = "true";
        exact = usable;
      } else {
        row.dataset.cffRangeFiltered = "false";
      }
      if (!exact.length) throw new Error("The CFF linear range has no usable values after CFF = 1 is removed.");
    } else {
      row.dataset.cffRangeFiltered = "false";
    }
    const generated = row.querySelector("[data-axis-generated]");
    if (monochromatorCff && row.dataset.cffRangeFiltered === "true") {
      generated.textContent = `Generated ${exact.length} usable values. CFF = 1 was automatically removed.`;
      generated.classList.add("notice-error");
    } else {
      generated.textContent = item.kind === "int" ? `Exact integer values: ${exact.join(", ")}${exact.length !== count ? ` (requested ${count}; reduced to ${exact.length})` : ""}` : `Generated ${exact.length} values.`;
      generated.classList.remove("notice-error");
    }
    return exact.map((value) => castPreview(value, item));
  }

  function buildConfig() {
    const bySource = new Map();
    coupledEditors.forEach((editor) => {
      const group = editor.read();
      const existing = bySource.get(group.parameters[0]);
      if (existing) {
        existing.parameters.push(group.parameters[1]);
        Object.assign(existing.input_settings, group.input_settings);
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
      if (mode === "range" && parameter === "cff" && root.closest("form")?.querySelector("[data-run-workflow]")?.value === "monochromator") {
        return {parameter, mode: "list", values: axisValues(row)};
      }
      if (mode === "range") return {parameter, mode, minimum: parseDecimal(row.querySelector("[data-minimum]").value), maximum: parseDecimal(row.querySelector("[data-maximum]").value), count: parseDecimal(row.querySelector("[data-count]").value)};
      return {parameter, mode, values: axisValues(row)};
    });
    const display_order = [
      ...independent.map((axis) => ({kind: "independent", parameter: axis.parameter})),
      ...coupled_groups.map((group, index) => ({kind: "coupled_group", index, name: group.name})),
    ];
    const editor_axes = sourceAxes().map((row) => ({parameter: row.querySelector('.custom-grid-parameter').value,
      mode: row.querySelector('[data-axis-mode]').value, values: axisValues(row),
      minimum: parseDecimal(row.querySelector('[data-minimum]').value), maximum: parseDecimal(row.querySelector('[data-maximum]').value),
      count: parseDecimal(row.querySelector('[data-count]').value)}));
    return {schema_version: 1, independent, coupled_groups, display_order, editor_axes};
  }

  function gridSummary(config) {
    const axes = config.independent.map((axis) => axisValues(sourceAxes().find((row) => row.querySelector(".custom-grid-parameter").value === axis.parameter)).map((value) => ({[axis.parameter]: value})));
    config.coupled_groups.forEach((group) => axes.push(group.rows));
    const count = axes.reduce((total, axis) => total * axis.length, 1);
    const uniqueCount = axes.reduce((total, axis) => total * new Set(axis.map((item) => JSON.stringify(item))).size, 1);
    return {count, duplicateCount: count - uniqueCount, axes};
  }

  function refresh() {
    cffRangeAutoRemoved = false;
    cffExplicitInvalid = false;
    try {
      syncParameterOptions();
      sourceAxes().forEach((row) => materialAxes.get(row)?.sync());
      coupledEditors.forEach((editor) => editor.update());
      const config = buildConfig(); const summary = gridSummary(config);
      configField.value = JSON.stringify(config);
      const form = root.closest("form");
      const mode = form.querySelector("[data-run-workflow]").value;
      const cffVariationWarning = root.querySelector("[data-cff-variation-warning]");
      const hasCffVariation = config.independent.some((axis) => axis.parameter === "cff")
        || config.coupled_groups.some((group) => group.parameters.includes("cff"));
      root.dataset.cffIsVaried = String(hasCffVariation);
      if (cffRangeAutoRemoved) {
        cffVariationWarning.textContent = "CFF = 1 was included in the linear range, but has been automatically removed because it cannot be used.";
        cffVariationWarning.classList.remove("is-hidden");
      } else if (cffExplicitInvalid) {
        cffVariationWarning.textContent = "The explicit CFF values include 1, which cannot be used. Remove CFF = 1 from the values before starting the calculation.";
        cffVariationWarning.classList.remove("is-hidden");
      } else {
        cffVariationWarning?.classList.add("is-hidden");
      }
      const axisCount = mode === "angle_sweep"
        ? parseDecimal(form.querySelector('[name="grazing_angle_points"]')?.value || 1)
        : parseDecimal(form.querySelector('[name="energy_points"]')?.value || 1);
      const terms = [...config.independent.map((axis) => `${axisValues(sourceAxes().find((row) => row.querySelector('.custom-grid-parameter').value === axis.parameter)).length} ${descriptor(axis.parameter)?.label || axis.parameter}`),
        ...config.coupled_groups.map((group) => `${group.rows.length} ${group.name}`)];
      countNode.textContent = root.hasAttribute("data-parameter-scans")
        ? `${terms.join(" × ") || "1 combination"} × ${axisCount} ${mode === "angle_sweep" ? "angles" : "energies"} = ${(summary.count * axisCount).toLocaleString()} simulation points`
        : `${summary.count.toLocaleString()} ${summary.count === 1 ? "case" : "cases"}`;
      const duplicateCount = summary.duplicateCount;
      const warnings = [];
      if (duplicateCount) warnings.push(`${duplicateCount} resolved duplicate ${duplicateCount === 1 ? "case is" : "cases are"} retained.`);
      if (cffExplicitInvalid) warnings.push("CFF = 1 is not allowed. Remove it from the explicit values.");
      warningNode.textContent = warnings.join(" "); warningNode.classList.toggle("notice-error", warnings.length > 0);
      casePreview = {...summary, rendered: 0};
      previewNode.textContent = "";
      previewNode.scrollTop = 0;
      appendPreviewCases();
      requestAnimationFrame(sizePreview);
      root.dataset.customGridValid = String(!cffExplicitInvalid);
      syncRunSweepButton(form);
    } catch (error) {
      casePreview = null;
      configField.value = "";
      previewNode.textContent = "Enter valid values to preview the grid.";
      requestAnimationFrame(sizePreview);
      countNode.textContent = "Grid configuration needs attention"; warningNode.textContent = error.message; warningNode.classList.add("notice-error"); root.dataset.customGridValid = "false";
      const cffVariationWarning = root.querySelector("[data-cff-variation-warning]");
      if (cffRangeAutoRemoved) {
        cffVariationWarning.textContent = "CFF = 1 was included in the linear range, but has been automatically removed because it cannot be used.";
        cffVariationWarning.classList.remove("is-hidden");
      }
      syncRunSweepButton(root.closest("form"));
    }
  }

  root.addEventListener("workflowchange", () => {
    if (catalogsByMode) {
      const next = catalogsByMode[root.closest("form").querySelector("[data-run-workflow]").value];
      if (next && JSON.stringify(next) !== JSON.stringify(baseCatalog)) {
        baseCatalog = next;
        root.querySelectorAll(".custom-grid-parameter").forEach((select) => {
          const selected = select.value;
          const options = descriptorOptions(selected);
          select.replaceChildren(...options.children);
          if (!descriptor(selected)) select.append(new Option(`${selected} (unavailable in this mode)`, selected));
          select.value = selected;
        });
      }
    }
    refresh();
  });
  root.closest("form").querySelectorAll('[name="energy_points"], [name="energy_start_ev"], [name="energy_stop_ev"], [name="grazing_angle_points"], [name="grazing_angle_start_deg"], [name="grazing_angle_stop_deg"]').forEach((field) => field.addEventListener("input", refresh));
  (initial.editor_axes || initial.independent || []).forEach(makeAxis);
  (initial.coupled_groups || []).forEach((group) => {
    const source = group.parameters[0];
    if (!sourceAxes().some((axis) => axis.querySelector('.custom-grid-parameter').value === source)) {
      makeAxis({parameter: source, mode: 'list', values: group.rows.map((row) => row[source])});
    }
    group.parameters.slice(1).forEach((target) => makeGroup({...group, parameters: [source, target]}));
  });
  root.querySelector("[data-custom-add-axis]").addEventListener("click", () => { makeAxis(); refresh(); });
  root.querySelector("[data-custom-add-group]").addEventListener("click", () => { makeGroup(); refresh(); });
  root.closest("form")?.addEventListener("submit", (event) => {
    if (!gridIsActive(root)) return;
    refresh();
    if (root.dataset.customGridValid !== "true") { event.preventDefault(); window.alert(warningNode.textContent); }
  });
  refresh();
}
