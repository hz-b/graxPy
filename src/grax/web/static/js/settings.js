const DECIMAL_SEPARATOR_KEY = "grax:decimal-separator";
const STANDARD_POLARIZATION_KEY = "grax:standard-polarization";
function decimalSeparator() {
  const settings = document.documentElement.dataset;
  if (settings.settingsSaved === "true" && ["dot", "comma"].includes(settings.decimalSeparator)) {
    return settings.decimalSeparator;
  }
  try { return localStorage.getItem(DECIMAL_SEPARATOR_KEY) === "comma" ? "comma" : "dot"; }
  catch { return "dot"; }
}
function parseDecimal(value) { return Number(String(value).trim().replace(",", ".")); }
function canonicalDecimal(value) { return String(value).trim().replace(",", "."); }
function displayDecimal(value) { return decimalSeparator() === "comma" ? String(value).replace(".", ",") : String(value).replace(",", "."); }
function applyDecimalLanguage() { document.documentElement.lang = decimalSeparator() === "comma" ? "de-DE" : "en"; }
const DECIMAL_COMPLETE = /^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?$/;
const nativeInputValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
function decimalFieldMessage(field) {
  const raw = field.value.trim();
  if (!raw) return field.required ? "Enter a number." : "";
  if (!DECIMAL_COMPLETE.test(raw)) return "Enter a valid decimal number.";
  const value = parseDecimal(raw);
  if (!Number.isFinite(value)) return "Enter a finite number.";
  if (field.min !== "" && value < Number(field.min)) return `Enter a value of at least ${displayDecimal(field.min)}.`;
  if (field.max !== "" && value > Number(field.max)) return `Enter a value of at most ${displayDecimal(field.max)}.`;
  return "";
}
function decimalFieldComplete(field) { return !decimalFieldMessage(field); }
function decimalFieldEligible(field) {
  if (field.type !== "number") return false;
  const step = field.getAttribute("step") || "1";
  return step === "any" || !Number.isInteger(Number(step));
}
function decimalStep(field, direction) {
  if (field.disabled || field.matches(":disabled")) return;
  const configuredStep = field.dataset.decimalArrowStep
    ?? (field.dataset.arrowStep || field.getAttribute("step") || "1");
  const step = configuredStep === "any" ? 1 : Number(configuredStep);
  if (!Number.isFinite(step) || step <= 0) return;
  const isDecimalField = field.matches("[data-decimal-field]");
  const hasValidValue = isDecimalField
    ? decimalFieldComplete(field)
    : Number.isFinite(field.valueAsNumber);
  const current = hasValidValue && field.value.trim() ? parseDecimal(field.value) : 0;
  const places = Math.min(12, Math.max((String(step).split(".")[1] || "").length,
    (String(current).split(".")[1] || "").length));
  let next = Number((current + direction * step).toFixed(places));
  if (field.min !== "") next = Math.max(next, Number(field.min));
  if (field.max !== "") next = Math.min(next, Number(field.max));
  field.value = String(next);
  if (isDecimalField) field.dataset.decimalCommitted = String(next);
  field.setCustomValidity("");
  field.dispatchEvent(new Event("input", {bubbles: true}));
  field.dispatchEvent(new Event("change", {bubbles: true}));
}
function initDecimalField(field) {
  if (!decimalFieldEligible(field)) return;
  field.dataset.decimalField = "";
  field.dataset.decimalArrowStep = field.dataset.arrowStep || field.getAttribute("step") || "1";
  const initial = field.value;
  field.type = "text";
  field.inputMode = "decimal";
  Object.defineProperty(field, "value", {
    configurable: true,
    get() { return nativeInputValue.get.call(this); },
    set(value) {
      const text = value == null || value === "" ? "" : displayDecimal(value);
      nativeInputValue.set.call(this, text);
      if (!text || DECIMAL_COMPLETE.test(text)) this.dataset.decimalCommitted = canonicalDecimal(text);
    },
  });
  field.value = initial;
  field.dataset.decimalCommitted = initial;
  const wrapper = document.createElement("span");
  wrapper.className = "decimal-field-wrap";
  field.before(wrapper);
  wrapper.append(field);
  const controls = document.createElement("span");
  controls.className = "decimal-field-controls";
  for (const [direction, label] of [[1, "Increase"], [-1, "Decrease"]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = direction > 0 ? "▲" : "▼";
    button.setAttribute("aria-label", `${label} ${field.name || "value"}`);
    button.addEventListener("click", () => decimalStep(field, direction));
    controls.append(button);
  }
  wrapper.append(controls);
}
function initDecimalFields(root = document) {
  if (root instanceof HTMLInputElement) initDecimalField(root);
  root.querySelectorAll?.('input[type="number"]').forEach(initDecimalField);
}
function initDecimalControls() {
  initDecimalFields();
  document.addEventListener("input", (event) => {
    const field = event.target;
    if (!field.matches?.("[data-decimal-field]")) return;
    const message = decimalFieldMessage(field);
    field.setCustomValidity("");
    if (!message) field.dataset.decimalCommitted = canonicalDecimal(field.value);
    else event.stopPropagation();
  }, true);
  document.addEventListener("change", (event) => {
    const field = event.target;
    if (!field.matches?.("[data-decimal-field]")) return;
    const message = decimalFieldMessage(field);
    field.setCustomValidity(message);
    if (message) event.stopPropagation();
    else field.value = field.value;
  }, true);
  document.addEventListener("keydown", (event) => {
    const field = event.target;
    if (!field.matches?.("[data-decimal-field]")) return;
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      decimalStep(field, event.key === "ArrowUp" ? 1 : -1);
    }
  }, true);
  document.addEventListener("wheel", (event) => {
    const focused = document.activeElement;
    const field = focused?.matches?.('[data-decimal-field], input[type="number"]')
      ? focused
      : focused?.closest?.(".decimal-field-wrap")?.querySelector('[data-decimal-field], input[type="number"]');
    if (!field || field.disabled || field.matches(":disabled") || event.deltaY === 0) return;
    event.preventDefault();
    decimalStep(field, event.deltaY < 0 ? 1 : -1);
  }, {capture: true, passive: false});
  document.addEventListener("submit", (event) => {
    for (const field of event.target.querySelectorAll?.("[data-decimal-field]:not(:disabled)") || []) {
      const message = decimalFieldMessage(field);
      field.setCustomValidity(message);
      if (message) { event.preventDefault(); field.reportValidity(); return; }
    }
  }, true);
  document.querySelectorAll("form").forEach((form) => form.addEventListener("formdata", (event) => {
    const groups = new Map();
    form.querySelectorAll("[data-decimal-field][name]:not(:disabled)").forEach((field) => {
      const values = groups.get(field.name) || [];
      values.push(!field.value.trim() ? "" : decimalFieldComplete(field)
        ? canonicalDecimal(field.value) : field.dataset.decimalCommitted || "");
      groups.set(field.name, values);
    });
    groups.forEach((values, name) => {
      event.formData.delete(name);
      values.forEach((value) => event.formData.append(name, value));
    });
  }));
  document.addEventListener("reset", (event) => {
    const form = event.target;
    window.setTimeout(() => form.querySelectorAll("[data-decimal-field]").forEach((field) => {
      field.value = field.value;
      field.dataset.decimalCommitted = field.value.trim() ? canonicalDecimal(field.value) : "";
      field.setCustomValidity("");
    }), 0);
  });
  new MutationObserver((records) => records.forEach((record) => record.addedNodes.forEach((node) => {
    if (node instanceof Element && !node.closest(".decimal-field-wrap")) initDecimalFields(node);
  }))).observe(document.body, {childList: true, subtree: true});
}
function standardPolarization() {
  const settings = document.documentElement.dataset;
  if (settings.settingsSaved === "true" && ["s", "p"].includes(settings.standardPolarization)) {
    return settings.standardPolarization;
  }
  try { return localStorage.getItem(STANDARD_POLARIZATION_KEY) === "s" ? "s" : "p"; }
  catch { return "p"; }
}
function polarizationLabel(value) {
  if (value === "s") return "S (TE)";
  if (value === "p") return "P (TM)";
  return value;
}
async function persistUserSettings() {
  const root = document.documentElement;
  const values = {
    decimal_separator: decimalSeparator(),
    standard_polarization: standardPolarization(),
  };
  try {
    const response = await fetch(document.body.dataset.settingsUrl, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(values),
    });
    if (!response.ok) throw new Error("Settings could not be saved.");
    root.dataset.decimalSeparator = values.decimal_separator;
    root.dataset.standardPolarization = values.standard_polarization;
    root.dataset.settingsSaved = "true";
  } catch { /* Keep the browser-local copy if the database folder is unavailable. */ }
}
function applyStandardPolarization() {
  const polarization = standardPolarization();
  document.querySelectorAll("select[data-standard-polarization-target]").forEach((select) => {
    if ([...select.options].some((option) => option.value === polarization)) select.value = polarization;
  });
}
// Keep the fine arrow step (such as 0.1) but accept typed values with more decimals.
document.addEventListener("input", (event) => {
  const field = event.target;
  if (!field.matches?.('input[type="number"]')) return;
  if (field.dataset.arrowStep === undefined) {
    const step = Number(field.getAttribute("step"));
    if (!(step > 0 && step < 1)) { field.dataset.arrowStep = ""; return; }
    field.dataset.arrowStep = String(step);
  }
  const step = Number(field.dataset.arrowStep);
  if (!step) return;
  const value = parseDecimal(field.value);
  const onStep = !Number.isFinite(value) || Math.abs(value / step - Math.round(value / step)) < 1e-6;
  field.step = onStep ? field.dataset.arrowStep : "any";
}, true);
function initDecimalSeparator() {
  applyDecimalLanguage();
  const picker = document.querySelector("select[data-decimal-separator]");
  if (!picker) return;
  picker.value = decimalSeparator();
  picker.addEventListener("change", () => {
    try { localStorage.setItem(DECIMAL_SEPARATOR_KEY, picker.value); } catch { /* Storage may be disabled. */ }
    document.documentElement.dataset.decimalSeparator = picker.value;
    applyDecimalLanguage();
    document.querySelectorAll("[data-decimal-field]").forEach((field) => { field.value = field.value; });
    persistUserSettings();
  });
}
function initStandardPolarization() {
  const picker = document.querySelector("select[data-standard-polarization]");
  if (picker) {
    picker.value = standardPolarization();
    picker.addEventListener("change", () => {
      try { localStorage.setItem(STANDARD_POLARIZATION_KEY, picker.value); } catch { /* Storage may be disabled. */ }
      document.documentElement.dataset.standardPolarization = picker.value;
      applyStandardPolarization();
      persistUserSettings();
    });
  }
  applyStandardPolarization();
}
