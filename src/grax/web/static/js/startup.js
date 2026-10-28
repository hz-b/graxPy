function initNavigationDropdowns() {
  const dropdowns = [...document.querySelectorAll(".nav-dropdown")];
  if (dropdowns.length === 0) return;

  let lastPointerType = "";
  const closeOthers = (active) => {
    dropdowns.forEach((dropdown) => {
      if (dropdown === active) return;
      dropdown.open = false;
      if (dropdown.contains(document.activeElement)) dropdown.classList.add("is-dismissed");
    });
  };
  const openDropdown = (dropdown) => {
    dropdown.classList.remove("is-dismissed");
    closeOthers(dropdown);
    dropdown.open = true;
  };

  document.addEventListener("keydown", () => {
    lastPointerType = "keyboard";
  });
  document.addEventListener("pointerdown", (event) => {
    lastPointerType = event.pointerType || "mouse";
    if (!event.target.closest(".nav-dropdown")) {
      dropdowns.forEach((dropdown) => {
        dropdown.open = false;
        if (dropdown.contains(document.activeElement)) dropdown.classList.add("is-dismissed");
        else dropdown.classList.remove("is-dismissed");
      });
    }
  });

  dropdowns.forEach((dropdown) => {
    const summary = dropdown.querySelector("summary");
    dropdown.addEventListener("pointerenter", (event) => {
      if (event.pointerType !== "mouse") return;
      lastPointerType = "mouse";
      openDropdown(dropdown);
    });
    dropdown.addEventListener("pointerleave", (event) => {
      if (event.pointerType !== "mouse") return;
      dropdown.open = false;
      if (dropdown.contains(document.activeElement) && !document.activeElement.matches(":focus-visible")) {
        dropdown.classList.add("is-dismissed");
      }
    });
    dropdown.addEventListener("focusin", (event) => {
      closeOthers(dropdown);
      if (event.target.matches(":focus-visible")) dropdown.classList.remove("is-dismissed");
    });
    dropdown.addEventListener("focusout", () => {
      window.requestAnimationFrame(() => {
        if (!dropdown.contains(document.activeElement) && !dropdown.matches(":hover")) {
          dropdown.open = false;
          dropdown.classList.remove("is-dismissed");
        }
      });
    });
    dropdown.addEventListener("toggle", () => {
      if (dropdown.open) closeOthers(dropdown);
      else if (dropdown.contains(document.activeElement) && document.activeElement.matches(":focus-visible")) {
        dropdown.classList.add("is-dismissed");
      }
    });
    summary.addEventListener("click", (event) => {
      dropdown.classList.remove("is-dismissed");
      if (lastPointerType === "mouse") {
        event.preventDefault();
        openDropdown(dropdown);
      }
    });
    dropdown.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      dropdown.open = false;
      dropdown.classList.add("is-dismissed");
      summary.focus();
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initNavigationDropdowns();
  initDecimalSeparator();
  initDecimalControls();
  initStandardPolarization();
  if (document.documentElement.dataset.settingsSaved !== "true") persistUserSettings();
});

document.addEventListener("DOMContentLoaded", () => {
  initMaterialDensitySync();
  document.querySelectorAll("[data-scan-viewer]").forEach(initScanViewer);
  const runForm = document.querySelector("[data-run-form]");
  let draft = null;
  if (runForm) {
    const draftKey = `grax:run-draft:${runForm.dataset.gratingId}`;
    try { draft = JSON.parse(sessionStorage.getItem(draftKey) || "null"); } catch { sessionStorage.removeItem(draftKey); }
    if (draft && !Array.isArray(draft.fields)) { draft = null; sessionStorage.removeItem(draftKey); }
    if (draft) {
      const fields = [...runForm.querySelectorAll("[name]")];
      draft.fields.forEach((saved) => {
        const field = fields.find((item) => item.name === saved.name);
        if (!field) return;
        if (["checkbox", "radio"].includes(field.type)) field.checked = saved.checked;
        else field.value = saved.value;
        if (field.hasAttribute("data-monochromator-default")) field.dataset.userEdited = "true";
        if (saved.name.startsWith("study_")) field.dataset.userEdited = "true";
      });
      runForm.querySelectorAll("[data-custom-grid-builder]").forEach((root) => {
        const name = root.querySelector("[data-custom-grid-config]").name;
        const saved = draft.grids?.[name] || (name === "custom_grid_config" ? draft.customGrid : null);
        if (saved) root.querySelector("[data-custom-grid-default]").textContent = saved;
      });
    }
    runForm.querySelectorAll("[data-edit-run-grating]").forEach((link) => link.addEventListener("click", () => {
      const fields = [...runForm.querySelectorAll("[name]")]
        .filter((field) => !field.matches("[data-custom-grid-config]"))
        .map((field) => ({name: field.name, value: field.value, checked: field.checked}));
      sessionStorage.setItem(draftKey, JSON.stringify({
        fields,
        grids: Object.fromEntries([...runForm.querySelectorAll("[data-custom-grid-config]")].map((field) => [field.name, field.value])),
      }));
    }));
    runForm.addEventListener("submit", (event) => {
      if (!event.defaultPrevented) sessionStorage.removeItem(draftKey);
    });
  }
  initNamedRanges();
  document.querySelectorAll("[data-custom-grid-builder]").forEach(initCustomGridBuilder);
  if (runForm && draft) {
    sessionStorage.removeItem(`grax:run-draft:${runForm.dataset.gratingId}`);
  }
});
