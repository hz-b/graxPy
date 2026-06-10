function syncGratingSections(select) {
  const selected = select.value;
  document.querySelectorAll("[data-grating-section]").forEach((section) => {
    const isActive = section.dataset.gratingSection === selected;
    section.classList.toggle("is-hidden", !isActive);
    section.querySelectorAll("input, select, textarea").forEach((field) => {
      field.disabled = !isActive;
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-grating-type]").forEach((select) => {
    syncGratingSections(select);
    select.addEventListener("change", () => syncGratingSections(select));
  });

  document.querySelectorAll("[data-plot-run-toggle]").forEach((toggle) => {
    const fieldset = toggle.closest("fieldset");
    const orders = fieldset ? fieldset.querySelectorAll("[data-plot-run-orders] input") : [];
    const sync = () => {
      orders.forEach((field) => {
        field.disabled = !toggle.checked;
      });
      if (!toggle.checked) {
        orders.forEach((field) => {
          field.checked = false;
        });
      }
    };
    sync();
    toggle.addEventListener("change", sync);
  });

  document.querySelectorAll("[data-confirm]").forEach((button) => {
    button.addEventListener("click", (event) => {
      const message = button.dataset.confirm;
      if (message && !window.confirm(message)) {
        event.preventDefault();
      }
    });
  });
});
