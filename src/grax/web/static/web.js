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
});
