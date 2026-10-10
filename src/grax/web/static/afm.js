(function () {
  "use strict";
  function init() {
    const processForm = document.querySelector("[data-afm-process-form]");
    const gratingForm = document.querySelector("[data-grating-preview-form]");
    if (!processForm || !gratingForm) return;
    const result = processForm.querySelector("[data-afm-result]");
    processForm.addEventListener("submit", async function (event) {
      event.preventDefault();
      result.classList.remove("is-hidden");
      result.textContent = "Processing AFM profile…";
      try {
        const response = await fetch(processForm.action, {method: "POST", body: new FormData(processForm)});
        const payload = await response.json();
        if (!response.ok || !payload.ok) throw new Error(payload.error || "AFM processing failed.");
        gratingForm.querySelector("[data-afm-profile-path]").value = payload.profile_path;
        gratingForm.querySelector("[data-afm-source-filename]").value = payload.source_filename;
        gratingForm.querySelector("[name=grating_type]").value = "afm";
        result.textContent = `Processed ${payload.points} points. Select AFM profile and save the grating.`;
      } catch (error) { result.textContent = error.message; }
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
