function initRunExport() {
  const dialog = document.querySelector("[data-run-export-dialog]");
  const form = dialog?.querySelector("[data-run-export-form]");
  if (!dialog || !form) {
    return;
  }
  const status = form.querySelector("[data-run-export-status]");
  const filenameField = form.querySelector("[data-run-export-filename]");
  let exportUrl = "";
  let defaultFilename = "grax-run-export.zip";

  document.querySelectorAll("[data-run-export-open]").forEach((button) => {
    button.addEventListener("click", () => {
      exportUrl = button.dataset.runExportUrl || "";
      defaultFilename = button.dataset.runExportName || "grax-run-export.zip";
      filenameField.value = defaultFilename;
      status.textContent = "";
      dialog.showModal();
    });
  });

  dialog.querySelectorAll("[data-run-export-cancel]").forEach((button) => {
    button.addEventListener("click", () => dialog.close());
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formats = [...form.querySelectorAll("input[name='formats']:checked")].map(
      (input) => input.value,
    );
    if (!exportUrl || formats.length === 0) {
      status.textContent = "Select at least one file format.";
      return;
    }
    const submitButton = form.querySelector("button[type='submit']");
    submitButton.disabled = true;
    status.textContent = "Preparing archive…";
    try {
      const body = new URLSearchParams();
      formats.forEach((format) => body.append("formats", format));
      const response = await fetch(exportUrl, {method: "POST", body});
      if (!response.ok) {
        throw new Error("The selected run data are not available yet.");
      }
      const blob = await response.blob();
      const filename = filenameField.value.trim() || defaultFilename;
      if (typeof window.showSaveFilePicker === "function") {
        const fileHandle = await window.showSaveFilePicker({
          suggestedName: filename.endsWith(".zip") ? filename : `${filename}.zip`,
          types: [{description: "Grax run export", accept: {"application/zip": [".zip"]}}],
        });
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        status.textContent = "Export saved.";
      } else {
        const objectUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = objectUrl;
        link.download = filename.endsWith(".zip") ? filename : `${filename}.zip`;
        document.body.append(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(objectUrl);
        status.textContent = "Export downloaded. Choose the destination in your browser’s download settings.";
      }
    } catch (error) {
      status.textContent = error.name === "AbortError" ? "Export cancelled." : error.message;
    } finally {
      submitButton.disabled = false;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initRunExport();
  document.querySelectorAll("[data-plane-mirror-form]").forEach((form) => {
    initPlaneMirror(form);
  });
  document.querySelectorAll("[data-plane-mirror-scan-form]").forEach(initPlaneMirrorScanForm);
});
