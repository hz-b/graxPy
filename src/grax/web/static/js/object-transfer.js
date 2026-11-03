function showTransferError(element, message) {
  if (!element) {
    return;
  }
  element.querySelector("p").textContent = message;
  element.hidden = !message;
}

async function transferFailureMessage(response) {
  try {
    const payload = await response.json();
    if (payload.error) {
      return payload.error;
    }
  } catch {
    // The server answered with something other than the expected JSON.
  }
  return `GRAx could not complete this request (error ${response.status}).`;
}

const TRANSFER_UNREACHABLE = "GRAx could not be reached. Check that it is still running and try again.";

function initObjectExportDialog() {
  const dialog = document.querySelector("[data-object-export-dialog]");
  const form = dialog?.querySelector("[data-object-export-form]");
  if (!dialog || !form) {
    return;
  }
  const status = form.querySelector("[data-object-export-status]");
  const progress = form.querySelector("[data-object-export-progress]");
  const list = form.querySelector("[data-object-export-runs]");
  const selectAll = form.querySelector("[data-object-export-all]");
  const total = form.querySelector("[data-object-export-total]");
  const title = form.querySelector("[data-object-export-name]");
  const choice = form.querySelector("[data-object-export-choice]");
  const kindSelect = form.querySelector("[data-object-export-kind]");
  const objectSelect = form.querySelector("[data-object-export-object]");
  const submit = form.querySelector("button[type='submit']");
  let exportUrl = "";
  let filename = "export.grax.zip";
  let catalog = [];

  const runBoxes = () => [...list.querySelectorAll("input[name='run_id']:not(:disabled)")];

  function updateTotal() {
    const boxes = runBoxes();
    const chosen = boxes.filter((box) => box.checked);
    const bytes = chosen.reduce((sum, box) => sum + Number(box.dataset.size || 0), 0);
    const megabytes = (bytes / (1024 * 1024)).toFixed(1);
    total.textContent = `${chosen.length} of ${boxes.length} runs · ${megabytes} MB`;
    selectAll.checked = boxes.length > 0 && chosen.length === boxes.length;
  }

  function renderRuns(runs) {
    list.replaceChildren();
    if (runs.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "This object has no runs. Only the object will be exported.";
      list.append(empty);
    }
    runs.forEach((run) => {
      const row = document.createElement("label");
      row.className = "transfer-run";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.name = "run_id";
      box.value = run.id;
      box.dataset.size = run.size_bytes;
      box.checked = !run.problem;
      box.disabled = Boolean(run.problem);
      box.addEventListener("change", updateTotal);
      const text = document.createElement("span");
      text.className = "transfer-run-name";
      text.textContent = run.name;
      const detail = document.createElement("span");
      detail.className = "subtle";
      detail.textContent = [run.workflow, run.status, run.created_at].filter(Boolean).join(" · ");
      text.append(detail);
      if (run.problem) {
        const problem = document.createElement("span");
        problem.className = "transfer-run-problem";
        problem.textContent = `Cannot be exported: ${run.problem}`;
        text.append(problem);
      }
      const size = document.createElement("span");
      size.className = "subtle";
      size.textContent = run.size_label;
      row.append(box, text, size);
      list.append(row);
    });
    updateTotal();
  }

  async function loadRuns(runsUrl, nextExportUrl) {
    exportUrl = "";
    submit.disabled = true;
    list.replaceChildren();
    total.textContent = "";
    showTransferError(status, "");
    progress.textContent = "Loading the runs…";
    try {
      const response = await fetch(runsUrl);
      progress.textContent = "";
      if (!response.ok) {
        showTransferError(status, await transferFailureMessage(response));
        return;
      }
      const payload = await response.json();
      title.textContent = payload.name;
      filename = payload.filename;
      exportUrl = nextExportUrl;
      submit.disabled = false;
      renderRuns(payload.runs);
    } catch {
      progress.textContent = "";
      showTransferError(status, TRANSFER_UNREACHABLE);
    }
  }

  function chooseKind() {
    const group = catalog.find((item) => item.kind === kindSelect.value);
    objectSelect.replaceChildren(...(group?.objects || []).map((object) => {
      const option = document.createElement("option");
      option.value = object.id;
      option.textContent = object.name;
      return option;
    }));
    chooseObject();
  }

  function chooseObject() {
    const group = catalog.find((item) => item.kind === kindSelect.value);
    const object = group?.objects.find((item) => item.id === objectSelect.value);
    if (object) {
      loadRuns(object.runs_url, object.export_url);
    } else {
      exportUrl = "";
      submit.disabled = true;
      title.textContent = "";
      list.replaceChildren();
      total.textContent = "";
      showTransferError(status, "");
    }
  }

  async function openChooser() {
    choice.hidden = false;
    title.textContent = "";
    submit.disabled = true;
    list.replaceChildren();
    total.textContent = "";
    showTransferError(status, "");
    try {
      const response = await fetch(dialog.dataset.objectsUrl);
      if (!response.ok) {
        showTransferError(status, await transferFailureMessage(response));
        return;
      }
      catalog = await response.json();
      kindSelect.replaceChildren(...catalog.map((group) => {
        const option = document.createElement("option");
        option.value = group.kind;
        option.textContent = `${group.label} (${group.objects.length})`;
        return option;
      }));
      const first = catalog.find((group) => group.objects.length > 0);
      if (first) {
        kindSelect.value = first.kind;
      }
      chooseKind();
    } catch {
      showTransferError(status, TRANSFER_UNREACHABLE);
    }
  }

  kindSelect.addEventListener("change", chooseKind);
  objectSelect.addEventListener("change", chooseObject);
  selectAll.addEventListener("change", () => {
    runBoxes().forEach((box) => {
      box.checked = selectAll.checked;
    });
    updateTotal();
  });

  document.querySelectorAll("[data-object-export-open]").forEach((button) => {
    button.addEventListener("click", () => {
      progress.textContent = "";
      dialog.showModal();
      if ("objectExportChoose" in button.dataset) {
        openChooser();
      } else {
        choice.hidden = true;
        loadRuns(button.dataset.runsUrl, button.dataset.exportUrl);
      }
    });
  });

  dialog.querySelectorAll("[data-object-export-cancel]").forEach((button) => {
    button.addEventListener("click", () => dialog.close());
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!exportUrl) {
      return;
    }
    showTransferError(status, "");
    submit.disabled = true;
    let fileHandle = null;
    try {
      // Browsers allow the save dialog only right after the click, so it comes before the request.
      if (typeof window.showSaveFilePicker === "function") {
        fileHandle = await window.showSaveFilePicker({
          suggestedName: filename,
          types: [{description: "GRAx package", accept: {"application/zip": [".zip"]}}],
        });
      }
      progress.textContent = "Preparing the file…";
      const body = new URLSearchParams();
      list.querySelectorAll("input[name='run_id']:checked").forEach((box) => body.append("run_id", box.value));
      const response = await fetch(exportUrl, {method: "POST", body});
      if (!response.ok) {
        await fileHandle?.remove?.();
        progress.textContent = "";
        showTransferError(status, await transferFailureMessage(response));
        return;
      }
      if (fileHandle && response.body) {
        await response.body.pipeTo(await fileHandle.createWritable());
        progress.textContent = "Export saved.";
      } else {
        const objectUrl = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = objectUrl;
        link.download = filename;
        document.body.append(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(objectUrl);
        progress.textContent = "Export downloaded.";
      }
    } catch (error) {
      progress.textContent = "";
      if (error.name !== "AbortError") {
        showTransferError(status, error instanceof TypeError ? TRANSFER_UNREACHABLE : error.message);
      }
    } finally {
      submit.disabled = false;
    }
  });
}

function initObjectImportUpload() {
  const form = document.querySelector("[data-object-import-form]");
  if (!form) {
    return;
  }
  const file = form.querySelector("[data-object-import-file]");
  const button = form.querySelector("[data-object-import-choose]");
  const status = document.querySelector("[data-object-import-status]");
  const label = button.textContent;
  button.addEventListener("click", () => file.click());
  file.addEventListener("change", async () => {
    if (!file.files.length) {
      return;
    }
    showTransferError(status, "");
    button.disabled = true;
    button.textContent = "Checking the file…";
    try {
      const response = await fetch(form.action, {method: "POST", body: new FormData(form)});
      if (response.ok) {
        window.location.assign((await response.json()).url);
        return;
      }
      showTransferError(status, await transferFailureMessage(response));
    } catch {
      showTransferError(status, TRANSFER_UNREACHABLE);
    }
    button.disabled = false;
    button.textContent = label;
    // The same file can be chosen again after the problem is fixed.
    file.value = "";
  });
}

function initObjectImportPreview() {
  const form = document.querySelector("[data-object-import-preview]");
  if (!form) {
    return;
  }
  const status = document.querySelector("[data-object-import-status]");
  const submit = form.querySelector("[data-object-import-submit]");
  const cancel = form.querySelector("[data-object-import-cancel]");
  const target = form.querySelector("[data-object-import-target]");
  const selectAll = form.querySelector("[data-object-import-all]");
  const boxes = [...form.querySelectorAll("[data-object-import-run]:not(:disabled)")];

  form.querySelectorAll("[data-object-import-mode]").forEach((radio) => {
    radio.addEventListener("change", () => {
      target.disabled = form.elements.mode.value !== "replace";
    });
  });
  selectAll?.addEventListener("change", () => {
    boxes.forEach((box) => {
      box.checked = selectAll.checked;
    });
  });
  boxes.forEach((box) => {
    box.addEventListener("change", () => {
      selectAll.checked = boxes.every((item) => item.checked);
    });
  });

  async function send(url, body) {
    showTransferError(status, "");
    submit.disabled = true;
    cancel.disabled = true;
    let stayOpen = true;
    try {
      const response = await fetch(url, {method: "POST", body});
      if (response.ok) {
        window.location.assign((await response.json()).url);
        return;
      }
      showTransferError(status, await transferFailureMessage(response));
      try {
        stayOpen = (await response.clone().json()).can_retry !== false;
      } catch {
        stayOpen = true;
      }
    } catch {
      showTransferError(status, TRANSFER_UNREACHABLE);
    }
    submit.disabled = !stayOpen;
    cancel.disabled = false;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    send(form.action, new URLSearchParams(new FormData(form)));
  });
  cancel.addEventListener("click", () => send(cancel.dataset.objectImportCancel));
}

document.addEventListener("DOMContentLoaded", () => {
  initObjectExportDialog();
  initObjectImportUpload();
  initObjectImportPreview();
});
