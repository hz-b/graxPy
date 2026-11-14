// Shows how many saved runs and gratings follow the current database standard.
function initDatabaseStatus() {
  const box = document.querySelector("[data-database-status]");
  if (!box) return;
  const message = box.querySelector("[data-database-message]");
  const bar = box.querySelector("[data-database-progress]");
  const check = async () => {
    let status;
    try {
      const response = await fetch(box.dataset.url);
      status = await response.json();
    } catch {
      return;
    }
    box.hidden = false;
    if (status.state === "current") {
      box.classList.add("is-current");
      bar.hidden = true;
      const note = status.failed ? ` ${status.failed} could not be updated.` : "";
      message.textContent = `Database updated to the latest version.${note}`;
      return;
    }
    box.classList.remove("is-current");
    bar.hidden = false;
    bar.max = Math.max(status.total, 1);
    bar.value = status.done;
    message.textContent = `Updating the database to the latest version: ${status.done} of ${status.total} items updated, ${status.missing} missing.`;
    window.setTimeout(check, 1000);
  };
  check();
}
document.addEventListener("DOMContentLoaded", initDatabaseStatus);
