function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Estimating";
  }
  const total = Math.max(0, Math.round(value));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) {
    return "Unavailable";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function initRunMonitor(container) {
  const statusUrl = container.dataset.runStatusUrl;
  const memoryUrl = container.dataset.memoryUrl;
  const plotImage = container.querySelector("[data-run-plot-image]");
  const stateNode = container.querySelector("[data-run-state]");
  const summaryNode = container.querySelector("[data-run-progress-summary]");
  const completedNode = container.querySelector("[data-run-completed]");
  const remainingNode = container.querySelector("[data-run-remaining]");
  const elapsedNode = container.querySelector("[data-run-elapsed]");
  const etaNode = container.querySelector("[data-run-eta]");
  const workersNode = container.querySelector("[data-run-workers]");
  const memoryNode = container.querySelector("[data-run-memory]");
  const errorNode = container.querySelector("[data-run-error]");
  const progressBar = container.querySelector("[data-run-progress-bar]");
  // Only some monitors render a "what is running right now" line; the shared
  // run-detail monitor does not, so every use below is null-guarded.
  const currentLabelNode = container.querySelector("[data-run-current-label]");
  // Run-detail keeps its action in the page header, outside the monitor. Stage
  // monitors have no abort link, so the page-level lookup is unambiguous.
  const abortButton = document.querySelector("[data-run-abort-action]");
  let latestPlotToken = "";
  let latestPlotUrl = plotImage?.getAttribute("src") || "";
  let statusTimerId = null;
  let memoryTimerId = null;

  function preloadAndSwapPlot(url, token) {
    const candidate = new window.Image();
    candidate.addEventListener("load", () => {
      latestPlotToken = token;
      latestPlotUrl = url;
      if (plotImage) {
        plotImage.src = url;
        plotImage.classList.remove("is-hidden");
      }
    });
    candidate.addEventListener("error", () => {
      if (latestPlotUrl) {
        if (plotImage) {
          plotImage.src = latestPlotUrl;
          plotImage.classList.remove("is-hidden");
        }
      }
    });
    candidate.src = url;
  }

  function updateFromStatus(payload) {
    stateNode.textContent = payload.state;
    summaryNode.textContent = `${payload.completed_points} / ${payload.total_points}`;
    completedNode.textContent = String(payload.completed_points);
    remainingNode.textContent = String(payload.remaining_points);
    elapsedNode.textContent = formatSeconds(payload.elapsed_seconds);
    const terminal = ["completed", "failed", "aborted"].includes(payload.state);
    etaNode.textContent = terminal ? "Complete" : formatSeconds(payload.eta_seconds);
    const requested = payload.worker_mode === "manual" ? String(payload.requested_workers) : "auto";
    const resolved = payload.resolved_workers ? ` -> ${payload.resolved_workers}` : "";
    workersNode.textContent = `${requested}${resolved}`;
    const percent = payload.total_points > 0 ? (payload.completed_points / payload.total_points) * 100 : 0;
    progressBar.style.width = `${percent}%`;
    errorNode.textContent = payload.error_text || "";
    if (currentLabelNode) {
      currentLabelNode.textContent = payload.current_label || "";
    }
    if (abortButton) {
      if (payload.can_abort) {
        abortButton.removeAttribute("aria-disabled");
      } else {
        abortButton.setAttribute("aria-disabled", "true");
      }
    }
    if (plotImage && payload.plot_url && payload.plot_token !== latestPlotToken) {
      preloadAndSwapPlot(payload.plot_url, payload.plot_token);
    }
    if (terminal) {
      if (statusTimerId !== null) {
        window.clearInterval(statusTimerId);
      }
      if (memoryTimerId !== null) {
        window.clearInterval(memoryTimerId);
      }
      // Results are rendered server-side, so a page loaded while the stage was
      // running has no way to show them. Monitors that opt in reload once; the
      // reloaded page renders the results instead of this monitor, so there is
      // no loop.
      if (container.hasAttribute("data-run-reload-on-finish")) {
        window.location.reload();
      }
    }
  }

  async function pollStatus() {
    const response = await fetch(statusUrl);
    const payload = await response.json();
    updateFromStatus(payload);
  }

  async function pollMemory() {
    const response = await fetch(memoryUrl);
    const payload = await response.json();
    if (!payload.ok) {
      memoryNode.textContent = "RAM unavailable";
      return;
    }
    memoryNode.textContent = `${formatBytes(payload.used_bytes)} used / ${formatBytes(payload.available_bytes)} free`;
  }

  pollStatus();
  pollMemory();
  statusTimerId = window.setInterval(pollStatus, 500);
  memoryTimerId = window.setInterval(pollMemory, 500);
}
