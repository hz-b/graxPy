// Multilayer diagnostics page: draw the stored plot. For the theta-scan view each
// scan file is one group (its rough and precise curves, its selected peak and one
// legend entry) that is shown, hidden or stepped through as a unit.
const PLOTLY_KEYS = ["scan_class", "group", "role"];

function initDiagnosticPlot() {
  const stage = document.querySelector("[data-diagnostic-plot]");
  const source = document.getElementById("diagnostic-plot-data");
  if (!stage || !source || typeof Plotly === "undefined") {
    return;
  }
  const data = JSON.parse(source.textContent);
  const meta = data.traces.map((trace) => ({
    scanClass: trace.scan_class,
    group: trace.group,
    role: trace.role,
  }));
  const traces = data.traces.map((trace) => {
    const clean = { ...trace };
    PLOTLY_KEYS.forEach((key) => delete clean[key]);
    return clean;
  });
  const config = { responsive: true, displaylogo: false, scrollZoom: true };
  const controls = document.querySelector("[data-diagnostic-controls]");
  if (!controls) {
    Plotly.newPlot(stage, traces, data.layout, config);
    return;
  }

  const groupNames = [];
  const groupColors = [];
  meta.forEach((item, index) => {
    if (item.role === "legend") {
      groupNames[item.group] = data.traces[index].name;
      groupColors[item.group] = data.traces[index].line.color;
    }
  });
  // Plotly fills a horizontal legend row by row; this one fills column by column.
  data.layout.showlegend = false;
  data.layout.margin = { ...data.layout.margin, b: 60 };
  const legend = document.querySelector("[data-diagnostic-legend]");
  const legendItems = groupNames.map((name, group) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "diagnostic-legend-item";
    item.title = "Click to show or hide this scan; double-click to show only this scan";
    item.innerHTML = '<span class="diagnostic-legend-swatch"></span><span></span>';
    item.firstChild.style.background = groupColors[group];
    item.lastChild.textContent = name;
    legend.appendChild(item);
    return item;
  });
  const kindBoxes = [...controls.querySelectorAll("[data-scan-kind]")];
  const stepBox = controls.querySelector("[data-step-mode]");
  const stepper = controls.querySelector("[data-stepper]");
  const picker = controls.querySelector("[data-step-select]");
  const allActions = controls.querySelector("[data-all-actions]");
  const filterD = controls.querySelector("[data-filter-d]");
  const filterBlaze = controls.querySelector("[data-filter-blaze]");
  const counter = controls.querySelector("[data-step-count]");
  const hidden = new Set();
  let current = 0;

  // Filter menus list the d-spacings and blaze angles found in the scan names.
  const designValue = (name, pattern) => {
    const match = name.match(pattern);
    return match ? match[1] : null;
  };
  const groupD = groupNames.map((name) => designValue(name, /d([0-9.]+)nm/));
  const groupBlaze = groupNames.map((name) => designValue(name, /blaze([0-9.]+)deg/));
  const fillFilter = (select, values, unit) => {
    select.add(new Option("All", ""));
    [...new Set(values.filter((value) => value !== null))]
      .sort((a, b) => Number(a) - Number(b))
      .forEach((value) => select.add(new Option(`${value} ${unit}`, value)));
  };
  fillFilter(filterD, groupD, "nm");
  fillFilter(filterBlaze, groupBlaze, "deg");
  const applyFilter = () => {
    hidden.clear();
    groupNames.forEach((_, index) => {
      const keepD = !filterD.value || groupD[index] === filterD.value;
      const keepBlaze = !filterBlaze.value || groupBlaze[index] === filterBlaze.value;
      if (!(keepD && keepBlaze)) {
        hidden.add(index);
      }
    });
    apply();
  };

  groupNames.forEach((name, index) => picker.add(new Option(name, String(index))));

  const apply = () => {
    const single = stepBox.checked;
    const kinds = new Set(kindBoxes.filter((box) => box.checked).map((box) => box.value));
    const visible = meta.map((item) => {
      const shown = single ? item.group === current : !hidden.has(item.group);
      if (item.role === "legend") {
        return shown ? true : single ? false : "legendonly";
      }
      if (item.role === "peak") {
        return shown;
      }
      return shown && kinds.has(item.scanClass);
    });
    legendItems.forEach((item, group) => {
      item.classList.toggle("is-off", single ? group !== current : hidden.has(group));
    });
    stepper.classList.toggle("is-hidden", !single);
    allActions.classList.toggle("is-hidden", single);
    picker.value = String(current);
    counter.textContent = `${current + 1} / ${groupNames.length}`;
    Plotly.restyle(stage, { visible }).then(() =>
      Plotly.relayout(stage, { "xaxis.autorange": true, "yaxis.autorange": true }),
    );
  };
  const step = (delta) => {
    current = (current + delta + groupNames.length) % groupNames.length;
    apply();
  };

  Plotly.newPlot(stage, traces, data.layout, config).then(() => {
    // A legend entry stands for a whole scan: its curves and its selected peak.
    legendItems.forEach((item, group) => {
      item.addEventListener("click", (event) => {
        if (event.detail > 1) {
          return;
        }
        if (stepBox.checked) {
          current = group;
        } else if (hidden.has(group)) {
          hidden.delete(group);
        } else {
          hidden.add(group);
        }
        apply();
      });
      item.addEventListener("dblclick", () => {
        if (stepBox.checked) {
          return;
        }
        hidden.clear();
        groupNames.forEach((_, index) => index !== group && hidden.add(index));
        apply();
      });
    });
    kindBoxes.forEach((box) => box.addEventListener("change", apply));
    stepBox.addEventListener("change", apply);
    picker.addEventListener("change", () => {
      current = Number(picker.value);
      apply();
    });
    filterD.addEventListener("change", applyFilter);
    filterBlaze.addEventListener("change", applyFilter);
    controls.querySelector("[data-hide-all]").addEventListener("click", () => {
      groupNames.forEach((_, index) => hidden.add(index));
      filterD.value = "";
      filterBlaze.value = "";
      apply();
    });
    controls.querySelector("[data-show-all]").addEventListener("click", () => {
      hidden.clear();
      filterD.value = "";
      filterBlaze.value = "";
      apply();
    });
    controls.querySelector("[data-step-prev]").addEventListener("click", () => step(-1));
    controls.querySelector("[data-step-next]").addEventListener("click", () => step(1));
    apply();
  });
}

document.addEventListener("DOMContentLoaded", initDiagnosticPlot);
