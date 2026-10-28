// The survey figures and the (d, blaze) picker live in different sections of the
// design detail page, so they talk through document-level events instead of
// reaching into each other: a click on the map asks the picker to toggle a case,
// and the picker announces its selection so the map can mark it.
const SURVEY_CELL_PICKED = "grax:survey-cell-picked";
const DESIGN_SELECTION_CHANGED = "grax:design-selection-changed";

function formatDesignValue(value) {
  return Number(value).toFixed(3);
}

function designPairKey(d, blaze) {
  return `${formatDesignValue(d)},${formatDesignValue(blaze)}`;
}

function nearestGridValue(values, target) {
  let best = null;
  let bestDistance = Infinity;
  values.forEach((value) => {
    const distance = Math.abs(Number(value) - Number(target));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = Number(value);
    }
  });
  return best;
}

function surveyFigureLayout(meta, xTitle, yTitle, title, options) {
  // The survey energy is a property of the survey, so it belongs on the survey
  // plots only -- naming it on an energy scan would be actively misleading.
  const parts = (options || {}).withEnergy === false
    ? [meta.coating_label, `order ${meta.diffraction_order}`]
    : [meta.coating_label, `${meta.target_energy_ev} eV`, `order ${meta.diffraction_order}`];
  const subtitle = parts
    .filter((part) => part !== undefined && part !== null && part !== "")
    .join(" · ");
  return {
    template: "plotly_white",
    title: {text: subtitle ? `${title}<br><sub>${subtitle}</sub>` : title, x: 0.02},
    margin: {l: 64, r: 24, t: 72, b: 56},
    xaxis: {title: {text: xTitle}},
    yaxis: {title: {text: yTitle}},
    showlegend: false,
  };
}

function buildSurveyFigures(options, meta, selectedKeys) {
  const dValues = (options.d_values || []).map(Number);
  const blazeValues = (options.blaze_values || []).map(Number);
  const efficiency = options.efficiency || {};
  const perD = options.per_d || [];

  const ridgeD = perD.map((pair) => Number(pair[0]));
  const ridgeBlaze = perD.map((pair) => Number(pair[1]));
  const ridgeEfficiency = perD.map((pair) => efficiency[designPairKey(pair[0], pair[1])] ?? null);

  const optimalBlaze = {
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x: ridgeD,
        y: ridgeBlaze,
        line: {color: "#b0b7bd", width: 1.4},
        marker: {
          size: 11,
          color: ridgeEfficiency,
          colorscale: "Viridis",
          colorbar: {title: {text: "peak efficiency"}},
        },
        customdata: ridgeEfficiency,
        hovertemplate:
          "d = %{x:.3f} nm<br>blaze = %{y:.3f} deg<br>efficiency = %{customdata:.4f}<extra></extra>",
      },
    ],
    layout: surveyFigureLayout(
      meta,
      "Bilayer d-spacing (nm)",
      "Optimal blaze angle (deg)",
      "Optimal blaze angle versus d-spacing",
    ),
  };

  const maxEfficiency = {
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x: ridgeD,
        y: ridgeEfficiency,
        line: {color: "#b0b7bd", width: 1.4},
        marker: {
          size: 11,
          color: ridgeBlaze,
          colorscale: "Plasma",
          colorbar: {title: {text: "optimal blaze (deg)"}},
        },
        customdata: ridgeBlaze,
        hovertemplate:
          "d = %{x:.3f} nm<br>efficiency = %{y:.4f}<br>blaze = %{customdata:.3f} deg<extra></extra>",
      },
    ],
    layout: surveyFigureLayout(
      meta,
      "Bilayer d-spacing (nm)",
      "Max selected-order efficiency",
      "Max efficiency versus d-spacing",
    ),
  };

  // z is indexed [blaze][d]; a cell the survey never solved stays null so Plotly
  // leaves a gap instead of drawing a misleading zero.
  const z = blazeValues.map((blaze) =>
    dValues.map((d) => {
      const value = efficiency[designPairKey(d, blaze)];
      return value === undefined ? null : Number(value);
    }),
  );
  const selected = Array.from(selectedKeys || []).map((key) => key.split(",").map(Number));
  const heatmapLayout = surveyFigureLayout(
    meta,
    "Bilayer d-spacing (nm)",
    "Blaze angle (deg)",
    "Peak efficiency over (d, blaze) — click a cell to scan it",
  );
  heatmapLayout.showlegend = true;
  // Inside the axes, like the matplotlib version: a legend above the plot would
  // land on the title's subtitle line.
  heatmapLayout.legend = {
    orientation: "h",
    x: 0.02,
    xanchor: "left",
    y: 0.98,
    yanchor: "top",
    bgcolor: "rgba(255, 255, 255, 0.78)",
  };
  const heatmap = {
    data: [
      {
        type: "heatmap",
        x: dValues,
        y: blazeValues,
        z,
        colorscale: "Viridis",
        colorbar: {title: {text: "peak efficiency"}},
        hovertemplate:
          "d = %{x:.3f} nm<br>blaze = %{y:.3f} deg<br>efficiency = %{z:.4f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "optimal blaze per d",
        x: ridgeD,
        y: ridgeBlaze,
        line: {color: "#ffffff", width: 1.4},
        marker: {color: "#ffffff", size: 6},
        hoverinfo: "skip",
      },
      {
        type: "scatter",
        mode: "markers",
        name: "selected designs",
        x: selected.map((pair) => pair[0]),
        y: selected.map((pair) => pair[1]),
        // square-open takes its stroke from marker.color, not marker.line.
        marker: {symbol: "square-open", size: 16, color: "#ff3b30", line: {width: 3}},
        hoverinfo: "skip",
      },
    ],
    layout: heatmapLayout,
  };

  return {optimal_blaze: optimalBlaze, max_efficiency: maxEfficiency, heatmap};
}

function initEnergyScanFigure(root) {
  const stage = root.querySelector("[data-energy-scan-figure]");
  const liveUrl = root.dataset.energyScanLiveUrl;
  if (!stage || !liveUrl || !window.Plotly) {
    return;
  }
  let meta;
  try {
    meta = JSON.parse(root.dataset.surveyPlotMeta || "{}");
  } catch (error) {
    meta = {};
  }

  function draw(designs) {
    const layout = surveyFigureLayout(
      meta,
      "Photon energy (eV)",
      "Selected-order efficiency",
      "Efficiency versus energy",
      {withEnergy: false},
    );
    // Below the axes: one entry per design, and the curves themselves leave no
    // reliable free corner.
    layout.showlegend = true;
    layout.legend = {orientation: "h", x: 0, y: -0.22, yanchor: "top"};
    layout.margin = {l: 64, r: 24, t: 72, b: 96};
    renderPlotlyFigure(stage, {
      data: designs.map((design) => ({
        type: "scatter",
        mode: "lines+markers",
        name: `d = ${formatDesignValue(design.d_spacing_nm)} nm, blaze = ${formatDesignValue(design.blaze_angle_deg)} deg`,
        x: design.energies_ev || [],
        y: design.efficiencies || [],
        marker: {size: 5},
        line: {width: 1.6},
        hovertemplate: "%{x:.1f} eV<br>efficiency = %{y:.4f}<extra>%{fullData.name}</extra>",
      })),
      layout,
    });
  }

  async function poll() {
    try {
      const response = await window.fetch(liveUrl, {cache: "no-store"});
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      if ((payload.designs || []).some((design) => (design.energies_ev || []).length)) {
        draw(payload.designs);
      }
    } catch (error) {
      // A dropped poll is not worth surfacing; the next one will catch up.
    }
  }

  poll();
  window.setInterval(poll, 4000);
}

function initSurveyFigures(root) {
  let options;
  let meta;
  try {
    options = JSON.parse(root.dataset.designOptions || "{}");
    meta = JSON.parse(root.dataset.surveyPlotMeta || "{}");
  } catch (error) {
    return;
  }
  const stages = new Map();
  root.querySelectorAll("[data-survey-figure]").forEach((node) => {
    stages.set(node.dataset.surveyFigure, node);
  });
  if (stages.size === 0 || !window.Plotly) {
    return;
  }
  const dValues = (options.d_values || []).map(Number);
  const blazeValues = (options.blaze_values || []).map(Number);
  let selectedKeys = new Set();

  function draw() {
    const figures = buildSurveyFigures(options, meta, selectedKeys);
    stages.forEach((node, name) => {
      if (figures[name]) {
        renderPlotlyFigure(node, figures[name]);
      }
    });
  }

  draw();

  // While the survey runs its table is rewritten after every cell, so re-reading
  // it shows the grid filling in. The monitor reloads the page when the stage
  // finishes, which is what ends this polling.
  const liveUrl = root.dataset.surveyLiveUrl;
  if (liveUrl) {
    window.setInterval(async () => {
      try {
        const response = await window.fetch(liveUrl, {cache: "no-store"});
        if (!response.ok) {
          return;
        }
        const fresh = await response.json();
        if ((fresh.per_d || []).length === 0) {
          return;
        }
        options = fresh;
        dValues.splice(0, dValues.length, ...(fresh.d_values || []).map(Number));
        blazeValues.splice(0, blazeValues.length, ...(fresh.blaze_values || []).map(Number));
        draw();
      } catch (error) {
        // A dropped poll is not worth surfacing; the next one will catch up.
      }
    }, 4000);
  }

  const heatmapNode = stages.get("heatmap");
  if (heatmapNode && typeof heatmapNode.on === "function") {
    heatmapNode.on("plotly_click", (event) => {
      const point = (event.points || [])[0];
      if (!point) {
        return;
      }
      const d = nearestGridValue(dValues, point.x);
      const blaze = nearestGridValue(blazeValues, point.y);
      if (d === null || blaze === null) {
        return;
      }
      if ((options.efficiency || {})[designPairKey(d, blaze)] === undefined) {
        return; // an unsolved cell is not a design anyone can scan
      }
      document.dispatchEvent(
        new CustomEvent(SURVEY_CELL_PICKED, {detail: {d, blaze}}),
      );
    });
  }

  document.addEventListener(DESIGN_SELECTION_CHANGED, (event) => {
    selectedKeys = new Set((event.detail && event.detail.pairs) || []);
    draw();
  });
}
