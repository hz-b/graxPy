function initBraggRun(page) {
  const mapFigure = page.querySelector('[data-bragg-figure="map"]');
  const curvesFigure = page.querySelector('[data-bragg-figure="curves"]');
  const logToggle = page.querySelector("[data-bragg-log]");
  const orderSelect = page.querySelector("[data-bragg-order]");
  // The values that change from curve to curve: amplitudes, or energies for an energy scan.
  const planned = JSON.parse(page.dataset.amplitudes);
  const seriesLabel = page.dataset.seriesLabel || "Deformation amplitude (nm)";
  const seriesUnit = page.dataset.seriesUnit || "nm";
  const angleLabel = page.dataset.angleLabel;
  const rounded = (value) => Math.round(value * 1e6) / 1e6;
  const config = { responsive: true, displaylogo: false };
  let lastData = null;
  let orderChosen = false;
  let timer = null;
  // Spreading long arrays into Math.max overflows the stack, so reduce instead.
  // Points not computed yet are null and are skipped.
  const largest = (values, start = -Infinity) => values.reduce(
    (best, value) => (value !== null && value > best ? value : best), start);

  // Keep the options of a select in step with the data, without losing the choice.
  const syncOptions = (select, values, labelFor, preferred) => {
    const current = select.value;
    const wanted = values.map(String);
    const existing = [...select.options].map((option) => option.value);
    if (wanted.join("|") !== existing.join("|")) {
      select.replaceChildren(...values.map((value) => new Option(labelFor(value), String(value))));
    }
    select.value = wanted.includes(current) ? current : String(preferred);
  };

  // The strongest odd order (switched on by the voltage), else the strongest
  // order other than the specular one, else the specular one.
  const preferredOrder = (curves, orders) => {
    const strongest = (candidates) => {
      let best = null;
      let bestValue = 1e-10;
      candidates.forEach((order) => {
        Object.values(curves).forEach((perOrder) => {
          const peak = largest(perOrder[String(order)] || [], 0);
          if (peak > bestValue) {
            best = order;
            bestValue = peak;
          }
        });
      });
      return best;
    };
    const odd = strongest(orders.filter((order) => Math.abs(order) % 2 === 1));
    if (odd !== null) {
      return odd;
    }
    const other = strongest(orders.filter((order) => order !== 0));
    return other !== null ? other : 0;
  };

  // Rows of a heatmap and the curves, one per amplitude, for the chosen data.
  const drawMapAndCurves = (angles, byAmplitude, options) => {
    const logScale = logToggle.checked;
    const amplitudes = [...new Set([...planned.map(rounded), ...byAmplitude.keys()])]
      .sort((first, second) => first - second);
    const empty = angles.map(() => null);
    const floor = options.floor;
    const rows = amplitudes.map((amplitude) => {
      const row = byAmplitude.get(amplitude);
      if (!row) {
        return empty;
      }
      return logScale ? row.map((value) => (value === null ? null : Math.log10(Math.max(value, floor)))) : row;
    });
    window.Plotly.react(mapFigure, [{
      type: "heatmap",
      x: angles,
      y: amplitudes,
      z: rows,
      colorscale: "Viridis",
      hoverongaps: false,
      zmin: logScale ? Math.log10(floor) : 0,
      zmax: logScale ? Math.log10(options.top) : options.top,
      colorbar: { title: { text: logScale ? "log10 intensity" : "Intensity" } },
    }], {
      xaxis: { title: { text: options.xTitle }, range: options.xRange },
      yaxis: { title: { text: seriesLabel } },
      margin: { t: 20 },
    }, config);
    const curveKeys = [...byAmplitude.keys()].sort((first, second) => first - second);
    window.Plotly.react(curvesFigure, curveKeys.map((amplitude) => ({
      x: angles,
      // On a log axis a zero would be dropped; show it at the floor instead.
      y: logScale ? byAmplitude.get(amplitude).map((value) => (value === null ? null : Math.max(value, floor))) : byAmplitude.get(amplitude),
      mode: options.markers ? "lines+markers" : "lines",
      marker: { size: 3 },
      name: `${amplitude} ${seriesUnit}`,
    })), {
      xaxis: { title: { text: options.xTitle }, range: options.xRange },
      yaxis: logScale
        ? { title: { text: `${options.yTitle} (log)` }, type: "log", range: [Math.log10(floor), Math.log10(options.top) + 0.05] }
        : { title: { text: options.yTitle }, range: [0, options.top * 1.05] },
      margin: { t: 20 },
    }, config);
  };

  const drawCrystal = (data) => {
    const byAmplitude = new Map(Object.entries(data.curves).map(([key, row]) => [rounded(Number(key)), row]));
    if (!byAmplitude.size) {
      return;
    }
    const limit = Math.min(0.02, largest(data.angles_deg));
    drawMapAndCurves(data.angles_deg, byAmplitude, {
      floor: 1e-3,
      top: 1,
      xTitle: angleLabel || "ΔΘ (°)",
      xRange: [-limit, limit],
      yTitle: "Intensity, normalized to the undeformed peak",
    });
  };

  const drawGrazing = (data) => {
    const amplitudeKeys = Object.keys(data.curves).sort((first, second) => Number(first) - Number(second));
    if (!amplitudeKeys.length) {
      return;
    }
    const orders = [...new Set(amplitudeKeys.flatMap((key) => Object.keys(data.curves[key]).map(Number)))]
      .sort((first, second) => first - second);
    const preferred = preferredOrder(data.curves, orders);
    syncOptions(orderSelect, ["total", ...orders], (value) => (value === "total" ? "All orders" : `Order ${value}`), preferred);
    if (!orderChosen) {
      orderSelect.value = String(preferred);
    }
    const chosen = orderSelect.value;
    const byAmplitude = new Map(amplitudeKeys.map((key) => [
      rounded(Number(key)),
      chosen === "total"
        ? data.total[key]
        : (data.curves[key][chosen] || data.total[key].map((value) => (value === null ? null : 0))),
    ]));
    const partial = [...byAmplitude.values()].some((row) => row.some((value) => value === null));
    const top = largest([...byAmplitude.values()].map((row) => largest(row)), 1e-12);
    drawMapAndCurves(data.angles_deg, byAmplitude, {
      floor: top * 1e-4,
      top,
      xTitle: angleLabel || "Grazing angle (°)",
      xRange: [data.angles_deg[0], data.angles_deg[data.angles_deg.length - 1]],
      yTitle: chosen === "total" ? "Reflected intensity, all orders" : `Order ${chosen} intensity`,
      markers: partial,
    });
  };

  const draw = (data) => {
    lastData = data;
    if (!window.Plotly || !mapFigure || !data) {
      return;
    }
    if (data.geometry === "grazing") {
      drawGrazing(data);
    } else {
      drawCrystal(data);
    }
  };

  logToggle.addEventListener("change", () => draw(lastData));
  if (orderSelect) {
    orderSelect.addEventListener("change", () => {
      orderChosen = true;
      draw(lastData);
    });
  }

  const poll = async () => {
    let terminal = false;
    try {
      const response = await fetch(page.dataset.statusUrl);
      if (response.ok) {
        const payload = await response.json();
        draw(payload.data);
        terminal = ["completed", "failed", "aborted"].includes(payload.state);
      } else {
        terminal = true;
      }
    } catch (error) {
      terminal = true;
    }
    if (!terminal) {
      timer = window.setTimeout(poll, 700);
    }
  };
  poll();
}

const braggRunPage = document.querySelector("[data-bragg-run-page]");
if (braggRunPage) {
  initBraggRun(braggRunPage);
}
