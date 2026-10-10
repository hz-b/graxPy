// Geometry schematics for gratings and plane mirrors.
//
// A view is a <div data-geometry-view> that draws a side-on SVG sketch of the
// optic with three switchable groups of indicators:
//
//   Beams    incident, specular and diffracted-order rays;
//   Angles   grazing/exit angles, blaze and anti-blaze angles, laminar wall angles;
//   Lengths  period, groove depth, plateau width, coating thickness, footprint.
//
// Values come from the forms named by data-geometry-bind (a CSS selector list,
// or "closest" for the form around the view) and/or the JSON in
// data-geometry-static, so a sketch follows what is typed. Focusing a field
// highlights the matching dimension even when its group is switched off.
//
// The sketch is not to scale: heights are exaggerated and beam angles are mapped
// monotonically to a legible range. Every label carries the real value.
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const GROUPS = ["beams", "angles", "lengths"];
  const STORAGE_KEY = "grax.geometry.groups";
  const HC_EV_NM = 1239.841984; // photon energy (eV) x wavelength (nm)

  // Input name -> indicator key, for highlighting while a field has focus.
  const FIELD_KEYS = {
    period_lpermm: "period",
    grating_density_lpermm: "period",
    depth_nm: "depth",
    depth_min_nm: "depth",
    depth_max_nm: "depth",
    width_to_period_ratio: "width",
    laminar_width_to_period_ratio: "width",
    left_wall_angle_deg: "wall_left",
    laminar_left_wall_angle_deg: "wall_left",
    right_wall_angle_deg: "wall_right",
    laminar_right_wall_angle_deg: "wall_right",
    blaze_angle_deg: "blaze",
    blaze_min_deg: "blaze",
    blaze_max_deg: "blaze",
    anti_blaze_angle_deg: "anti",
    grazing_angle_deg: "grazing",
    fixed_angle_deg: "grazing",
    angle_min: "grazing",
    angle_max: "grazing",
    diffraction_order: "order",
    energy_start_ev: "order",
    target_energy_ev: "order",
    layer_thickness_nm: "thickness",
    d_period_nm: "thickness",
    n_bilayers: "thickness",
    top_cap_thickness_nm: "thickness",
    footprint_length_mm: "footprint",
    grading_percent_per_mm: "grading",
    grading_mode: "grading",
  };

  // ------------------------------------------------------------------ math --

  const rad = (deg) => (deg * Math.PI) / 180;
  const deg = (radians) => (radians * 180) / Math.PI;
  const finite = (value) => typeof value === "number" && Number.isFinite(value);

  // Beam angles (measured from the surface) are mapped to a legible drawn angle.
  // The map is increasing, so the order of the beams on screen is the real one.
  function drawnAngle(angleDeg) {
    return 12 + 0.8 * angleDeg;
  }

  // Grazing exit angle of order m from cos(beta) = cos(alpha) - m*lambda/d, the
  // convention the solvers use (positive orders are the inside orders).
  // Returns null when the order is evanescent or an input is missing.
  function exitAngle(grazingDeg, order, energyEv, periodNm) {
    if (![grazingDeg, order, energyEv, periodNm].every(finite) || energyEv <= 0 || periodNm <= 0) {
      return null;
    }
    const cosBeta = Math.cos(rad(grazingDeg)) - (order * (HC_EV_NM / energyEv)) / periodNm;
    if (cosBeta > 1 || cosBeta < -1) {
      return null;
    }
    return deg(Math.acos(cosBeta));
  }

  // One period of the surface height profile, in nm, as [x, height] points.
  // Heights are the substrate surface height, as the solvers and the PNG
  // previews use them. Returns null when the geometry is not valid.
  function profilePoints(p) {
    const period = p.periodNm;
    if (!(period > 0)) {
      return null;
    }
    if (p.type === "laminar") {
      const depth = p.depthNm;
      const floor = (1 - p.widthRatio) * period;
      if (!(depth > 0) || !(p.widthRatio > 0 && p.widthRatio < 1)) {
        return null;
      }
      if (!(p.leftWallDeg > 0 && p.leftWallDeg <= 90 && p.rightWallDeg > 0 && p.rightWallDeg <= 90)) {
        return null;
      }
      const footLeft = depth / Math.tan(rad(p.leftWallDeg));
      const footRight = depth / Math.tan(rad(p.rightWallDeg));
      const land = period - floor - footLeft - footRight;
      if (land < 0) {
        return null;
      }
      const xRise = land / 2;
      const xTopLeft = xRise + footLeft;
      const xTopRight = xTopLeft + floor;
      const xFall = xTopRight + footRight;
      return {
        points: [[0, 0], [xRise, 0], [xTopLeft, depth], [xTopRight, depth], [xFall, 0], [period, 0]],
        height: depth,
        apex: null,
        feet: {xRise, xTopLeft, xTopRight, xFall, floor},
      };
    }
    if (p.type === "sinusoidal") {
      if (!(p.depthNm > 0)) {
        return null;
      }
      const points = [];
      const samples = 64;
      for (let index = 0; index <= samples; index += 1) {
        const x = (period * index) / samples;
        points.push([x, 0.5 * p.depthNm * (1 - Math.cos((2 * Math.PI * x) / period))]);
      }
      return {points, height: p.depthNm, apex: null, feet: null};
    }
    if (!(p.blazeDeg > 0 && p.blazeDeg < 90)) {
      return null;
    }
    const tanBlaze = Math.tan(rad(p.blazeDeg));
    if (finite(p.antiBlazeDeg) && p.antiBlazeDeg > 0) {
      if (!(p.antiBlazeDeg < 90)) {
        return null;
      }
      const tanAnti = Math.tan(rad(p.antiBlazeDeg));
      const apexX = (period * tanAnti) / (tanBlaze + tanAnti);
      const height = (period * tanBlaze * tanAnti) / (tanBlaze + tanAnti);
      return {points: [[0, 0], [apexX, height], [period, 0]], height, apex: apexX, feet: null};
    }
    const height = period * tanBlaze;
    return {points: [[0, 0], [period, height], [period, 0]], height, apex: period, feet: null};
  }

  // Mean surface height of a profile (area under the polyline / period).
  function meanHeight(points, period) {
    let area = 0;
    for (let i = 1; i < points.length; i += 1) {
      const [x0, y0] = points[i - 1];
      const [x1, y1] = points[i];
      area += ((y0 + y1) / 2) * (x1 - x0);
    }
    return area / period;
  }

  // ----------------------------------------------------------- param reading --

  // A source answers get(name) with the raw value of an input or JSON key.
  function formSource(form) {
    return {
      form,
      get(name) {
        const field = form.elements ? form.elements[name] : null;
        if (!field) {
          return undefined;
        }
        const node = field.length !== undefined && field.tagName !== "SELECT" ? field[field.length - 1] : field;
        return node && node.value !== undefined ? node.value : undefined;
      },
    };
  }

  function staticSource(data) {
    return {get: (name) => (data ? data[name] : undefined)};
  }

  function first(sources, names) {
    for (const source of sources) {
      for (const name of names) {
        const value = source.get(name);
        if (value !== undefined && value !== null && value !== "") {
          return value;
        }
      }
    }
    return undefined;
  }

  function number(sources, names) {
    const raw = first(sources, names);
    const value = raw === undefined ? NaN : Number(raw);
    return Number.isFinite(value) ? value : null;
  }

  function midpoint(sources, lowNames, highNames) {
    const low = number(sources, lowNames);
    const high = number(sources, highNames);
    if (low !== null && high !== null) {
      return (low + high) / 2;
    }
    return low !== null ? low : high;
  }

  function readGrating(sources) {
    const lines = number(sources, ["period_lpermm", "grating_density_lpermm"]);
    const type = String(first(sources, ["grating_type"]) || "blazed");
    const workflow = first(sources, ["workflow"]);
    const grazingFixed = workflow === undefined || workflow === "fixed_angle" || workflow === "parameter_study";
    return {
      kind: "grating",
      type: ["laminar", "sinusoidal"].includes(type) ? type : "blazed",
      periodNm: lines && lines > 0 ? 1e6 / lines : null,
      lpermm: lines,
      depthNm:
        number(sources, ["depth_nm"]) ?? midpoint(sources, ["depth_min_nm"], ["depth_max_nm"]),
      widthRatio: number(sources, ["width_to_period_ratio", "laminar_width_to_period_ratio"]),
      leftWallDeg: number(sources, ["left_wall_angle_deg", "laminar_left_wall_angle_deg"]),
      rightWallDeg: number(sources, ["right_wall_angle_deg", "laminar_right_wall_angle_deg"]),
      blazeDeg:
        number(sources, ["blaze_angle_deg"]) ?? midpoint(sources, ["blaze_min_deg"], ["blaze_max_deg"]),
      antiBlazeDeg: number(sources, ["anti_blaze_angle_deg"]),
      grazingDeg: grazingFixed ? number(sources, ["grazing_angle_deg"]) : null,
      grazingFromWorkflow: !grazingFixed,
      searchedAngle: first(sources, ["target_energy_ev"]) !== undefined,
      order: number(sources, ["diffraction_order"]),
      energyEv: number(sources, ["energy_start_ev", "target_energy_ev"]),
    };
  }

  function stackThickness(sources) {
    const cap = number(sources, ["top_cap_thickness_nm"]) || 0;
    if (String(first(sources, ["stack_type"]) || "single_layer") === "multilayer") {
      const period = number(sources, ["d_period_nm"]);
      const count = number(sources, ["n_bilayers"]);
      return period !== null && count !== null ? period * count + cap : null;
    }
    const layer = number(sources, ["layer_thickness_nm"]);
    return layer !== null ? layer + cap : null;
  }

  function readMirror(sources) {
    const mode = String(first(sources, ["scan_mode"]) || "energy");
    const fixed = number(sources, ["fixed_angle_deg"]);
    const low = number(sources, ["angle_min"]);
    const high = number(sources, ["angle_max"]);
    const ranged = mode === "angle" || mode === "map";
    return {
      kind: "mirror",
      grazingDeg: ranged ? null : fixed,
      rangeDeg: ranged && low !== null && high !== null ? [Math.min(low, high), Math.max(low, high)] : null,
      thicknessNm: stackThickness(sources),
      footprintMm: number(sources, ["footprint_length_mm"]),
      grading: String(first(sources, ["grading_mode"]) || "none"),
      gradingPercentPerMm: number(sources, ["grading_percent_per_mm"]),
    };
  }

  // -------------------------------------------------------------- svg helpers --

  function el(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs || {}).forEach(([name, value]) => node.setAttribute(name, String(value)));
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  const fmt = (value, digits) => {
    const text = Number(value).toFixed(digits === undefined ? 2 : digits);
    return text.includes(".") ? text.replace(/0+$/, "").replace(/\.$/, "") : text;
  };

  // Everything a scene adds goes through add(): it tags the element with its
  // group (for the switches) and an optional key (for highlight-on-focus).
  function makeScene(svg) {
    return function add(group, key, node) {
      node.classList.add("geo-item", `geo-${group}`);
      if (key) {
        node.dataset.key = key;
      }
      svg.appendChild(node);
      return node;
    };
  }

  function line(x1, y1, x2, y2, cls, extra) {
    return el("line", {x1, y1, x2, y2, class: cls, ...(extra || {})});
  }

  function label(x, y, text, cls, anchor) {
    return el("text", {x, y, class: `geo-text ${cls || ""}`, "text-anchor": anchor || "middle"}, text);
  }

  // Arc of radius r around (cx, cy) between two directions, in screen degrees
  // (0 = +x, 90 = down). Returns the path and the label position.
  function arc(cx, cy, r, fromDeg, toDeg, cls) {
    const a0 = rad(fromDeg);
    const a1 = rad(toDeg);
    const sweep = toDeg > fromDeg ? 1 : 0;
    const d = `M ${cx + r * Math.cos(a0)} ${cy + r * Math.sin(a0)} A ${r} ${r} 0 0 ${sweep} ${cx + r * Math.cos(a1)} ${cy + r * Math.sin(a1)}`;
    const mid = (a0 + a1) / 2;
    return {path: el("path", {d, class: cls}), labelX: cx + (r + 16) * Math.cos(mid), labelY: cy + (r + 16) * Math.sin(mid) + 4};
  }

  // Double-headed dimension line with a label.
  function dimension(add, group, key, x1, y1, x2, y2, text, offset, anchor) {
    add(group, key, line(x1, y1, x2, y2, "geo-dim", {"marker-start": "url(#geo-tick)", "marker-end": "url(#geo-tick)"}));
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const vertical = Math.abs(x2 - x1) < Math.abs(y2 - y1);
    const labelNode = vertical
      ? label(mx + (offset || 8), my + 4, text, "geo-dim-text", anchor || "start")
      : label(mx, my + (offset || 16), text, "geo-dim-text");
    add(group, key, labelNode);
  }

  function ray(add, key, x1, y1, x2, y2, cls) {
    add("beams", key, line(x1, y1, x2, y2, cls, {"marker-end": "url(#geo-arrow)"}));
  }

  // ------------------------------------------------------------------ scene --

  const W = 720;
  const H = 370;
  const X0 = 70;
  const X1 = 610;
  const BASE_Y = 225;

  function defs(svg) {
    const d = el("defs");
    const arrow = el("marker", {id: "geo-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"});
    arrow.appendChild(el("path", {d: "M 0 0 L 10 5 L 0 10 z", class: "geo-arrow-head"}));
    const tick = el("marker", {id: "geo-tick", viewBox: "0 0 10 10", refX: 5, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto"});
    tick.appendChild(el("path", {d: "M 5 0 L 5 10", class: "geo-tick-mark"}));
    d.append(arrow, tick);
    svg.appendChild(d);
  }

  // Incident ray, specular ray and the mean-surface/normal references, all
  // anchored at (hx, hy). Returns the drawn rays' geometry for the caller.
  function drawBeams(add, hx, hy, p) {
    const length = 230;
    const alpha = finite(p.grazingDeg) ? p.grazingDeg : 6;
    const f = drawnAngle(alpha);
    const dx = Math.cos(rad(f));
    const dy = Math.sin(rad(f));
    ray(add, "grazing", hx - length * dx, hy - length * dy, hx, hy, "geo-ray geo-incident");
    add("beams", "grazing", label(hx - length * dx + 4, hy - length * dy - 8, "incident", "geo-ray-text", "start"));
    ray(add, "grazing", hx, hy, hx + length * dx, hy - length * dy, "geo-ray geo-specular");
    // The order ray can sit just above or below the specular one; keep the two
    // labels on the outer sides of the pair.
    const beta = p.kind === "grating" && finite(p.order) && p.order !== 0 && finite(p.grazingDeg)
      ? exitAngle(alpha, p.order, p.energyEv, p.periodNm)
      : null;
    const orderAbove = beta !== null && beta > alpha;
    const specularLabelDy = beta === null ? -8 : orderAbove ? 22 : -8;
    add("beams", "grazing", label(hx + length * dx - 4, hy - length * dy + specularLabelDy, "specular (0)", "geo-ray-text", "end"));

    const alphaText = finite(p.grazingDeg)
      ? `α = ${fmt(p.grazingDeg)}°`
      : p.grazingFromWorkflow
        ? "α (from workflow)"
        : p.searchedAngle
          ? "α (theta search)"
          : "α (run form)";
    const incident = arc(hx, hy, 58, 180, 180 + f, "geo-arc");
    add("angles", "grazing", incident.path);
    add("angles", "grazing", label(hx - 100 * dx - 6, hy - 100 * dy - 12, alphaText, "geo-angle-text", "end"));

    let exit = null;
    if (p.kind === "grating" && finite(p.order) && p.order !== 0) {
      if (beta !== null) {
        exit = {beta, f: drawnAngle(beta)};
        const ex = Math.cos(rad(exit.f));
        const ey = Math.sin(rad(exit.f));
        ray(add, "order", hx, hy, hx + length * ex, hy - length * ey, "geo-ray geo-order");
        add("beams", "order", label(hx + length * ex - 4, hy - length * ey + (orderAbove ? -8 : 22), `order ${p.order}`, "geo-ray-text geo-order-text", "end"));
        const outgoing = arc(hx, hy, 86, 0, -exit.f, "geo-arc geo-order-arc");
        add("angles", "order", outgoing.path);
        add("angles", "order", label(hx + 110 * ex + 8, hy - 110 * ey + (orderAbove ? -10 : 20), `β = ${fmt(beta)}°`, "geo-angle-text geo-order-text", "start"));
      } else if (finite(p.energyEv) && finite(p.grazingDeg)) {
        add("beams", "order", label(hx + 20, hy - 120, `order ${p.order} is evanescent`, "geo-ray-text geo-order-text", "start"));
      }
    }

    // Mean-surface reference and the grating normal frame the angles.
    add("angles", "grazing", line(hx - 110, hy, hx + 110, hy, "geo-ref"));
    add("angles", null, line(hx, hy - 100, hx, hy + 30, "geo-ref"));
    add("angles", null, label(hx + 6, hy - 102, "N", "geo-ref-text", "start"));
    return exit;
  }

  function drawGrating(svg, p, status) {
    const add = makeScene(svg);
    const profile = profilePoints({
      type: p.type,
      periodNm: p.periodNm,
      depthNm: p.depthNm,
      widthRatio: p.widthRatio,
      leftWallDeg: p.leftWallDeg,
      rightWallDeg: p.rightWallDeg,
      blazeDeg: p.blazeDeg,
      antiBlazeDeg: p.antiBlazeDeg,
    });
    if (!profile) {
      status.textContent = "Enter a valid period and profile to see the geometry.";
      return;
    }
    status.textContent = "";
    const periods = 3;
    const kx = (X1 - X0) / periods / p.periodNm;
    const truePx = profile.height * kx;
    const heightPx = Math.min(70, Math.max(18, truePx));
    const kz = heightPx / profile.height;
    const exaggeration = kz / kx;
    const px = (nm) => nm * kx;
    const pz = (nm) => nm * kz;

    // Substrate fill and the surface line across three periods.
    const outline = [];
    for (let k = 0; k < periods; k += 1) {
      profile.points.forEach(([x, h]) => outline.push([X0 + px(x + k * p.periodNm), BASE_Y - pz(h)]));
    }
    const polygon = [[X0, BASE_Y + 80], ...outline, [X1, BASE_Y + 80]];
    add("base", null, el("polygon", {points: polygon.map((pt) => pt.join(",")).join(" "), class: "geo-substrate"}));
    add("base", null, el("polyline", {points: outline.map((pt) => pt.join(",")).join(" "), class: "geo-surface"}));

    const hx = X0 + px(0.9 * p.periodNm); // leaves the right-hand period free for the dimensions
    const mean = meanHeight(profile.points, p.periodNm);
    const hy = BASE_Y - pz(mean);
    drawBeams(add, hx, hy, p);

    // ---- profile-specific angles and lengths, in the middle period ----
    const k = 2;
    const ox = X0 + px(k * p.periodNm);
    const sx = (nm) => ox + px(nm);
    const sy = (nm) => BASE_Y - pz(nm);

    if (p.type === "blazed") {
      const apexX = profile.apex;
      const blazeArc = arc(sx(0), sy(0), 40, 0, -deg(Math.atan2(pz(profile.height), px(apexX))), "geo-arc geo-blaze-arc");
      add("angles", "blaze", blazeArc.path);
      add("angles", "blaze", label(sx(0) + 8, sy(0) + 20, `blaze θB = ${fmt(p.blazeDeg)}°`, "geo-angle-text", "start"));
      add("angles", "blaze", line(sx(0), sy(0), sx(0) + 70, sy(0), "geo-ref"));
      if (finite(p.antiBlazeDeg) && p.antiBlazeDeg > 0) {
        const fall = deg(Math.atan2(pz(profile.height), px(p.periodNm - apexX)));
        const antiArc = arc(sx(p.periodNm), sy(0), 40, 180, 180 + fall, "geo-arc geo-anti-arc");
        add("angles", "anti", antiArc.path);
        add("angles", "anti", label(sx(p.periodNm) - 8, sy(0) + 40, `anti-blaze θA = ${fmt(p.antiBlazeDeg)}°`, "geo-angle-text", "end"));
      } else {
        add("angles", "anti", label(sx(p.periodNm) - 6, sy(0) - 18, "sawtooth (abrupt drop)", "geo-angle-text", "end"));
      }
      dimension(add, "lengths", "depth", sx(apexX) + 14, sy(0), sx(apexX) + 14, sy(profile.height), `h = ${fmt(profile.height)} nm`, 8);
    } else if (p.type === "laminar") {
      const f = profile.feet;
      const leftArc = arc(sx(f.xRise), sy(0), 30, 0, -deg(Math.atan2(pz(profile.height), px(f.xTopLeft - f.xRise))), "geo-arc geo-wall-arc");
      add("angles", "wall_left", leftArc.path);
      add("angles", "wall_left", label(sx((f.xRise + f.xTopLeft) / 2), sy(profile.height / 2) + 22, `left wall ${fmt(p.leftWallDeg)}°`, "geo-angle-text"));
      const rightArc = arc(sx(f.xFall), sy(0), 30, 180, 180 + deg(Math.atan2(pz(profile.height), px(f.xFall - f.xTopRight))), "geo-arc geo-wall-arc");
      add("angles", "wall_right", rightArc.path);
      add("angles", "wall_right", label(sx((f.xTopRight + f.xFall) / 2), sy(profile.height / 2) + 22, `right wall ${fmt(p.rightWallDeg)}°`, "geo-angle-text"));
      const dimX = sx(f.xRise) - 34;
      add("lengths", "depth", line(dimX, sy(profile.height), sx(f.xTopLeft), sy(profile.height), "geo-ref"));
      dimension(add, "lengths", "depth", dimX, sy(0), dimX, sy(profile.height), `depth = ${fmt(profile.height)} nm`, -8, "end");
      dimension(add, "lengths", "width", sx(f.xTopLeft), sy(profile.height) - 14, sx(f.xTopRight), sy(profile.height) - 14, `plateau (1−r)·d = ${fmt(f.floor)} nm`, -6);
    } else {
      dimension(add, "lengths", "depth", sx(p.periodNm / 2) + 14, sy(0), sx(p.periodNm / 2) + 14, sy(profile.height), `depth = ${fmt(profile.height)} nm`, 8);
      add("angles", null, label(sx(p.periodNm / 2), sy(profile.height) - 18, "sinusoidal profile", "geo-angle-text"));
    }

    dimension(add, "lengths", "period", sx(0), BASE_Y + 64, sx(p.periodNm), BASE_Y + 64, `d = ${fmt(p.periodNm)} nm (${fmt(p.lpermm)} l/mm)`, 16);
    add("angles", null, label(X1, BASE_Y + 104, exaggeration > 1.05 ? `heights exaggerated ×${fmt(exaggeration, 1)}; angle labels give the real values` : "to scale; angle labels give the real values", "geo-note", "end"));
  }

  function drawMirror(svg, p, status) {
    const add = makeScene(svg);
    status.textContent = "";
    const thickness = 10;
    add("base", null, el("rect", {x: X0, y: BASE_Y, width: X1 - X0, height: 80, class: "geo-substrate"}));
    add("base", null, el("rect", {x: X0, y: BASE_Y - thickness, width: X1 - X0, height: thickness, class: "geo-coating"}));
    add("base", null, line(X0, BASE_Y - thickness, X1, BASE_Y - thickness, "geo-surface"));

    const hx = (X0 + X1) / 2;
    const hy = BASE_Y - thickness;
    drawBeams(add, hx, hy, {...p, grazingFromWorkflow: false, searchedAngle: false});
    if (p.rangeDeg) {
      // Angle scans: show the two extreme incident directions as a wedge.
      p.rangeDeg.forEach((angle, index) => {
        const f = drawnAngle(angle);
        const length = 200;
        add("beams", "grazing", line(hx - length * Math.cos(rad(f)), hy - length * Math.sin(rad(f)), hx, hy, "geo-ray-range"));
        add("angles", "grazing", label(hx - length * Math.cos(rad(f)) - 6, hy - length * Math.sin(rad(f)) + (index ? 14 : -4), `θ = ${fmt(angle)}°`, "geo-angle-text", "end"));
      });
    }

    if (p.footprintMm !== null && p.footprintMm > 0) {
      dimension(add, "lengths", "footprint", hx - 130, BASE_Y + 22, hx + 130, BASE_Y + 22, `footprint L = ${fmt(p.footprintMm)} mm`, 16);
    }
    if (p.thicknessNm !== null && p.thicknessNm > 0) {
      add("lengths", "thickness", line(X1 - 24, BASE_Y - thickness, X1 - 24, BASE_Y, "geo-dim", {"marker-start": "url(#geo-tick)", "marker-end": "url(#geo-tick)"}));
      add("lengths", "thickness", label(X1 - 32, BASE_Y - thickness - 10, `coating t = ${fmt(p.thicknessNm)} nm`, "geo-dim-text", "end"));
    }
    if (p.grading !== "none") {
      const text = p.grading === "linear" && p.gradingPercentPerMm !== null ? `thickness grading ${fmt(p.gradingPercentPerMm)} %/mm` : "thickness grading along the footprint";
      add("lengths", "grading", line(X0 + 30, BASE_Y + 44, X1 - 30, BASE_Y + 44, "geo-grading", {"marker-end": "url(#geo-arrow)"}));
      add("lengths", "grading", label((X0 + X1) / 2, BASE_Y + 60, text, "geo-dim-text"));
    }
    add("angles", null, label(X1, BASE_Y + 104, "not to scale; the coating is drawn thicker than it is", "geo-note", "end"));
  }

  // ------------------------------------------------------------------ mount --

  function loadGroups() {
    try {
      const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null");
      if (stored && typeof stored === "object") {
        return stored;
      }
    } catch (error) {
      // localStorage can be unavailable; fall back to the defaults.
    }
    return {};
  }

  function saveGroups(state) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
      // Not persisting is fine.
    }
  }

  function mount(root) {
    const kind = root.dataset.geometryKind === "mirror" ? "mirror" : "grating";
    let staticData = null;
    try {
      staticData = root.dataset.geometryStatic ? JSON.parse(root.dataset.geometryStatic) : null;
    } catch (error) {
      staticData = null;
    }
    const forms = [];
    const bind = root.dataset.geometryBind || "";
    bind
      .split(",")
      .map((selector) => selector.trim())
      .filter(Boolean)
      .forEach((selector) => {
        const form = selector === "closest" ? root.closest("form") : document.querySelector(selector);
        if (form && !forms.includes(form)) {
          forms.push(form);
        }
      });

    const groups = {beams: true, angles: true, lengths: true, ...loadGroups()};
    const bar = document.createElement("div");
    bar.className = "geo-controls";
    const title = document.createElement("strong");
    title.textContent = "Show on drawing:";
    bar.appendChild(title);
    GROUPS.forEach((group) => {
      const wrapper = document.createElement("label");
      wrapper.className = "inline-check";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(groups[group]);
      input.dataset.geometryGroup = group;
      const text = document.createElement("span");
      text.textContent = group.charAt(0).toUpperCase() + group.slice(1);
      wrapper.append(input, text);
      bar.appendChild(wrapper);
      input.addEventListener("change", () => {
        groups[group] = input.checked;
        saveGroups(groups);
        applyGroups();
      });
    });

    const status = document.createElement("p");
    status.className = "subtle geo-status";
    const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, class: "geo-svg", role: "img", "aria-label": `${kind} geometry`});
    root.replaceChildren(bar, svg, status);

    function applyGroups() {
      GROUPS.forEach((group) => root.classList.toggle(`show-${group}`, Boolean(groups[group])));
    }

    function render() {
      const sources = [...forms.map(formSource)];
      if (staticData) {
        sources.push(staticSource(staticData));
      }
      svg.replaceChildren();
      defs(svg);
      const params = kind === "mirror" ? readMirror(sources) : readGrating(sources);
      (kind === "mirror" ? drawMirror : drawGrating)(svg, params, status);
    }

    forms.forEach((form) => {
      form.addEventListener("input", render);
      form.addEventListener("change", render);
      form.addEventListener("focusin", (event) => highlight(event.target && event.target.name));
      form.addEventListener("focusout", () => highlight(null));
    });

    function highlight(name) {
      const key = name ? FIELD_KEYS[name] : null;
      svg.querySelectorAll(".is-focused").forEach((node) => node.classList.remove("is-focused"));
      if (key) {
        svg.querySelectorAll(`[data-key="${key}"]`).forEach((node) => node.classList.add("is-focused"));
      }
    }

    applyGroups();
    render();
    root.geometryRender = render;
  }

  function init() {
    document.querySelectorAll("[data-geometry-view]").forEach(mount);
  }

  window.GraxGeometry = {exitAngle, profilePoints, meanHeight, drawnAngle, init};
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
