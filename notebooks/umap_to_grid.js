// UMAP -> grid morph animation, drawn with Observable Plot.
//
// Shared by notebooks/umap_to_grid.py (via wigglystuff ObservablePlot) and the
// exported Observable notebook cell. This file is a single JS *expression*
// that evaluates to a DOM node, with these variables in scope:
//
//   Plot, d3                  - Observable Plot and d3
//   width, height             - available size in px, including `chrome`
//   chrome                    - px above the plot for summary + controls
//   points                    - [{sx, sy, dx, dy, color, img, label}, ...]
//                               start (sx, sy) and end (dx, dy) in mosaic pixels
//   cols, rows                - grid size
//   cellW, cellH              - cell size in mosaic pixels
//   canvasW, canvasH          - mosaic canvas size in pixels
//   imgAspect, meanDist       - OPTIONAL; if defined, a summary line is shown
//                               (image aspect ratio; mean displacement in cell diagonals)
(() => {
  const root = document.createElement("div");
  if (typeof meanDist !== "undefined" && meanDist != null) {
    const nThumbs = points.filter((p) => p.img).length;
    const summary = document.createElement("div");
    summary.style.cssText = "font:13px sans-serif;margin-bottom:6px;color:#444";
    summary.textContent = [
      nThumbs ? `${nThumbs} thumbnails` : `${points.length} points`,
      `grid ${cols}×${rows}`,
      `image aspect ${(+imgAspect).toFixed(2)} (cells ${cellW}×${cellH}px)`,
      `mean displacement ${(+meanDist).toFixed(2)} cell diagonals`,
    ].join(" · ");
    root.append(summary);
  }
  const controls = document.createElement("div");
  controls.style.cssText =
    "display:flex;gap:8px;align-items:center;margin-bottom:6px;font:13px sans-serif";
  const play = document.createElement("button");
  play.textContent = "▶";
  const slider = Object.assign(document.createElement("input"), {
    type: "range", min: 0, max: 1, step: 0.005, value: 0,
  });
  slider.style.width = "300px";
  const label = document.createElement("span");
  const lines = Object.assign(document.createElement("input"), { type: "checkbox" });
  const linesLabel = document.createElement("label");
  linesLabel.append(lines, " displacement lines");
  // how thumbnails fill their cells: cover (crop to fill, like the mosaic)
  // or fit (whole image, letterboxed); neither distorts the image
  const fitMode = document.createElement("select");
  fitMode.append(new Option("cover", "slice"), new Option("fit", "meet"));
  const fitLabel = document.createElement("label");
  fitLabel.append("thumbnails: ", fitMode);
  controls.append(play, slider, label, linesLabel);
  if (points.some((p) => p.img)) controls.append(fitLabel);
  const holder = document.createElement("div");
  root.append(controls, holder);

  const ease = (t) => 0.5 - 0.5 * Math.cos(Math.PI * t);
  // all coordinates are in mosaic pixels: canvasW x canvasH, cells cellW x cellH
  const xs = d3.range(cols + 1).map((i) => i * cellW);
  const ys = d3.range(rows + 1).map((i) => i * cellH);
  // fit the canvas into the available space, preserving its aspect ratio
  const k = Math.min(width / canvasW, (height - chrome) / canvasH);
  const plotW = Math.round(canvasW * k), plotH = Math.round(canvasH * k);
  const cellPxW = cellW * k, cellPxH = cellH * k;
  const useImg = points.some((p) => p.img);

  function draw(t) {
    const e = ease(t);
    const pos = points.map((p) => ({
      ...p,
      x: (1 - e) * p.sx + e * p.dx,
      y: (1 - e) * p.sy + e * p.dy,
    }));
    const size = 0.25 + 0.75 * e;
    const marks = [
      Plot.ruleX(xs, { stroke: "#ccc", strokeOpacity: e }),
      Plot.ruleY(ys, { stroke: "#ccc", strokeOpacity: e }),
    ];
    if (lines.checked)
      marks.push(Plot.link(points, {
        x1: "sx", y1: "sy", x2: "dx", y2: "dy", stroke: "color", strokeOpacity: 0.4,
      }));
    marks.push(useImg
      ? Plot.image(pos, {
          x: "x", y: "y", src: "img",
          width: cellPxW * size, height: cellPxH * size, title: "label",
          preserveAspectRatio: `xMidYMid ${fitMode.value}`,
        })
      : Plot.dot(pos, {
          x: "x", y: "y", fill: "color",
          r: 2 + (Math.min(cellPxW, cellPxH) / 2 - 2) * e, title: "label",
        }));
    const plot = Plot.plot({
      width: plotW, height: plotH, margin: 0,
      x: { domain: [0, canvasW], axis: null },
      // row 0 at top, like the mosaic
      y: { domain: [0, canvasH], reverse: true, axis: null },
      marks,
    });
    holder.replaceChildren(plot);
    label.textContent = `t = ${t.toFixed(2)}`;
  }

  let raf = null, dir = 1;
  function step() {
    let t = +slider.value + dir * 0.008;
    if (t >= 1) { t = 1; stop(); dir = -1; }
    if (t <= 0) { t = 0; stop(); dir = 1; }
    slider.value = t;
    draw(t);
    if (raf) raf = requestAnimationFrame(step);
  }
  function stop() {
    cancelAnimationFrame(raf);
    raf = null;
    play.textContent = "▶";
  }
  play.onclick = () => {
    if (raf) return stop();
    play.textContent = "⏸";
    raf = requestAnimationFrame(step);
  };
  slider.oninput = () => { stop(); draw(+slider.value); };
  lines.onchange = () => draw(+slider.value);
  fitMode.onchange = () => draw(+slider.value);
  // in Observable, stop animating when the cell re-runs (e.g. on resize)
  if (typeof invalidation !== "undefined") invalidation.then(stop);
  draw(0);
  return root;
})()
