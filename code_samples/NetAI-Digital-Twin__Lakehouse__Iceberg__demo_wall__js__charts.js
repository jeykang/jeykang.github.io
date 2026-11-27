/*
 * charts.js — the wall's small charts, as plain HTML and SVG (no libraries).
 *
 * Signage rules: thin marks, hairline axes, labels in ink colours (never the
 * series colour), selective direct labels, a legend wherever two series share a
 * chart, and status colours only next to an icon and a word. Nothing here is
 * interactive: nobody touches a hallway kiosk.
 */
(function () {
  "use strict";
  const fmt = new Intl.NumberFormat("en-US");
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // Medallion funnel. Linear scale: the 100-to-1 drop is the point, so no log axis.
  function funnel(el, rows) {
    const max = Math.max(...rows.map(r => r.clips));
    el.innerHTML = rows.map(r => `
      <div class="f-row">
        <div class="f-lbl"><b>${esc(r.tier)}</b>${esc(r.label)}</div>
        <div class="f-track">
          <div class="f-bar" style="background:${r.color}" data-w="${(100 * r.clips / max).toFixed(2)}"></div>
          <div class="f-val">${fmt.format(r.clips)}${r.note ? `<small>${esc(r.note)}</small>` : ""}</div>
        </div>
      </div>`).join("");
  }
  function grow(el) {
    // the value label rides the bar end; leave it room on the widest bar
    el.querySelectorAll(".f-bar").forEach(b => { b.style.width = `calc(${b.dataset.w}% * 0.78)`; });
  }
  function shrink(el) { el.querySelectorAll(".f-bar").forEach(b => { b.style.width = "0"; }); }

  // The two difficulty signals and their noisy-OR union, 0..1.
  function axes(el, clip) {
    const row = (label, v, color) => `
      <div class="ax"><span>${label}</span>
        <div class="ax-track"><div class="ax-fill" style="background:${color}" data-w="${(100 * v).toFixed(1)}"></div></div>
        <span class="ax-v">${v.toFixed(2)}</span></div>`;
    el.innerHTML =
      row("Traffic conflict", clip.conflict ?? 0, "var(--s1)") +
      row("Camera perception", clip.camera ?? 0, "var(--s3)") +
      row("<b style='color:var(--ink)'>Difficulty</b>", clip.difficulty ?? 0, "var(--ink-2)");
    requestAnimationFrame(() => requestAnimationFrame(() =>
      el.querySelectorAll(".ax-fill").forEach(f => { f.style.width = f.dataset.w + "%"; })));
  }

  // Percentile meter with the Gold zone (top 10 %) and this clip's marker.
  function meter(el, pct) {
    el.innerHTML = `
      <div class="meter">
        <div class="meter-track">
          <div class="meter-gold"><span>Gold · top 10 %</span></div>
          <div class="meter-mark"></div>
        </div>
        <div class="meter-axis"><span>easiest</span><span>percentile among 31,737 scored clips</span><span>hardest</span></div>
      </div>`;
    const mark = el.querySelector(".meter-mark");
    requestAnimationFrame(() => requestAnimationFrame(() => { mark.style.left = Math.max(0.5, Math.min(99.5, pct)) + "%"; }));
  }

  // Scenario x serving-mode table: counts from 1 to 101,382, so numbers, not shading.
  function scenarioGrid(el, g, hotCol) {
    let h = `<div class="h"></div>` + g.cols.map(c => `<div class="h">${esc(c)}</div>`).join("");
    for (const r of g.rows) {
      h += `<div class="c">${esc(r.cond)}</div>` + r.cells.map((v, i) =>
        `<div class="${v ? "" : "z"}${i === hotCol ? " hot" : ""}">${v ? fmt.format(v) : "—"}</div>`).join("");
    }
    h += `<div class="c t">Total</div>` + g.totals.map((v, i) => `<div class="t${i === hotCol ? " hot" : ""}">${fmt.format(v)}</div>`).join("");
    el.innerHTML = `<div class="sg">${h}</div>`;
  }

  // One twin's build time as a part-to-whole bar, with a legend (5 segments).
  const SEG_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
  function hm(min) {
    const h = Math.floor(min / 60), m = Math.round(min - 60 * h);
    return h ? `${h} h ${String(m).padStart(2, "0")} m` : `${Math.max(1, m)} min`;
  }
  function buildBar(el, build, steps) {
    if (!build) {
      el.innerHTML = `<div class="bb-title">The first twin: built step by step, before the pipeline existed</div>`;
      return;
    }
    if (build.resumed) {
      el.innerHTML = `<div class="bb-title">Built in two sessions (the run was resumed), so no single timeline</div>`;
      return;
    }
    const segs = steps.map((s, i) => ({ ...s, min: build[s.key] || 0, color: SEG_COLORS[i] }));
    const total = segs.reduce((a, s) => a + s.min, 0);
    el.innerHTML = `
      <div class="bb-title">Built in<b>${hm(build.total_min)}</b> on one A10 GPU, unattended</div>
      <div class="bb-track">${segs.map((s, i) =>
        `<div class="bb-seg" style="flex:${Math.max(s.min, total * 0.006)};background:${s.color};transition-delay:${0.1 + i * 0.12}s"></div>`).join("")}</div>
      <div class="bb-legend">${segs.map(s => `<span><i style="background:${s.color}"></i>${esc(s.label)}<b>${hm(s.min)}</b></span>`).join("")}</div>`;
  }

  // Triage waffle: one cell per closed-loop scene, in the screen's rollout order.
  // Shape carries the class too (filled = at-fault failure, hollow = none), so it
  // survives colour-blindness.
  function waffle(el, clips) {
    el.innerHTML = clips.map(c => `<div class="wc${c.at_fault ? " fail" : ""}"></div>`).join("");
    return Array.from(el.children);
  }

  // Recall vs budget: the screen against random order (the diagonal). Two series,
  // so a legend; the screen also gets a direct label at the budget it is run at.
  function recallChart(el, curve, budget) {
    const W = el.clientWidth, H = el.clientHeight;
    if (W < 40 || H < 40) return null;
    const u = Math.min(window.innerHeight / 100, window.innerWidth * 0.005625);
    const fs = 1.5 * u, pad = { l: 5.4 * u, r: 2 * u, t: 3.6 * u, b: 4.2 * u };
    const x = v => pad.l + v * (W - pad.l - pad.r);
    const y = v => H - pad.b - v * (H - pad.t - pad.b);
    const grid = [0, 0.5, 1].map(v => `
      <line x1="${x(0)}" x2="${x(1)}" y1="${y(v)}" y2="${y(v)}" stroke="var(--hair)" stroke-width="1"/>
      <text x="${x(0) - 0.8 * u}" y="${y(v)}" fill="var(--ink-3)" font-size="${fs}" text-anchor="end" dominant-baseline="middle">${v * 100}%</text>`).join("");
    const xt = [0, 0.25, 0.5, 0.75, 1].map(v =>
      `<text x="${x(v)}" y="${H - pad.b + 2.2 * u}" fill="var(--ink-3)" font-size="${fs}" text-anchor="middle">${v * 100}%</text>`).join("");
    const path = curve.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    const at = curve.reduce((best, p) => Math.abs(p[0] - budget) < Math.abs(best[0] - budget) ? p : best, curve[0]);
    el.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" aria-label="Share of at-fault failures found against the share of clips rolled out">
        ${grid}${xt}
        <text x="${(x(0) + x(1)) / 2}" y="${H - 0.2 * u}" fill="var(--ink-3)" font-size="${fs}" text-anchor="middle">clips rolled out, hardest-predicted first</text>
        <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--ink-3)" stroke-width="2" stroke-linecap="round"/>
        <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
        <line class="rc-guide" x1="${x(at[0])}" x2="${x(at[0])}" y1="${y(0)}" y2="${y(at[1])}" stroke="var(--ink-2)" stroke-width="1"/>
        <circle class="rc-dot" cx="${x(at[0])}" cy="${y(at[1])}" r="${0.75 * u}" fill="var(--accent)" stroke="var(--surface)" stroke-width="3"/>
        <text x="${x(at[0]) + 1.2 * u}" y="${y(at[1]) + 0.5 * u}" fill="var(--ink)" font-size="${2.2 * u}" font-weight="700">${Math.round(at[1] * 100)}%</text>
        <g font-size="${fs}" fill="var(--ink-2)">
          <line x1="${x(0) + 1 * u}" x2="${x(0) + 3 * u}" y1="${pad.t - 2 * u}" y2="${pad.t - 2 * u}" stroke="var(--accent)" stroke-width="3" stroke-linecap="round"/>
          <text x="${x(0) + 3.8 * u}" y="${pad.t - 2 * u}" dominant-baseline="middle">triage screen</text>
          <line x1="${x(0) + 17 * u}" x2="${x(0) + 19 * u}" y1="${pad.t - 2 * u}" y2="${pad.t - 2 * u}" stroke="var(--ink-3)" stroke-width="2" stroke-linecap="round"/>
          <text x="${x(0) + 19.8 * u}" y="${pad.t - 2 * u}" dominant-baseline="middle">random order</text>
        </g>
      </svg>`;
    return { x, y };
  }

  window.WallCharts = { funnel, grow, shrink, axes, meter, scenarioGrid, buildBar, hm, waffle, recallChart };
})();
