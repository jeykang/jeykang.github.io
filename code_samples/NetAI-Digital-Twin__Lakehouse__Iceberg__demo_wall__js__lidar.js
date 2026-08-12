/*
 * lidar.js — self-contained Canvas2D point-cloud renderer.
 *
 * No three.js / WebGL / CDN: a tiny hand-rolled 3D projector so the hallway
 * kiosk has zero external dependencies. Renders either
 *   (a) a REAL decoded LiDAR sweep (Float32 xyz array from assets/clips/*.bin), or
 *   (b) a procedural 360° driving scene (fallback when no real cloud is loaded).
 * Continuously rotates with a sweeping "scan" highlight so it always looks live.
 */
(function () {
  "use strict";

  function makeProceduralCloud(seed) {
    // Deterministic-ish PRNG so a given seed is stable but seeds differ per loop.
    let s = (seed * 9301 + 49297) % 233280;
    const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    const pts = [];
    // Ground rings (road surface) — concentric, denser near ego.
    for (let r = 2; r < 46; r += 1.1) {
      const n = Math.floor(r * 4);
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 + rnd() * 0.05;
        const jitter = (rnd() - 0.5) * 0.6;
        pts.push([Math.cos(a) * (r + jitter), Math.sin(a) * (r + jitter), -1.6 + rnd() * 0.15]);
      }
    }
    // Two building walls flanking a road corridor (x = lateral).
    for (let side = -1; side <= 1; side += 2) {
      for (let y = -40; y < 40; y += 0.8) {
        if (Math.abs(y) < 3) continue;
        for (let z = -1.4; z < 6; z += 0.7) {
          if (rnd() > 0.55) continue;
          pts.push([side * (9 + rnd() * 1.5), y, z]);
        }
      }
    }
    // A handful of car-sized boxes (other vehicles) scattered on the road.
    const cars = [[-2.4, 12], [2.6, 24], [-2.2, -10], [3.0, 34], [-2.6, 6]];
    for (const [cx, cy] of cars) {
      for (let i = 0; i < 90; i++) {
        pts.push([cx + (rnd() - 0.5) * 2.0, cy + (rnd() - 0.5) * 4.2, -1.4 + rnd() * 1.5]);
      }
    }
    return pts;
  }

  // Accepts a flat Float32Array [x,y,z,x,y,z,...] (real cloud) → array of [x,y,z].
  function unpackFlat(flat) {
    const out = [];
    for (let i = 0; i + 2 < flat.length; i += 3) out.push([flat[i], flat[i + 1], flat[i + 2]]);
    return out;
  }

  class LidarScene {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.points = makeProceduralCloud(1);
      this.rot = 0;
      this.scan = 0;
      this._raf = null;
      this._resize();
      window.addEventListener("resize", () => this._resize());
      // the GPU stage expands (flex-grow) when active, changing the canvas's
      // CSS size; re-resize the backing buffer so the cloud stays crisp.
      if (window.ResizeObserver) {
        this._ro = new ResizeObserver(() => this._resize());
        this._ro.observe(canvas);
      }
    }

    _resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = this.canvas.getBoundingClientRect();
      this.w = Math.max(2, r.width); this.h = Math.max(2, r.height);
      this.canvas.width = this.w * dpr; this.canvas.height = this.h * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    // Load a real decoded cloud (Float32Array xyz). Recenters & normalizes scale.
    setRealCloud(flat) {
      let pts = unpackFlat(flat);
      if (!pts.length) { this.points = makeProceduralCloud(Math.random() * 1e5 | 0); return; }
      // recenter on median, scale so the 95th pct radius ~ 38 units
      let cx = 0, cy = 0, cz = 0;
      for (const p of pts) { cx += p[0]; cy += p[1]; cz += p[2]; }
      cx /= pts.length; cy /= pts.length; cz /= pts.length;
      const rad = pts.map(p => Math.hypot(p[0] - cx, p[1] - cy)).sort((a, b) => a - b);
      const p95 = rad[Math.floor(rad.length * 0.95)] || 1;
      const k = 38 / p95;
      this.points = pts.map(p => [(p[0] - cx) * k, (p[1] - cy) * k, (p[2] - cz) * k]);
    }

    setProcedural(seed) { this.points = makeProceduralCloud(seed); }

    _project(p) {
      // rotate around vertical (z) axis, then a fixed downward tilt for a 3/4 view.
      const c = Math.cos(this.rot), s = Math.sin(this.rot);
      let x = p[0] * c - p[1] * s;
      let y = p[0] * s + p[1] * c;
      let z = p[2];
      // tilt: pitch the world forward so we look down at ~28°
      const tilt = 0.50;
      const yt = y * Math.cos(tilt) - z * Math.sin(tilt);
      const zt = y * Math.sin(tilt) + z * Math.cos(tilt);
      // simple perspective: camera back along +y'
      const camDist = 70;
      const depth = camDist - yt;
      if (depth <= 6) return null;
      const f = 620 / depth;
      return { sx: this.w / 2 + x * f, sy: this.h * 0.56 - zt * f, depth, ang: Math.atan2(p[1], p[0]) };
    }

    frame() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      this.rot += 0.0045;
      this.scan += 0.06;
      const scanAng = (this.scan % (Math.PI * 2)) - Math.PI;

      // painter's order: far → near
      const proj = [];
      for (const p of this.points) {
        const q = this._project(p);
        if (q) proj.push({ q, z: p[2] });
      }
      proj.sort((a, b) => b.q.depth - a.q.depth);

      for (const { q, z } of proj) {
        // colour by height: low=cyan/blue road, mid=green, high=amber (buildings)
        const t = Math.max(0, Math.min(1, (z + 1.8) / 7));
        let r, g, b;
        if (t < 0.5) { const u = t / 0.5; r = 30 + u * 30; g = 150 + u * 80; b = 255 - u * 90; }
        else { const u = (t - 0.5) / 0.5; r = 60 + u * 180; g = 230 - u * 40; b = 165 - u * 120; }
        // brighten points near the rotating scan beam
        let near = Math.abs(((q.ang - scanAng + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
        const beam = near < 0.18 ? (1 - near / 0.18) : 0;
        const fade = Math.max(0.18, Math.min(1, 64 / q.depth));
        const size = Math.max(0.7, 2.4 * (620 / q.depth) * 0.06 + (beam * 1.4));
        ctx.globalAlpha = fade * (0.55 + beam * 0.45);
        ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
        ctx.fillRect(q.sx, q.sy, size, size);
      }

      // ego vehicle marker at center
      ctx.globalAlpha = 1;
      const ex = this.w / 2, ey = this.h * 0.56;
      ctx.strokeStyle = "rgba(120,227,107,0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(ex, ey, 6, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ex, ey - 11); ctx.lineTo(ex, ey + 11); ctx.moveTo(ex - 11, ey); ctx.lineTo(ex + 11, ey); ctx.stroke();
    }

    start() {
      if (this._raf) return;
      const loop = () => { this.frame(); this._raf = requestAnimationFrame(loop); };
      this._raf = requestAnimationFrame(loop);
    }
    stop() { if (this._raf) cancelAnimationFrame(this._raf); this._raf = null; }
  }

  window.LidarScene = LidarScene;
})();
