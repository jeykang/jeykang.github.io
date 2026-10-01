/*
 * app.js — orchestration for the NetAI Lakehouse wall display.
 *
 * Drives a continuous 4-stage loop (Ingest → Curate → GPU → Serve). To avoid
 * looking like a fixed replay it is MANIFEST-DRIVEN: if assets/clips/manifest.json
 * exists (produced by extract_assets.py from real NFS clips), every loop streams
 * a DIFFERENT real clip through the pipeline — new camera frame, new LiDAR sweep,
 * new metadata, new difficulty score. Counters tick continuously as if data is
 * still arriving. Without a manifest it falls back to the baked DEMO_DATA so the
 * kiosk can never show a broken screen.
 */
(function () {
  "use strict";
  const D = window.DEMO_DATA;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const fmt = new Intl.NumberFormat("en-US");
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // ── live-ish totals. These HOVER around the real baselines (bounded random
  // jitter) rather than accumulating, so a permanently-looping wall never
  // drifts to implausible numbers. ─────────────────────────────────────────
  const BASE_CLIPS = D.dataset.clips;
  const BASE_ROWS = D.process.bronze_rows;
  const BASE_TB = D.dataset.tb_downloaded;
  let liveClips = BASE_CLIPS;
  let liveScored = D.gpu.clips_scored;

  // ── clip playlist (real clips if manifest present) ───────────────────────
  let playlist = null;      // array of {id, frame, cloud, country, season, hour, score, ...}
  let playIdx = 0;
  let lidar = null;

  // ===========================================================================
  // static one-time rendering of baked structures
  // ===========================================================================
  function renderClock() {
    const t = new Date();
    $("#clock").textContent = t.toLocaleTimeString("en-GB");
  }

  function renderSensors() {
    const icons = { camera: "📷", lidar: "🛰️", radar: "📡", ego: "🧭" };
    $("#sensor-grid").innerHTML = D.dataset.sensors.map(s => `
      <div class="sensor-card">
        <div class="sc-top"><span class="sc-count">${s.count}</span><span class="sc-label">${icons[s.kind] || ""} ${s.label}</span></div>
        <div class="sensor-note">${s.note}</div>
      </div>`).join("");
    $("#ds-name").textContent = D.active_dataset.name;
    $("#ds-uploader").textContent = D.active_dataset.uploader;
    $("#ds-raw").textContent = D.active_dataset.raw_size;
    $("#ds-countries").textContent = D.dataset.countries;
    $("#ds-source").textContent = D.dataset.source;
  }

  // Stage 2 — "trim the fat": the dataset's raw→curated size reduction.
  function renderTrim() {
    const t = D.curate;
    $("#trim-stat").innerHTML = `
      <div class="trim-side raw"><div class="ts-val">${t.raw}</div><div class="ts-lbl">${t.raw_label}</div></div>
      <div class="trim-arrow"><span class="ta-ico">⟶</span><span class="ta-pct">${t.pct_kept}</span></div>
      <div class="trim-side kept"><div class="ts-val">${t.curated}</div><div class="ts-lbl">${t.curated_label}</div></div>`;
  }

  // Stage 4 — the dataset catalog: processed datasets shown side by side.
  const BADGE_BG = { active: "#76e36b", progress: "#38d6ff", registered: "#b98bff", synthetic: "#e8b923" };
  function renderCatalog() {
    $("#dataset-catalog").innerHTML = D.catalog.map(d => `
      <div class="dcard ${d.kind === "active" ? "active" : ""}" style="border-left-color:${d.color}">
        <div class="dcard-head"><span class="dcard-name">${d.name}</span><span class="dcard-src">${d.source}</span></div>
        <span class="dcard-badge" style="background:${BADGE_BG[d.kind] || "#7d8aa0"}">${d.status}</span>
        <div class="dcard-scale">${d.scale}</div>
        <div class="dcard-mods">${d.modalities}</div>
        <div class="dcard-note">${d.note}</div>
        ${d.thumb ? `<img class="dcard-thumb" src="${d.thumb}" alt="" onerror="this.style.display='none'"/>` : ""}
      </div>`).join("");
  }
  // refresh the active dataset card's thumbnail to the current real sample frame
  function updateActiveThumb(clip) {
    if (!clip || !clip.img) return;
    const t = document.querySelector(".dcard.active .dcard-thumb");
    if (t) { t.onerror = () => { t.style.display = "none"; }; t.src = clip.img; }
  }

  function renderFunnel() {
    const f = D.process.funnel;
    const max = f[0].clips;
    $("#funnel").innerHTML = f.map(r => `
      <div class="funnel-row">
        <div class="funnel-bar-wrap">
          <div class="funnel-bar" data-w="${(r.clips / max * 100).toFixed(1)}" style="width:0%;background:${r.color}">${r.tier}</div>
        </div>
        <div class="funnel-meta"><span class="fm-tier">${r.label}</span><span class="fm-clips">${fmt.format(r.clips)} clips</span></div>
      </div>`).join("");
    $("#pr-tables").textContent = D.process.bronze_tables;
    $("#pr-storage").textContent = D.process.bronze_storage_gb + " GB";
  }

  function renderTableBars() {
    const t = D.process.tables;
    const max = Math.log10(t[0].rows);
    const fmtRows = n => n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : n >= 1e6 ? (n / 1e6).toFixed(0) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "K" : n;
    $("#table-bars").innerHTML = t.map(r => `
      <div class="tbar">
        <span class="tb-name">${r.name}</span>
        <span class="tb-track"><span class="tb-fill" data-w="${(Math.log10(r.rows) / max * 100).toFixed(0)}"></span></span>
        <span class="tb-val">${fmtRows(r.rows)}</span>
      </div>`).join("");
  }

  function renderGauges() {
    $("#gpu-gauges").innerHTML = D.gpu.devices.map(g => `
      <div class="gauge">
        <div class="gauge-ring" data-target="0" style="position:relative"><span>0%</span></div>
        <div class="gauge-lbl">${g.name.replace("NVIDIA ", "")}<br>${g.vram_gb} GB</div>
      </div>`).join("");
  }

  function renderServices() {
    $("#serve-services").innerHTML = D.serve.services.map(s => `
      <div class="svc"><span class="svc-dot"></span><span class="svc-name">${s.name}</span><span class="svc-role">${s.role}</span></div>`).join("");
  }

  function renderStack() {
    $("#stack-strip").innerHTML = D.stack.map(s => `<span class="stack-pill">${s}</span>`).join("");
  }

  // per-stage badges. Time values already carry their own ⏱ prefix; the GPU
  // stage shows hardware (perception isn't benchmarked) so it has no clock.
  function renderStageTimes() {
    const t = D.stage_time;
    $("#time-0").textContent = t.ingest;
    $("#time-1").textContent = t.curate;
    $("#time-2").textContent = t.gpu;
    $("#time-3").textContent = t.serve;
  }

  // Bronze › Silver › Gold processing-time breakdown (Curate stage)
  function renderTimeSteps() {
    const s = D.pipeline_steps;
    $("#time-steps").innerHTML = s.map((x, i) => `
      <div class="tstep"><div class="tstep-t">${x.time}</div><div class="tstep-l">${x.step}</div><div class="tstep-n">${x.note}</div></div>
      ${i < s.length - 1 ? '<div class="tstep-arrow">›</div>' : ""}`).join("");
  }

  // hardware / runtime environment strip (context for the timings)
  function renderEnv() {
    $("#env-strip").innerHTML = D.environment
      .map(e => `<span class="env-item"><span class="env-ico">${e.icon}</span>${e.text}</span>`)
      .join('<span class="env-sep"></span>');
  }

  // ===========================================================================
  // histogram (canvas) — drawn once, with a moving highlight per loop
  // ===========================================================================
  function drawHistogram(highlightBin) {
    const cv = $("#hist-canvas");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    cv.width = r.width * dpr; cv.height = r.height * dpr;
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = r.width, H = r.height, h = D.gpu.histogram;
    const max = Math.max(...h);
    const bw = W / h.length;
    for (let i = 0; i < h.length; i++) {
      const bh = (h[i] / max) * (H - 8);
      const x = i * bw + 2, y = H - bh;
      const hot = i === highlightBin;
      const grad = ctx.createLinearGradient(0, y, 0, H);
      grad.addColorStop(0, hot ? "#e8b923" : "#7aa0ff");
      grad.addColorStop(1, hot ? "rgba(232,185,35,0.15)" : "rgba(122,160,255,0.12)");
      ctx.fillStyle = grad;
      ctx.fillRect(x, y, bw - 4, bh);
      if (hot) { ctx.shadowColor = "#e8b923"; ctx.shadowBlur = 16; ctx.fillRect(x, y, bw - 4, bh); ctx.shadowBlur = 0; }
    }
  }

  // ===========================================================================
  // bounding-box overlay on the GPU camera frame
  // ===========================================================================
  // intrinsic size of extracted frames (extract_assets.py --frame-w/--frame-h)
  const FRAME_W = 960, FRAME_H = 540;
  function drawBoxes(dets, reveal) {
    const cv = $("#bbox-canvas");
    const r = cv.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = r.width * dpr; cv.height = r.height * dpr;
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    // replicate object-fit:cover so normalized boxes land on the visible frame
    const scale = Math.max(r.width / FRAME_W, r.height / FRAME_H);
    const dW = FRAME_W * scale, dH = FRAME_H * scale;
    const offX = (r.width - dW) / 2, offY = (r.height - dH) / 2;
    const n = Math.floor(dets.length * reveal);
    const colors = { car: "#76e36b", truck: "#76e36b", bus: "#76e36b", building: "#38d6ff", vegetation: "#e8b923", person: "#ff6b9d", pedestrian: "#ff6b9d", bicycle: "#ffa641", motorcycle: "#ffa641", "traffic light": "#b98bff", "stop sign": "#ff6b6b" };
    for (let i = 0; i < n; i++) {
      const d = dets[i];
      const x = offX + d.x * dW, y = offY + d.y * dH, w = d.w * dW, h = d.h * dH;
      const col = colors[d.label] || "#38d6ff";
      ctx.strokeStyle = col; ctx.lineWidth = 2.2;
      ctx.shadowColor = col; ctx.shadowBlur = 8;
      ctx.strokeRect(x - w / 2, y - h / 2, w, h);
      ctx.shadowBlur = 0;
      ctx.fillStyle = col;
      ctx.font = "600 13px Segoe UI, sans-serif";
      const tag = `${d.label} ${(d.conf * 100) | 0}%`;
      const tw = ctx.measureText(tag).width + 8;
      ctx.fillRect(x - w / 2, y - h / 2 - 17, tw, 16);
      ctx.fillStyle = "#06121a";
      ctx.fillText(tag, x - w / 2 + 4, y - h / 2 - 4);
    }
  }

  // ===========================================================================
  // counter animation helpers
  // ===========================================================================
  function animateCount(el, from, to, dur, suffix) {
    const t0 = performance.now();
    function step(now) {
      const p = clamp((now - t0) / dur, 0, 1);
      const e = 1 - Math.pow(1 - p, 3);
      const v = from + (to - from) * e;
      el.textContent = fmt.format(Math.round(v)) + (suffix || "");
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ===========================================================================
  // STAGE SEQUENCER
  // ===========================================================================
  const stages = $$(".stage");
  const dots = $$("#stage-dots span");
  const STAGE_MS = 8200;
  let cur = -1;

  function activateStage(i) {
    stages.forEach((s, k) => s.classList.toggle("active", k === i));
    dots.forEach((d, k) => d.classList.toggle("on", k === i));
  }

  function onIngest(clip) {
    const numEl = $("[data-count='310895']");
    const target = BASE_CLIPS + Math.floor(Math.random() * 700);
    animateCount(numEl, target - 320, target, 1400, " clips");
    // download bar sweep — conveys ingest activity; the TB/archive totals stay real
    const fill = $("#dl-fill"), txt = $("#dl-txt");
    fill.style.width = "0%";
    let p = 0;
    const iv = setInterval(() => {
      p = Math.min(100, p + 2.4 + Math.random() * 3);
      fill.style.width = p + "%";
      txt.textContent = `${BASE_TB.toFixed(2)} TB · ${D.dataset.archives_recovered}/${D.dataset.archives_total} archives`;
      if (p >= 100) clearInterval(iv);
    }, 90);
  }

  function onCurate() {
    const numEl = $("[data-count='12206108151']");
    const target = BASE_ROWS + Math.floor(Math.random() * 4e6);
    animateCount(numEl, target - 4e6, target, 1600, " rows");
    setTimeout(() => $$(".funnel-bar").forEach(b => b.style.width = b.dataset.w + "%"), 150);
    setTimeout(() => $$(".tb-fill").forEach(b => b.style.width = b.dataset.w + "%"), 300);
  }

  function onGPU(clip) {
    // spin gauges to live-looking utilisation
    $$(".gauge-ring").forEach((g) => {
      const target = 55 + Math.random() * 42;
      g.style.setProperty("--pct", target.toFixed(0));
      g.querySelector("span").textContent = Math.round(target) + "%";
    });
    // reveal detection boxes progressively
    const dets = (clip && clip.detections) || D.gpu.detections;
    let r = 0;
    const iv = setInterval(() => { r += 0.12; drawBoxes(dets, clamp(r, 0, 1)); if (r >= 1) clearInterval(iv); }, 110);
    // score meter to this clip's score
    const score = clip && typeof clip.score === "number" ? clip.score
      : (D.gpu.score_min + Math.random() * (D.gpu.score_max - D.gpu.score_min));
    const fill = $("#score-fill"), val = $("#score-val");
    const pct = (score - 0.1) / (0.85 - 0.1) * 100;
    setTimeout(() => { fill.style.width = clamp(pct, 4, 100) + "%"; }, 200);
    let sv = 0; const sIv = setInterval(() => { sv += score / 12; if (sv >= score) { sv = score; clearInterval(sIv); } val.textContent = sv.toFixed(3); }, 70);
    liveScored += Math.floor(Math.random() * 6 + 1);
    // swap the GPU camera frame to the current clip's real frame
    if (clip && clip.img) { const im = $("#gpu-cam-img"); im.onerror = () => {}; im.src = clip.img; }
  }

  function onServe(clip) {
    drawHistogram(Math.floor(Math.random() * D.gpu.histogram.length));
    updateActiveThumb(clip);  // refresh active-dataset thumbnail to current sample
  }

  // current featured clip for this loop
  function currentClip() {
    if (playlist && playlist.length) return playlist[playIdx % playlist.length];
    // fallback: rotate through the 3 baked hard clips
    const hc = D.serve.hard_clips;
    const c = hc[playIdx % hc.length];
    return { img: c.img, score: c.score, factor: c.factor, where: c.where, detections: D.gpu.detections };
  }

  function tick() {
    cur = (cur + 1) % 4;
    activateStage(cur);
    const clip = currentClip();
    if (cur === 0) {
      onIngest(clip);
      // new LiDAR sweep for this loop's clip (real cloud if available)
      if (lidar) {
        if (clip && clip._cloudData) lidar.setRealCloud(clip._cloudData);
        else lidar.setProcedural((playIdx * 2654435761) % 1e6 | 0);
      }
    } else if (cur === 1) onCurate();
    else if (cur === 2) {
      // if the real cloud finished loading after the ingest stage, show it now
      if (lidar && clip && clip._cloudData) lidar.setRealCloud(clip._cloudData);
      onGPU(clip);
    }
    else if (cur === 3) {
      onServe(clip);
      playIdx++; // advance to next sample after a full pass
      maybePrefetchCloud();
    }
  }

  // base64 → Float32Array (used by the self-contained portable build, which
  // inlines clouds so the page runs from file:// with no server / no fetch).
  function b64ToFloat32(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }

  // Portable build (large libraries): each cloud is a tiny per-clip script that
  // sets window.__C[id]=<base64>. Loaded by injecting <script src> — which IS
  // permitted under file:// (unlike fetch), so this scales to thousands of
  // clips without one giant inlined file.
  function loadCloudJs(clip) {
    if (window.__C && window.__C[clip.id]) {
      try { clip._cloudData = b64ToFloat32(window.__C[clip.id]); } catch (e) {}
      delete window.__C[clip.id];
      return;
    }
    const s = document.createElement("script");
    s.src = clip.cloud_js;
    s.onload = () => {
      const b64 = window.__C && window.__C[clip.id];
      if (b64) { try { clip._cloudData = b64ToFloat32(b64); } catch (e) {} delete window.__C[clip.id]; }
      s.remove();
    };
    s.onerror = () => s.remove();
    document.head.appendChild(s);
  }

  // ── real point-cloud prefetch (binary Float32 xyz) ───────────────────────
  function maybePrefetchCloud() {
    if (!playlist || !playlist.length) return;
    const next = playlist[(playIdx) % playlist.length];
    if (!next || next._cloudData) return;
    if (next.cloud_b64) { try { next._cloudData = b64ToFloat32(next.cloud_b64); } catch (e) {} return; }
    if (next.cloud_js) { loadCloudJs(next); return; }   // portable: lazy per-clip script
    if (!next.cloud) return;                            // served build: fetch the .bin
    fetch(next.cloud).then(r => r.ok ? r.arrayBuffer() : null).then(buf => {
      if (buf) next._cloudData = new Float32Array(buf);
    }).catch(() => {});
  }

  // ===========================================================================
  // boot
  // ===========================================================================
  function ambientFlow() {
    // background particle stream flowing left→right across the whole wall
    const cv = $("#flow-canvas");
    const ctx = cv.getContext("2d");
    let parts = [];
    function resize() {
      cv.width = window.innerWidth; cv.height = window.innerHeight;
      parts = Array.from({ length: 70 }, () => ({
        x: Math.random() * cv.width, y: Math.random() * cv.height,
        v: 0.4 + Math.random() * 1.6, r: 0.6 + Math.random() * 1.8,
        a: 0.05 + Math.random() * 0.22,
      }));
    }
    resize(); window.addEventListener("resize", resize);
    (function loop() {
      ctx.clearRect(0, 0, cv.width, cv.height);
      for (const p of parts) {
        p.x += p.v; if (p.x > cv.width + 5) { p.x = -5; p.y = Math.random() * cv.height; }
        ctx.globalAlpha = p.a; ctx.fillStyle = "#38d6ff";
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      requestAnimationFrame(loop);
    })();
  }

  function applyManifest(j) {
    if (!j || !Array.isArray(j.clips) || !j.clips.length) return null;
    // shuffle so each kiosk restart orders clips differently
    playlist = j.clips.slice();
    for (let i = playlist.length - 1; i > 0; i--) { const k = Math.floor(Math.random() * (i + 1)); [playlist[i], playlist[k]] = [playlist[k], playlist[i]]; }
    if (typeof j.total_clips === "number") liveClips = j.total_clips;
    if (typeof j.total_scored === "number") liveScored = j.total_scored;
    return playlist;
  }

  function loadManifest() {
    // portable/self-contained build inlines the manifest as a global — no fetch,
    // so the page runs straight from file:// off a USB stick.
    if (window.DEMO_MANIFEST) return Promise.resolve(applyManifest(window.DEMO_MANIFEST));
    return fetch("assets/clips/manifest.json", { cache: "no-store" })
      .then(r => r.ok ? r.json() : null)
      .then(applyManifest)
      .catch(() => null);
  }

  function start() {
    renderSensors(); renderFunnel(); renderTableBars(); renderGauges();
    renderServices(); renderTrim(); renderCatalog();
    renderStageTimes(); renderTimeSteps(); renderEnv();
    renderClock(); setInterval(renderClock, 1000);
    drawHistogram(-1);
    ambientFlow();
    lidar = new window.LidarScene($("#lidar-canvas"));
    lidar.start();
    loadManifest().then(() => {
      maybePrefetchCloud();
      tick();
      setInterval(tick, STAGE_MS);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
