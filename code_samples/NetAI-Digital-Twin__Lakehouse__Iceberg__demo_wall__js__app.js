/*
 * app.js — the wall's loop: six chapters, one real sample each per loop.
 *
 * Collect → Curate follow the same real clip (camera frame, LiDAR sweep, its Gold
 * difficulty); Reconstruct → Validate follow the same twin (real vs render, then
 * VaVAM driving it beside NVIDIA's reconstruction); Serve shows one Cosmos window,
 * Triage the 80-scene screen. Samples rotate every loop, so the wall rarely repeats.
 *
 * Inputs: window.WALL_DATA (js/data.js, always present), window.WALL_MEDIA
 * (assets/media/media.js, optional) and the clip library (assets/clips/manifest.json,
 * or window.DEMO_MANIFEST in the portable build, optional). Without the optional
 * parts every chapter still renders, from data.js and the three baked frames.
 */
(function () {
  "use strict";
  const D = window.WALL_DATA;
  const M = window.WALL_MEDIA || { twins: [], pairs: [], cosmos: [], triage: null, scores: null };
  const C = window.WallCharts;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const fmt = new Intl.NumberFormat("en-US");
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const chap = Object.fromEntries(D.chapters.map(c => [c.id, c]));
  const order = D.chapters.map(c => c.id);
  const section = id => $(`.chapter[data-ch="${id}"]`);
  const FALLBACK_FRAMES = ["assets/cam1.jpg", "assets/cam2.jpg", "assets/cam3.jpg"];

  let library = null;      // real clips (shuffled)
  let loop = 0;
  let cur = -1;
  let timer = null;
  let lidar = null;
  const pick = {};         // this loop's samples
  const stops = [];        // cleanup for the chapter that is leaving

  // ── static parts ────────────────────────────────────────────────────────
  function renderStatic() {
    $("#brand-name").innerHTML = D.brand.name;
    $("#brand-sub").textContent = D.brand.sub;
    $("#brand-ko").textContent = D.brand.sub_ko;
    $("#loop-line").textContent = D.brand.loop;

    $("#rail").innerHTML = D.chapters.map((c, i) => `
      ${i ? '<div class="link"></div>' : ""}
      <div class="node" data-node="${c.id}">
        <span class="n-idx">${c.idx}</span>
        <span class="n-title">${esc(c.title)}<span class="ko">${esc(c.ko)}</span></span>
        <span class="n-fig"><b>${esc(c.rail.value)}</b>${esc(c.rail.unit)}</span>
        <span class="n-prog"></span>
      </div>`).join("");

    for (const c of D.chapters) {
      section(c.id).querySelector(".info").innerHTML = `
        <div class="eyebrow"><span class="idx">${c.idx}</span>${esc(c.title)}<span class="ko">${esc(c.ko)}</span></div>
        <h2>${esc(c.headline)}</h2>
        <p class="lede">${esc(c.lede)}</p>
        <div class="stats">${c.stats.map(s => `
          <div class="stat"><div class="stat-v" ${typeof s.value === "number" ? `data-n="${s.value}"` : ""}>${typeof s.value === "number" ? fmt.format(s.value) : esc(s.value)}${s.suffix ? `<small>${esc(s.suffix)}</small>` : ""}</div>
          <div class="stat-l">${esc(s.label)}</div></div>`).join("")}</div>
        <div class="now"></div>`;
    }

    C.funnel($("#funnel"), chap.curate.funnel);
    C.scenarioGrid($("#scenario-grid"), chap.serve.scenario_grid, 2);
    $("#collect-chips").innerHTML = ["<b>7</b>cameras", "<b>1</b>LiDAR 360°", "<b>+</b>radar", "<b>+</b>ego-motion"]
      .map(t => `<span class="chip">${t}</span>`).join("");
    $("#foot-stack").innerHTML = D.footer.map(t => `<span>${esc(t)}</span>`).join("");
    const snap = M.scores ? ` · difficulty scores of ${String(M.scores.committed_at).slice(0, 10)}` : "";
    $("#foot-snap").textContent = `Figures as of ${D.snapshot}${snap}`;
  }

  function tick() {
    const t = new Date();
    $("#clock").textContent = t.toLocaleTimeString("en-GB");
    $("#date").textContent = t.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric" }).toUpperCase();
  }

  function setNow(id, html) { section(id).querySelector(".now").innerHTML = html; }

  function countUp(id) {
    for (const el of $$(".stat-v[data-n]", section(id))) {
      const to = +el.dataset.n, small = el.querySelector("small");
      const t0 = performance.now(), dur = 1300;
      const step = now => {
        const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
        el.firstChild.nodeValue = fmt.format(Math.round(to * e));
        if (p < 1) requestAnimationFrame(step);
      };
      if (el.firstChild && el.firstChild.nodeType === 3) requestAnimationFrame(step);
    }
  }

  function playVideo(v, src) {
    if (v.getAttribute("src") !== src) v.src = src;
    v.currentTime = 0;
    const p = v.play();
    if (p && p.catch) p.catch(() => {});
  }

  // ── samples for this loop ───────────────────────────────────────────────
  function pickSamples() {
    if (library && library.length) {
      // prefer clips that carry a difficulty, so Curate always has a story
      for (let tries = 0; tries < 8; tries++) {
        pick.clip = library[(loop * 8 + tries) % library.length];
        if (pick.clip.difficulty != null) break;
      }
    } else {
      pick.clip = { img: FALLBACK_FRAMES[loop % 3], id: null, detections: [] };
    }
    const twins = M.twins || [], pairs = M.pairs || [];
    pick.twin = twins.length ? twins[loop % twins.length] : null;
    pick.pair = pick.twin ? pairs.find(p => p.short === pick.twin.short) : (pairs[loop % Math.max(1, pairs.length)] || null);
    pick.cosmos = M.cosmos && M.cosmos.length ? M.cosmos[(loop * 7) % M.cosmos.length] : null;
    prefetchCloud(pick.clip);
  }

  // ── LiDAR cloud for the clip (served: .bin via fetch; portable: per-clip script)
  function b64ToFloat32(b64) {
    const bin = atob(b64), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }
  function prefetchCloud(clip) {
    if (!clip || clip._cloud || !clip.id) return;
    if (clip.cloud_js) {
      const s = document.createElement("script");
      s.src = clip.cloud_js;
      s.onload = () => {
        const b = window.__C && window.__C[clip.id];
        if (b) { try { clip._cloud = b64ToFloat32(b); } catch (e) { /* procedural fallback */ } delete window.__C[clip.id]; }
        s.remove();
      };
      s.onerror = () => s.remove();
      document.head.appendChild(s);
    } else if (clip.cloud) {
      fetch(clip.cloud).then(r => (r.ok ? r.arrayBuffer() : null))
        .then(buf => { if (buf) clip._cloud = new Float32Array(buf); }).catch(() => {});
    }
  }

  // ── chapter: Collect ────────────────────────────────────────────────────
  const FRAME_W = 960, FRAME_H = 540;      // extract_assets.py frame size
  function drawBoxes(dets, reveal) {
    const cv = $("#collect-boxes"), r = cv.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = r.width * dpr; cv.height = r.height * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const k = Math.max(r.width / FRAME_W, r.height / FRAME_H);       // object-fit: cover
    const dw = FRAME_W * k, dh = FRAME_H * k, ox = (r.width - dw) / 2, oy = (r.height - dh) / 2;
    const u = Math.min(window.innerHeight / 100, window.innerWidth * 0.005625);
    ctx.font = `600 ${1.5 * u}px "Segoe UI", system-ui, sans-serif`;
    const n = Math.floor((dets || []).length * reveal);
    for (let i = 0; i < n; i++) {
      const d = dets[i];
      const x = ox + (d.x - d.w / 2) * dw, y = oy + (d.y - d.h / 2) * dh, w = d.w * dw, h = d.h * dh;
      ctx.strokeStyle = "#38d6ff"; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);
      const tag = `${d.label} ${Math.round(d.conf * 100)}%`, tw = ctx.measureText(tag).width + 0.9 * u;
      ctx.fillStyle = "rgba(5,9,17,0.82)"; ctx.fillRect(x, y - 2.1 * u, tw, 2.1 * u);
      ctx.fillStyle = "#eef3fb"; ctx.fillText(tag, x + 0.45 * u, y - 0.55 * u);
    }
  }
  function whereOf(c) {
    const bits = [c.country, c.season, c.hour != null && c.hour !== "" ? `${String(c.hour).padStart(2, "0")}:00` : null];
    return bits.filter(Boolean).join(" · ");
  }
  function enterCollect() {
    const c = pick.clip, img = $("#collect-img");
    img.onerror = () => { img.onerror = null; img.src = FALLBACK_FRAMES[loop % 3]; };
    img.src = c.img;
    const dets = c.detections || [];
    let r = 0;
    const iv = setInterval(() => { r = Math.min(1, r + 0.1); drawBoxes(dets, r); if (r >= 1) clearInterval(iv); }, 120);
    stops.push(() => clearInterval(iv));
    if (lidar) {
      if (c._cloud) lidar.setRealCloud(c._cloud);
      else {
        lidar.setProcedural((loop * 2654435761) % 1e6 | 0);
        // the real sweep is still downloading: swap it in when it lands
        const wait = setInterval(() => { if (c._cloud) { lidar.setRealCloud(c._cloud); clearInterval(wait); } }, 250);
        stops.push(() => clearInterval(wait));
      }
      lidar.start();
      stops.push(() => lidar.stop());
    }
    const short = c.id ? c.id.slice(0, 8) : null;
    $("#collect-cap").innerHTML = short
      ? `<b>Clip ${short}</b> · ${esc(whereOf(c))} · front-wide camera, 120°`
      : "A real clip from the on-disk sample";
    const n = dets.length;
    const seen = n ? `A camera-only detector finds <b>${n}</b> object${n === 1 ? "" : "s"} in this frame.`
                   : "A camera-only detector finds no objects in this frame.";
    setNow("collect", short
      ? `<span class="now-k">Now showing</span>Clip <b>${short}</b>, one of 32,986 on disk. ${seen} Next: is it hard enough for Gold?`
      : `<span class="now-k">Now showing</span>A real frame from the on-disk sample.`);
  }

  // ── chapter: Curate ─────────────────────────────────────────────────────
  function enterCurate() {
    const c = pick.clip;
    $("#curate-thumb").src = c.img;
    $("#curate-id").textContent = c.id ? c.id.slice(0, 8) : "";
    setTimeout(() => C.grow($("#funnel")), 150);
    stops.push(() => C.shrink($("#funnel")));
    if (c.difficulty == null) {
      $("#curate-axes").innerHTML = `<div class="lede">This clip lacks the sensors the difficulty signals need, so it stays out of Gold.</div>`;
      $("#curate-meter").innerHTML = "";
      setNow("curate", `<span class="now-k">This clip</span>Not scored: missing sensor data.`);
      return;
    }
    C.axes($("#curate-axes"), c);
    C.meter($("#curate-meter"), c.pct);
    const meterEl = $("#curate-meter");
    const verdict = c.gold
      ? `<span class="badge gold">Gold</span> harder than <b>${Math.floor(c.pct)} %</b> of the scored clips`
      : `<span class="badge silver">Silver</span> harder than <b>${Math.floor(c.pct)} %</b>; Gold starts at 90 %`;
    meterEl.insertAdjacentHTML("beforeend", `<div class="verdict-line">${verdict}</div>`);
    const why = (c.conflict ?? 0) >= (c.camera ?? 0)
      ? `<b>traffic conflict</b> (${(c.conflict ?? 0).toFixed(2)}): actors pass close to each other in space and time`
      : `<b>camera perception</b> (${(c.camera ?? 0).toFixed(2)}): darkness, or a camera-only detector that is unsure`;
    setNow("curate", `<span class="now-k">Why this clip scores ${c.difficulty.toFixed(2)}</span>Mostly ${why}.`);
  }

  // ── chapter: Serve ──────────────────────────────────────────────────────
  function enterServe() {
    const v = $("#serve-video"), x = pick.cosmos, g = chap.serve.gate;
    const frame = v.closest(".frame");
    if (!x) {
      frame.style.display = "none";
      setNow("serve", `<span class="now-k">Augmentation</span>Cosmos-Transfer re-renders recorded windows as night, rain or fog; a hallucination gate keeps ${g.kept} of ${g.of}.`);
      return;
    }
    frame.style.display = "";
    playVideo(v, x.video);
    stops.push(() => v.pause());
    $("#serve-cond").textContent = `Cosmos-Transfer · ${x.cond}`;
    const det = x.det && x.det[0] != null ? ` · camera detector ${x.det[0].toFixed(1)} → <b>${x.det[1].toFixed(1)}</b> detections, confidence ${x.conf[0].toFixed(2)} → <b>${x.conf[1].toFixed(2)}</b>` : "";
    $("#serve-cap").innerHTML = `<b>Clip ${esc(x.short)}</b>${x.actors != null ? ` · ${x.actors} actors in the window` : ""}${det}`;
    setNow("serve", `<span class="now-k">Hallucination gate</span>A variant is served only if it invents no actors and is harder for a camera-only detector: <b>${g.kept} of ${g.of}</b> passed.`);
  }

  // ── chapter: Reconstruct (one video: real | twin | depth, wiped on a canvas)
  function enterReconstruct() {
    const t = pick.twin, v = $("#wipe-video"), cv = $("#wipe-canvas");
    C.buildBar($("#build-bar"), t && t.build, chap.reconstruct.steps);
    if (!t) {
      const ctx = cv.getContext("2d"), img = new Image();
      img.onload = () => { cv.width = img.width; cv.height = img.height; ctx.drawImage(img, 0, 0); };
      img.src = FALLBACK_FRAMES[1];
      $("#wipe-cap").textContent = "";
      setNow("reconstruct", `<span class="now-k">Twins</span>Nine clips reconstructed so far.`);
      return;
    }
    playVideo(v, t.video);
    stops.push(() => v.pause());
    const ctx = cv.getContext("2d");
    const t0 = performance.now(), split = (chap.reconstruct && D.durations.reconstruct * 0.55) * 1000;
    let raf = 0, phase = -1;
    const draw = now => {
      const r = cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (cv.width !== Math.round(r.width * dpr)) { cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
      const W = cv.width, H = cv.height, el = now - t0;
      const ph = el < split ? 0 : 1;               // 0: real | twin, 1: twin | depth
      if (ph !== phase) {
        phase = ph;
        $("#wipe-left").textContent = ph ? "Twin render" : "Real camera";
        $("#wipe-right").textContent = ph ? "Twin depth" : "Twin render";
      }
      if (v.readyState >= 2 && v.videoWidth) {
        const pw = v.videoWidth / 3, ph2 = v.videoHeight;
        const s = 0.5 + 0.3 * Math.sin((el / 1000) * (2 * Math.PI / 7) - Math.PI / 2);
        const L = ph ? pw : 0, R = ph ? 2 * pw : pw;
        ctx.drawImage(v, L, 0, pw * s, ph2, 0, 0, W * s, H);
        ctx.drawImage(v, R + pw * s, 0, pw * (1 - s), ph2, W * s, 0, W * (1 - s), H);
        ctx.fillStyle = "#38d6ff";
        ctx.fillRect(W * s - 1.5 * dpr, 0, 3 * dpr, H);
        ctx.beginPath(); ctx.arc(W * s, H / 2, 10 * dpr, 0, Math.PI * 2); ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    stops.push(() => cancelAnimationFrame(raf));
    const where = [t.country, t.season].filter(Boolean).join(" · ");
    $("#wipe-cap").innerHTML = `<b>Clip ${esc(t.short)}</b>${where ? ` · ${esc(where)}` : ""} · front-wide camera · PSNR <b>${t.psnr ?? "—"} dB</b>`;
    setNow("reconstruct", `<span class="now-k">Same viewpoint as the real camera</span>The twin also renders views the car never took, which is what lets a different driving policy take over the wheel. Next: that drive.`);
  }

  // ── chapter: Validate ───────────────────────────────────────────────────
  const OUT_WORD = { clean: "Clean", offroad: "Off-road", collision: "Collision" };
  const OUT_ICON = { clean: "✓", offroad: "!", collision: "×" };
  function verdictHtml(r) {
    const when = r.outcome === "clean" ? "" : ` · at ${r.t_end_s.toFixed(1)} s`;
    return `<span class="v-ico">${OUT_ICON[r.outcome]}</span><span class="v-out">${OUT_WORD[r.outcome]}</span><span class="v-det">${r.dist_m.toFixed(1)} m${when}</span>`;
  }
  function enterValidate() {
    const p = pick.pair, v = $("#pair-video");
    const vo = $("#verdict-ours"), vn = $("#verdict-nvidia");
    vo.className = "verdict left"; vn.className = "verdict right";
    if (!p) {
      v.closest(".frame").style.display = "none";
      setNow("validate", `<span class="now-k">Closed loop</span>Nine twins driven by VaVAM beside NVIDIA's own reconstructions.`);
      return;
    }
    v.closest(".frame").style.display = "";
    vo.innerHTML = verdictHtml(p.ours); vn.innerHTML = verdictHtml(p.nvidia);
    vo.classList.add(p.ours.outcome); vn.classList.add(p.nvidia.outcome);
    // the sim clock is the video clock: show each verdict at that side's scored end
    const onTime = () => {
      vo.classList.toggle("show", v.currentTime >= Math.max(0.3, p.ours.t_end_s));
      vn.classList.toggle("show", v.currentTime >= Math.max(0.3, p.nvidia.t_end_s));
    };
    let again = null;
    const onEnd = () => { again = setTimeout(() => { vo.classList.remove("show"); vn.classList.remove("show"); playVideo(v, p.video); }, 2200); };
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("ended", onEnd);
    playVideo(v, p.video);
    stops.push(() => { v.pause(); v.removeEventListener("timeupdate", onTime); v.removeEventListener("ended", onEnd); clearTimeout(again); });
    const where = [p.country, p.season].filter(Boolean).join(" · ");
    const agree = p.vavam_agree
      ? `Same outcome on both: <b>${OUT_WORD[p.ours.outcome].toLowerCase()}</b>.`
      : `Outcomes differ: ours <b>${OUT_WORD[p.ours.outcome].toLowerCase()}</b>, NVIDIA's <b>${OUT_WORD[p.nvidia.outcome].toLowerCase()}</b>.`;
    const note = (chap.validate.notes || {})[p.short];
    setNow("validate", `<span class="now-k">Clip ${esc(p.short)}${where ? " · " + esc(where) : ""} · VaVAM, recorded route</span>${agree}${note ? " " + esc(note) : ""}`);
  }

  // ── chapter: Triage ─────────────────────────────────────────────────────
  function enterTriage() {
    const tr = M.triage, budget = chap.triage.budget;
    const waffleEl = $("#waffle"), read = $("#waffle-readout");
    let curve = chap.triage.curve;
    if (tr && tr.clips && tr.clips.length) {
      const clips = tr.clips.slice().sort((a, b) => a.rank - b.rank);
      const cells = C.waffle(waffleEl, clips);
      const n = clips.length, pos = clips.reduce((a, c) => a + c.at_fault, 0);
      curve = [[0, 0]];
      let found = 0;
      clips.forEach((c, i) => { found += c.at_fault; curve.push([(i + 1) / n, found / pos]); });
      const k = Math.round(budget * n);
      const caught = clips.slice(0, k).reduce((a, c) => a + c.at_fault, 0);
      read.innerHTML = `
        <span><b id="tr-run">0</b>of ${n} rolled out</span>
        <span><b id="tr-found">0</b>of ${pos} at-fault failures found</span>
        <span class="keys">
          <span class="key"><i class="fail"></i>at-fault failure</span>
          <span class="key"><i></i>none</span>
          <span class="key"><i class="run"></i>rolled out</span>
        </span>`;
      let i = 0, f = 0;
      const iv = setInterval(() => {
        if (i >= k) { clearInterval(iv); return; }
        cells[i].classList.add("run"); f += clips[i].at_fault; i++;
        $("#tr-run").textContent = i; $("#tr-found").textContent = f;
      }, Math.max(40, 3600 / k));
      stops.push(() => { clearInterval(iv); cells.forEach(c => c.classList.remove("run")); });
      setNow("triage", `<span class="now-k">The dial at 50 %</span>Rolling out the ${k} clips the screen ranks riskiest finds <b>${caught} of ${pos}</b> at-fault failures; running all ${n} would take twice the GPU time.`);
    } else {
      waffleEl.closest(".panel").style.display = "none";
      setNow("triage", `<span class="now-k">The dial</span>At a 50 % budget the screen finds 66 % of at-fault failures.`);
    }
    requestAnimationFrame(() => C.recallChart($("#recall-chart"), curve, budget));
  }

  const ENTER = { collect: enterCollect, curate: enterCurate, serve: enterServe, reconstruct: enterReconstruct, validate: enterValidate, triage: enterTriage };

  // ── sequencer ───────────────────────────────────────────────────────────
  // ?ch=<chapter id> starts there, &hold=1 stays there, &speed=<n> runs n times faster
  // (previews, screenshots, soak tests)
  const params = new URLSearchParams(location.search);
  const HOLD = params.get("hold") === "1";
  const SPEED = Math.max(0.1, +params.get("speed") || 1);

  function go(i) {
    while (stops.length) { try { stops.pop()(); } catch (e) { /* keep looping */ } }
    if (i === 0) { if (cur !== -1) loop++; pickSamples(); }
    cur = i;
    const id = order[i], dur = (D.durations[id] || 12) * 1000 / SPEED;
    $$(".chapter").forEach(s => s.classList.toggle("on", s.dataset.ch === id));
    $$(".node").forEach((n, k) => {
      n.classList.toggle("active", k === i);
      n.classList.toggle("done", k < i);
      const bar = n.querySelector(".n-prog");
      bar.style.transition = "none"; bar.style.width = k < i ? "100%" : "0";
      if (k === i) { void bar.offsetWidth; bar.style.transition = `width ${dur}ms linear`; bar.style.width = "100%"; }
    });
    try { ENTER[id](); } catch (e) { console.error(e); }
    countUp(id);
    clearTimeout(timer);
    if (!HOLD) timer = setTimeout(() => go((i + 1) % order.length), dur);
    if (i === order.length - 1) prefetchCloud(library && library[((loop + 1) * 8) % library.length]);
  }

  // ── ambient particles (slow, sparse: signage, not a screensaver) ─────────
  function ambient() {
    const cv = $("#ambient"), ctx = cv.getContext("2d");
    let parts = [];
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const resize = () => {
      cv.width = innerWidth; cv.height = innerHeight;
      parts = Array.from({ length: reduce ? 0 : 46 }, () => ({
        x: Math.random() * cv.width, y: Math.random() * cv.height,
        v: 0.25 + Math.random() * 0.9, r: 0.6 + Math.random() * 1.5, a: 0.04 + Math.random() * 0.14 }));
    };
    resize(); addEventListener("resize", resize);
    (function frame() {
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.fillStyle = "#38d6ff";
      for (const p of parts) {
        p.x += p.v; if (p.x > cv.width + 4) { p.x = -4; p.y = Math.random() * cv.height; }
        ctx.globalAlpha = p.a; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      requestAnimationFrame(frame);
    })();
  }

  // ── boot ────────────────────────────────────────────────────────────────
  function loadLibrary() {
    const take = j => {
      if (!j || !Array.isArray(j.clips) || !j.clips.length) return;
      library = j.clips.slice();
      for (let i = library.length - 1; i > 0; i--) { const k = Math.floor(Math.random() * (i + 1)); [library[i], library[k]] = [library[k], library[i]]; }
    };
    if (window.DEMO_MANIFEST) { take(window.DEMO_MANIFEST); return Promise.resolve(); }
    return fetch("assets/clips/manifest.json", { cache: "no-store" })
      .then(r => (r.ok ? r.json() : null)).then(take).catch(() => {});
  }

  function start() {
    renderStatic();
    tick(); setInterval(tick, 1000);
    ambient();
    if (window.LidarScene) { lidar = new window.LidarScene($("#collect-lidar")); lidar.stop(); }
    // shuffle the twin order once per page load so restarts do not always open on the same clip
    if (M.twins && M.twins.length) {
      const k = Math.floor(Math.random() * M.twins.length);
      M.twins = M.twins.slice(k).concat(M.twins.slice(0, k));
    }
    loadLibrary().then(() => {
      const first = Math.max(0, order.indexOf(params.get("ch")));
      if (first) pickSamples();
      go(first);
    });
    // a kiosk runs for weeks: reload once a day to pick up rebuilt media and drop any leak
    setTimeout(() => location.reload(), 24 * 3600 * 1000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
