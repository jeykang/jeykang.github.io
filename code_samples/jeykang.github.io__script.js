// Dark profile with background typing from local harvested files.
// Prefers ./code_manifest.json (list of ./code_samples/*) produced by a GitHub Action.
// Falls back to ./snippets.json if no manifest present.

(async function(){
  const $ = (id) => document.getElementById(id);

  function setTheme(theme){
    if (!theme) return;
    for (const [k,v] of Object.entries(theme)){
      if (k && v) document.documentElement.style.setProperty(`--${k}`, v);
    }
  }

  function linkIsExternal(url){
    try { const u = new URL(url, window.location.href); return u.origin !== window.location.origin; }
    catch { return false; }
  }

  // Typing overlay controls
  const stream = document.getElementById("code-stream");
  let typingEnabled = true;
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches){
    typingEnabled = false;
  }
  function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }
  function rand(min, max){ return Math.floor(Math.random()*(max-min+1))+min; }
  const MAX_VISIBLE_LINES = 24;
  const MIN_CHAR_DELAY = 6;
  const MAX_CHAR_DELAY = 28;
  const LINE_DELAY = 250;

  function trimOverlay(){
    const lines = stream.innerText.split("\n");
    if (lines.length > MAX_VISIBLE_LINES){
      const excess = lines.length - MAX_VISIBLE_LINES;
      stream.textContent = lines.slice(excess).join("\n");
    }
  }

  async function typeLine(line){
    if (!typingEnabled){
      stream.textContent += (stream.textContent ? "\n" : "") + line;
      trimOverlay();
      return;
    }
    let buf = "";
    for (let i=0;i<line.length;i++){
      buf += line[i];
      const prefix = stream.textContent ? stream.textContent + "\n" : "";
      stream.textContent = prefix + buf + "▌";
      trimOverlay();
      await sleep(rand(MIN_CHAR_DELAY, MAX_CHAR_DELAY));
    }
    const prefix = stream.textContent ? stream.textContent.replace(/▌$/,"") : "";
    stream.textContent = prefix;
  }

  async function typeSession(lines){
    for (const line of lines){
      await typeLine(line);
      await sleep(LINE_DELAY);
    }
  }

  // Content population
  let cfg = {};
  try{
    const res = await fetch('config.json', { cache: 'no-store' });
    cfg = await res.json();
  }catch{}

  if (cfg.theme) setTheme(cfg.theme);
  if (cfg.name){
    $("name").textContent = cfg.name;
    $("name-footer").textContent = cfg.name;
    document.title = cfg.name + " — Home";
  }
  $("tagline").textContent = cfg.tagline || "";
  $("about").textContent = cfg.about || "";

  const avatar = $("avatar");
  if (cfg.avatar){
    avatar.src = cfg.avatar;
    avatar.alt = (cfg.name ? cfg.name + " — profile photo" : "Profile photo");
  }else{
    avatar.style.display = "none";
  }

  if (Array.isArray(cfg.links)){
    const links = $("links");
    links.innerHTML = '';
    for (const item of cfg.links){
      if (!item || !item.label || !item.url) continue;
      const a = document.createElement('a');
      a.className = 'btn';
      a.href = item.url;
      a.textContent = item.label;
      if (linkIsExternal(item.url)) a.target = '_blank';
      const dot = document.createElement('span');
      dot.className = 'dot';
      a.prepend(dot);
      links.appendChild(a);
    }
  }
  $("year").textContent = new Date().getFullYear();

  // === Typing source: code_manifest.json -> random file lines ===
  async function loadManifest(){
    try{
      const r = await fetch('code_manifest.json', { cache: 'no-store' });
      if (!r.ok) return null;
      return await r.json();
    }catch { return null; }
  }

  function pickLinesFromText(text, maxCount){
    const lines = text.split(/\r?\n/).map(s => s.trimEnd());
    const ok = lines.filter(s => s && s.length <= 120 && !/^[{}\[\]();,:]+$/.test(s.trim()));
    if (!ok.length) return [];
    const chosen = [];
    for (let i=0;i<maxCount;i++){
      chosen.push(ok[Math.floor(Math.random()*ok.length)]);
    }
    return chosen;
  }

  async function getRandomLinesFromFiles(manifest, maxLines){
    if (!manifest || !Array.isArray(manifest.files) || !manifest.files.length) return null;
    const item = manifest.files[Math.floor(Math.random()*manifest.files.length)];
    const path = item.file;
    try{
      const r = await fetch(path, { cache: 'no-store' });
      if (!r.ok) return null;
      const text = await r.text();
      const lines = pickLinesFromText(text, Math.max(1, maxLines));
      return lines.length ? lines : null;
    }catch{
      return null;
    }
  }

  async function loadFallbackSnippets(){
    try{
      const r = await fetch('snippets.json', { cache: 'no-store' });
      if (!r.ok) return [];
      const arr = await r.json();
      return Array.isArray(arr) ? arr : [];
    }catch{ return []; }
  }

  const manifest = await loadManifest();
  const fallbackSnippets = await loadFallbackSnippets();

  (async function loop(){
    while (true){
      let lines = null;
      if (manifest){
        lines = await getRandomLinesFromFiles(manifest, Math.floor(Math.random()*3)+1);
      }
      if (!lines || !lines.length){
        if (fallbackSnippets.length){
          lines = [];
          const count = Math.max(1, Math.floor(Math.random()*3)+1);
          for (let i=0;i<count;i++){
            lines.push(fallbackSnippets[Math.floor(Math.random()*fallbackSnippets.length)]);
          }
        }else{
          lines = ["// typing source not available"];
        }
      }
      await typeSession(lines);
      await sleep(Math.max(300, Math.floor(Math.random()*1500)));
    }
  })();

})();