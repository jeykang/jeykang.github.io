// Types short lines from files listed in ./code_manifest.json (created by your harvester).
// Falls back to ./snippets.json if the manifest is missing.
// Page content is already pre-rendered; this script never alters the shell text.

(async function(){
  const stream = document.getElementById("code-stream");
  if (!stream) return;

  let typingEnabled = true;
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches){
    typingEnabled = false;
  }
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const rand  = (a,b) => Math.floor(Math.random()*(b-a+1))+a;

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

  async function loadJSON(path){
    try { const r = await fetch(path, { cache: "no-store" }); return r.ok ? r.json() : null; }
    catch { return null; }
  }
  function pickLinesFromText(text, n){
    const lines = text.split(/\r?\n/).map(s => s.trimEnd());
    const ok = lines.filter(s => s && s.length <= 120 && !/^[{}\[\]();,:]+$/.test(s.trim()));
    if (!ok.length) return [];
    const out = [];
    for (let i=0;i<n;i++) out.push(ok[Math.floor(Math.random()*ok.length)]);
    return out;
  }
  async function getRandomLinesFromManifest(manifest, n){
    if (!manifest || !Array.isArray(manifest.files) || !manifest.files.length) return null;
    const item = manifest.files[Math.floor(Math.random()*manifest.files.length)];
    try{
      const r = await fetch(item.file, { cache: "no-store" });
      if (!r.ok) return null;
      const text = await r.text();
      const lines = pickLinesFromText(text, Math.max(1, n));
      return lines.length ? lines : null;
    }catch { return null; }
  }

  const manifest = await loadJSON("code_manifest.json");
  const fallback = await loadJSON("snippets.json") || [];

  (async function loop(){
    while (true){
      let lines = await getRandomLinesFromManifest(manifest, rand(1,3));
      if (!lines || !lines.length){
        if (fallback.length){
          lines = [];
          for (let i=0;i<rand(1,3);i++){
            lines.push(fallback[Math.floor(Math.random()*fallback.length)]);
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
