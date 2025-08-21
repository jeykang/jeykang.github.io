// Dark profile page + background typing overlay.
// Config-driven; will attempt to pull random lines from GitHub public repos
// if a username is provided in config.json. Falls back to local snippets.json.

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

  // Typing overlay
  const stream = $("code-stream");
  let typingEnabled = true;
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches){
    typingEnabled = false;
  }

  function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }
  function rand(min, max){ return Math.floor(Math.random()*(max-min+1))+min; }

  const MAX_VISIBLE_LINES = 24;
  const MIN_CHAR_DELAY = 6;   // ms
  const MAX_CHAR_DELAY = 28;  // ms
  const LINE_DELAY = 250;     // ms between lines

  function trimOverlay(){
    const lines = stream.innerText.split("\n");
    if (lines.length > MAX_VISIBLE_LINES){
      const excess = lines.length - MAX_VISIBLE_LINES;
      stream.textContent = lines.slice(excess).join("\n");
    }
  }

  async function typeLine(line){
    if (!typingEnabled){
      // No animation: append quickly
      stream.textContent += (stream.textContent ? "\n" : "") + line;
      trimOverlay();
      return;
    }
    // Typewriter effect
    let buf = "";
    for (let i=0;i<line.length;i++){
      buf += line[i];
      // Use textContent to avoid HTML injection; overlay is inert anyway
      const prefix = stream.textContent ? stream.textContent + "\n" : "";
      stream.textContent = prefix + buf + "▌";
      trimOverlay();
      await sleep(rand(MIN_CHAR_DELAY, MAX_CHAR_DELAY));
    }
    // replace cursor
    const prefix = stream.textContent ? stream.textContent.replace(/▌$/,"") : "";
    stream.textContent = prefix;
  }

  async function typeSession(lines){
    for (const line of lines){
      await typeLine(line);
      await sleep(LINE_DELAY);
    }
  }

  // Load config
  let cfg = {};
  try{
    const res = await fetch('config.json', { cache: 'no-store' });
    cfg = await res.json();
  }catch(e){
    console.error("Failed to load config.json", e);
  }

  // Theme + content
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

  // Build a snippet provider (GitHub → fallback to local snippets.json)
  const extAllow = (cfg.github && Array.isArray(cfg.github.extensions)) ? cfg.github.extensions :
    ["py","js","ts","tsx","jsx","c","cpp","cc","h","hpp","sh","rb","rs","go","java","kt","scala","swift","m","mm","lua","pl","sql","yaml","toml","ini","json","md","txt","html","css"];

  async function getGitHubSnippets(){
    if (!cfg.github || !cfg.github.username) return [];
    const user = cfg.github.username;
    const perRepo = Math.min(cfg.github.filesPerRepo || 2, 6);
    const maxRepos = Math.min(cfg.github.maxRepos || 5, 15);
    const includeForks = !!cfg.github.includeForks;

    try{
      const repoRes = await fetch(`https://api.github.com/users/${encodeURIComponent(user)}/repos?per_page=100&type=owner&sort=updated`);
      const repos = await repoRes.json();
      if (!Array.isArray(repos)) return [];

      // Pick a subset of repos
      const candidates = repos.filter(r => includeForks || !r.fork);
      candidates.sort((a,b) => new Date(b.pushed_at) - new Date(a.pushed_at));
      const chosen = candidates.slice(0, maxRepos);

      const lines = [];

      // For each repo, fetch tree and sample a few files
      for (const r of chosen){
        const branch = r.default_branch || "main";
        const treeRes = await fetch(`https://api.github.com/repos/${encodeURIComponent(user)}/${encodeURIComponent(r.name)}/git/trees/${encodeURIComponent(branch)}?recursive=1`);
        const tree = await treeRes.json();
        if (!tree || !Array.isArray(tree.tree)) continue;
        const files = tree.tree
          .filter(entry => entry.type === "blob")
          .filter(entry => {
            const m = entry.path.match(/\.([a-zA-Z0-9]+)$/);
            if (!m) return false;
            return extAllow.includes(m[1].toLowerCase());
          });

        // Randomly sample up to perRepo files
        for (let i=0;i<Math.min(perRepo, files.length);i++){
          const f = files[Math.floor(Math.random()*files.length)];
          // Fetch raw content
          const rawUrl = `https://raw.githubusercontent.com/${encodeURIComponent(user)}/${encodeURIComponent(r.name)}/${encodeURIComponent(branch)}/${f.path}`;
          try{
            const rawRes = await fetch(rawUrl);
            if (!rawRes.ok) continue;
            const text = await rawRes.text();
            const fileLines = text.split(/\r?\n/).map(s => s.trimEnd());
            // pick some non-empty, reasonably short lines
            const ok = fileLines.filter(s => s && s.length <= 120);
            // add up to 8 random lines per file
            for (let k=0;k<Math.min(8, ok.length);k++){
              const L = ok[Math.floor(Math.random()*ok.length)];
              lines.push(L);
            }
          }catch {}
        }
      }
      return lines;
    }catch(e){
      console.warn("GitHub snippet load failed:", e);
      return [];
    }
  }

  async function getLocalSnippets(){
    try{
      const res = await fetch('snippets.json', { cache: 'no-store' });
      const arr = await res.json();
      if (Array.isArray(arr)) return arr;
      return [];
    }catch{
      return [];
    }
  }

  const ghLines = await getGitHubSnippets();
  const localLines = await getLocalSnippets();
  let pool = ghLines.length ? ghLines : localLines;

  if (!pool.length){
    pool = [
      "def hello(name):",
      "    return f\"Hello, {name}!\"",
      "",
      "class Model(nn.Module):",
      "    def forward(self, x):",
      "        return self.net(x)",
      "",
      "if __name__ == '__main__':",
      "    print('Ready.')"
    ];
  }

  // Start a continuous typing session with random lines
  (async function loop(){
    while (true){
      // pick 1–3 lines per "burst"
      const count = Math.max(1, Math.floor(Math.random()*3)+1);
      const lines = [];
      for (let i=0;i<count;i++){
        lines.push(pool[Math.floor(Math.random()*pool.length)]);
      }
      await typeSession(lines);
      await sleep(Math.max(300, Math.floor(Math.random()*1500)));
    }
  })();

})();