// Load profile from config.json and populate the page.
(async function(){
  const el = (id) => document.getElementById(id);

  function setAccent(hex){
    if (!hex) return;
    document.documentElement.style.setProperty('--accent', hex);
  }

  function linkIsExternal(url){
    try { const u = new URL(url, window.location.href); return u.origin !== window.location.origin; }
    catch { return false; }
  }

  try{
    const res = await fetch('config.json', { cache: 'no-store' });
    const cfg = await res.json();

    if (cfg.accent) setAccent(cfg.accent);

    if (cfg.name){
      el('name').textContent = cfg.name;
      el('name-footer').textContent = cfg.name;
      document.title = cfg.name + ' — Home';
    }
    if (cfg.tagline) el('tagline').textContent = cfg.tagline;
    if (cfg.about) el('about').textContent = cfg.about;

    const avatar = el('avatar');
    if (cfg.avatar){
      avatar.src = cfg.avatar;
      avatar.alt = cfg.name ? cfg.name + ' — profile photo' : 'Profile photo';
    }else{
      avatar.style.display = 'none';
    }

    if (Array.isArray(cfg.links)){
      const links = document.getElementById('links');
      links.innerHTML = '';
      for (const item of cfg.links){
        if (!item || !item.label || !item.url) continue;
        const a = document.createElement('a');
        a.className = 'btn';
        a.href = item.url;
        a.textContent = item.label;
        if (linkIsExternal(item.url)) a.target = '_blank';
        // small accent dot
        const dot = document.createElement('span');
        dot.className = 'dot';
        a.prepend(dot);
        links.appendChild(a);
      }
    }

    document.getElementById('year').textContent = new Date().getFullYear();
  }catch(e){
    console.error('Failed to load config.json:', e);
    // Fail gracefully with defaults
    document.getElementById('year').textContent = new Date().getFullYear();
  }
})();