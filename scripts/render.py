#!/usr/bin/env python3
import json, os, shutil, pathlib, datetime, html

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
TEMPLATE = ROOT / "site.template.html"
CONFIG = ROOT / "config.json"

def esc(s): return html.escape(s or "")

def main():
    DIST.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

    name = cfg.get("name","")
    tagline = cfg.get("tagline","")
    about = cfg.get("about","")
    avatar = cfg.get("avatar","")
    links = cfg.get("links", [])
    theme = cfg.get("theme", {}) or {}
    year = str(datetime.datetime.utcnow().year)

    # Build links HTML
    links_html = []
    for item in links:
        label = esc(item.get("label",""))
        url   = item.get("url","")
        if not label or not url: continue
        links_html.append(
            f'<a class="btn" href="{html.escape(url)}"'
            f'{" target=\\"_blank\\"" if url.startswith("http") else ""}>'
            f'<span class="dot" aria-hidden="true"></span>{label}</a>'
        )
    links_html = "\n        ".join(links_html) if links_html else ""

    # Avatar (conditionally render or omit)
    avatar_html = ""
    if avatar:
        avatar_html = f'<img class="avatar" src="{html.escape(avatar)}" alt="{esc(name)} — profile photo" />'

    # Read template and replace tokens
    page = TEMPLATE.read_text(encoding="utf-8")
    page = (page
        .replace("[[NAME]]", esc(name))
        .replace("[[TAGLINE]]", esc(tagline))
        .replace("[[ABOUT]]", esc(about))
        .replace("[[YEAR]]", year)
        .replace("[[AVATAR_HTML]]", avatar_html)
        .replace("[[LINKS_HTML]]", links_html)
        .replace("[[THEME_BG]]", theme.get("bg","#0f1115"))
        .replace("[[THEME_FG]]", theme.get("fg","#d1d5db"))
        .replace("[[THEME_MUTED]]", theme.get("muted","#9ca3af"))
        .replace("[[THEME_ACCENT]]", theme.get("accent","#60a5fa"))
    )

    # Write index.html
    (DIST / "index.html").write_text(page, encoding="utf-8")

    # Copy static assets if present
    for rel in ["styles.css", "typing.js", "snippets.json", "code_manifest.json"]:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, DIST / src.name)

    # Copy directories if present
    for dirname in ["assets", "code_samples"]:
        srcdir = ROOT / dirname
        if srcdir.exists():
            dstdir = DIST / dirname
            if dstdir.exists():
                shutil.rmtree(dstdir)
            shutil.copytree(srcdir, dstdir)

    # Custom domain
    cdomain = (cfg.get("custom_domain") or "").strip()
    if cdomain:
        (DIST / "CNAME").write_text(cdomain + "\n", encoding="utf-8")

    print("Build complete → dist/")

if __name__ == "__main__":
    main()
