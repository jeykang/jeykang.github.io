#!/usr/bin/env python3
import json, os, shutil, pathlib, datetime, html

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
TEMPLATE = ROOT / "site.template.html"
CONFIG = ROOT / "config.json"

def esc(s):  # HTML-escape helper for text nodes
    return html.escape(s or "")

def main():
    DIST.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

    name = cfg.get("name", "")
    tagline = cfg.get("tagline", "")
    about = cfg.get("about", "")
    avatar = cfg.get("avatar", "")
    links = cfg.get("links", []) or []
    theme = cfg.get("theme", {}) or {}
    year = str(datetime.datetime.utcnow().year)

    # Build links HTML
    links_html_parts = []
    for item in links:
        label = esc(item.get("label", ""))
        url = item.get("url", "")
        if not label or not url:
            continue
        # External links open in a new tab; no escaping needed on the attribute itself
        is_external = url.startswith(("http://", "https://"))
        target_attr = ' target="_blank" rel="noopener"' if is_external else ""
        link_html = (
            f'<a class="btn" href="{html.escape(url)}"{target_attr}>'
            f'<span class="dot" aria-hidden="true"></span>{label}</a>'
        )
        links_html_parts.append(link_html)
    links_html = "\n        ".join(links_html_parts)

    # Avatar (optional)
    avatar_html = (
        f'<img class="avatar" src="{html.escape(avatar)}" alt="{esc(name)} — profile photo" />'
        if avatar else ""
    )

    # Apply tokens to the template
    page = TEMPLATE.read_text(encoding="utf-8")
    page = (
        page
        .replace("[[NAME]]", esc(name))
        .replace("[[TAGLINE]]", esc(tagline))
        .replace("[[ABOUT]]", esc(about))
        .replace("[[YEAR]]", year)
        .replace("[[AVATAR_HTML]]", avatar_html)
        .replace("[[LINKS_HTML]]", links_html)
        .replace("[[THEME_BG]]", theme.get("bg", "#0f1115"))
        .replace("[[THEME_FG]]", theme.get("fg", "#d1d5db"))
        .replace("[[THEME_MUTED]]", theme.get("muted", "#9ca3af"))
        .replace("[[THEME_ACCENT]]", theme.get("accent", "#60a5fa"))
    )

    # Write pre-rendered page
    (DIST / "index.html").write_text(page, encoding="utf-8")

    # Copy static assets (only if they exist)
    for rel in ["styles.css", "typing.js", "snippets.json", "code_manifest.json"]:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, DIST / src.name)

    # Copy directories (assets, harvested code files)
    for dirname in ["assets", "code_samples"]:
        srcdir = ROOT / dirname
        if srcdir.exists():
            dstdir = DIST / dirname
            if dstdir.exists():
                shutil.rmtree(dstdir)
            shutil.copytree(srcdir, dstdir)

    # Custom domain (optional)
    custom_domain = (cfg.get("custom_domain") or "").strip()
    if custom_domain:
        (DIST / "CNAME").write_text(custom_domain + "\n", encoding="utf-8")

    print("Build complete → dist/")

if __name__ == "__main__":
    main()
