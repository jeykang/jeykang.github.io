#!/usr/bin/env python3
import json, os, shutil, pathlib, datetime, html

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
TEMPLATE = ROOT / "site.template.html"
CONFIG = ROOT / "config.json"
CV_MD = ROOT / "cv.md"

def esc(s):
    return html.escape(s or "")

def read_text(path):
    return path.read_text(encoding="utf-8")

def write_text(path, text):
    path.write_text(text, encoding="utf-8")

def render_markdown(md_text: str) -> str:
    """
    Convert Markdown to HTML with proper nested list handling.
    Requires 'markdown' package (pip install markdown).
    """
    try:
        import markdown
        extensions = [
            "extra",       # tables, fenced_code, footnotes, etc.
            "sane_lists",  # predictable list behavior (GitHub-like)
            "toc",
        ]
        extension_configs = {
            "toc": {"permalink": False}
        }
        return markdown.markdown(
            md_text,
            extensions=extensions,
            extension_configs=extension_configs,
            output_format="html5"
        )
    except Exception:
        # Fallback: show raw Markdown as preformatted text
        return "<pre>" + html.escape(md_text) + "</pre>"


def build_cv_html(theme: dict, site_title: str) -> None:
    if not CV_MD.exists():
        return
    md = read_text(CV_MD)
    body_html = render_markdown(md)

    bg  = theme.get("bg", "#0f1115")
    fg  = theme.get("fg", "#d1d5db")
    mut = theme.get("muted", "#9ca3af")
    acc = theme.get("accent", "#60a5fa")

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{esc(site_title)} — CV</title>
  <meta name="description" content="Curriculum Vitae">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@300;400;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="styles.css">
  <style>
    :root{{ --bg:{bg}; --fg:{fg}; --muted:{mut}; --accent:{acc}; }}
    .prose code, .prose pre{{font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;}}
    .prose h1, .prose h2, .prose h3, .prose h4, .prose h5, .prose h6{{color: var(--fg); margin-top: 1.25em;}}
    .prose p, .prose li{{color: var(--fg); line-height: 1.7;}}
    .prose a{{color: var(--accent); text-decoration: none;}}
    .prose a:hover{{text-decoration: underline;}}
    .prose pre{{background: rgba(255,255,255,0.06); padding: 12px; border-radius: 10px; overflow:auto;}}
    .prose table{{border-collapse: collapse; width:100%;}}
    .prose th, .prose td{{border:1px solid rgba(255,255,255,0.1); padding:8px;}}
  </style>
</head>
<body>
  <main class="container">
    <section class="panel" role="region" aria-label="CV">
      <article class="prose">
        {body_html}
      </article>
      <footer class="footer"><small>&copy; {datetime.datetime.utcnow().year} {esc(site_title)}</small></footer>
    </section>
  </main>
</body>
</html>
"""
    write_text(DIST / "cv.html", page)

def main():
    DIST.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(read_text(CONFIG))
    name = cfg.get("name", "")
    tagline = cfg.get("tagline", "")
    about = cfg.get("about", "")
    avatar = cfg.get("avatar", "")
    links = cfg.get("links", []) or []
    theme = cfg.get("theme", {}) or {}
    year = str(datetime.datetime.utcnow().year)

    # Build link buttons
    parts = []
    for item in links:
        label = esc(item.get("label", ""))
        url = item.get("url", "")
        if not label or not url:
            continue
        is_external = url.startswith(("http://", "https://"))
        target_attr = ' target="_blank" rel="noopener"' if is_external else ""
        link_html = (
            f'<a class="btn" href="{html.escape(url)}"{target_attr}>'
            f'<span class="dot" aria-hidden="true"></span>{label}</a>'
        )
        parts.append(link_html)
    links_html = "\n        ".join(parts)

    # Optional avatar
    avatar_html = (
        f'<img class="avatar" src="{html.escape(avatar)}" alt="{esc(name)} — profile photo" />'
        if avatar else ""
    )

    # Fill main template
    page = read_text(TEMPLATE)
    page = (
        page.replace("[[NAME]]", esc(name))
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

    # Write pre-rendered homepage
    write_text(DIST / "index.html", page)

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

    # Render cv.md -> cv.html if available
    build_cv_html(theme, name)

    # Optional custom domain
    custom_domain = (cfg.get("custom_domain") or "").strip()
    if custom_domain:
        write_text(DIST / "CNAME", custom_domain + "\n")

    print("Build complete → dist/")

if __name__ == "__main__":
    main()
