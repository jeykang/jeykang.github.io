#!/usr/bin/env python3
"""
Harvest random readable code lines from a user's public GitHub repositories,
then write them to ./snippets.json at the repo root.

Environment variables:
  GITHUB_TOKEN        (provided automatically by Actions; increases the rate limit)
  GH_USERNAME         (required) GitHub username to scan
  EXTENSIONS          Comma-separated whitelist of file extensions (default given below)
  MAX_REPOS           Max recent repos to sample (default: 5)
  FILES_PER_REPO      Files to sample per repo (default: 2)
  LINES_PER_FILE      Lines to sample per file (default: 16)
  MAX_LINES_TOTAL     Upper bound on total lines written (default: 250)
  REPO_ALLOWLIST      Comma-separated substrings; only repos containing any substring are included
  REPO_BLOCKLIST      Comma-separated substrings; repos containing any substring are excluded
"""

import os, sys, json, base64, random, re
from typing import List, Dict, Any
import requests

SESSION = requests.Session()
API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
TOKEN = os.getenv("GITHUB_TOKEN")
if TOKEN:
    API_HEADERS["Authorization"] = f"Bearer {TOKEN}"
SESSION.headers.update(API_HEADERS)

USERNAME        = os.getenv("GH_USERNAME", "").strip()
EXTENSIONS      = [e.strip().lower() for e in os.getenv(
                   "EXTENSIONS",
                   "py,js,ts,tsx,jsx,c,cpp,h,hpp,sh,rs,go,java,kt,md,txt,html,css,yaml,toml,ini,json"
                 ).split(",") if e.strip()]
MAX_REPOS       = int(os.getenv("MAX_REPOS", "5"))
FILES_PER_REPO  = int(os.getenv("FILES_PER_REPO", "2"))
LINES_PER_FILE  = int(os.getenv("LINES_PER_FILE", "16"))
MAX_LINES_TOTAL = int(os.getenv("MAX_LINES_TOTAL", "250"))
ALLOWLIST       = [s.strip() for s in os.getenv("REPO_ALLOWLIST", "").split(",") if s.strip()]
BLOCKLIST       = [s.strip() for s in os.getenv("REPO_BLOCKLIST", "").split(",") if s.strip()]

OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "..", "snippets.json")
OUTPUT_PATH     = os.path.abspath(OUTPUT_PATH)

def api_get(url: str, params: Dict[str, Any] = None) -> Any:
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def list_user_repos(user: str) -> List[Dict[str, Any]]:
    # Most recently updated public repos (owned by user)
    url = f"https://api.github.com/users/{user}/repos"
    repos = api_get(url, params={"per_page": 100, "type": "owner", "sort": "updated"})
    if not isinstance(repos, list):
        return []
    def ok_repo(r):
        name = (r.get("name") or "").lower()
        if BLOCKLIST and any(b.lower() in name for b in BLOCKLIST):
            return False
        if ALLOWLIST and not any(a.lower() in name for a in ALLOWLIST):
            return False
        return True
    repos = [r for r in repos if ok_repo(r)]
    repos.sort(key=lambda r: r.get("pushed_at") or r.get("updated_at") or "", reverse=True)
    return repos[:MAX_REPOS]

def get_tree(owner: str, repo: str, ref: str) -> List[Dict[str, Any]]:
    # Ask for the full recursive tree for default branch
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}"
    j = api_get(url, params={"recursive": "1"})
    return j.get("tree", []) if isinstance(j, dict) else []

def get_file_content(owner: str, repo: str, path: str, ref: str) -> str:
    # Use the contents API (token-authenticated) to fetch base64 content
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    j = api_get(url, params={"ref": ref})
    if isinstance(j, list):
        return ""
    enc = j.get("encoding")
    if enc == "base64" and "content" in j:
        try:
            data = base64.b64decode(j["content"])
            return data.decode("utf-8", "ignore")
        except Exception:
            return ""
    # Sometimes GitHub returns a redirect URL in 'download_url' for text
    download_url = j.get("download_url")
    if download_url:
        r = SESSION.get(download_url, timeout=30)
        if r.ok:
            try:
                return r.content.decode("utf-8", "ignore")
            except Exception:
                return ""
    return ""

def looks_texty_line(s: str) -> bool:
    if not s: return False
    if len(s) > 120: return False
    if s.strip() == "": return False
    # Filter super-noisy lines (pure braces, long comments, etc.)
    if re.fullmatch(r"[{}\[\]();,:]+", s.strip()):
        return False
    return True

def pick_from_file(text: str, limit: int) -> List[str]:
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    candidates = [ln for ln in lines if looks_texty_line(ln)]
    if not candidates: return []
    random.shuffle(candidates)
    return candidates[:limit]

def main() -> int:
    if not USERNAME:
        print("GH_USERNAME is required", file=sys.stderr)
        return 2

    repos = list_user_repos(USERNAME)
    pool: List[str] = []
    for r in repos:
        if len(pool) >= MAX_LINES_TOTAL:
            break
        owner = r.get("owner", {}).get("login", USERNAME)
        name  = r.get("name")
        ref   = r.get("default_branch") or "main"
        if not name: continue

        tree = get_tree(owner, name, ref)
        files = [t for t in tree if t.get("type") == "blob"]
        # extension filter
        def allowed_path(p):
            m = re.search(r"\.([A-Za-z0-9]+)$", p)
            return bool(m and m.group(1).lower() in EXTENSIONS)
        files = [f for f in files if allowed_path(f.get("path",""))]
        if not files:
            continue

        random.shuffle(files)
        files = files[:FILES_PER_REPO]

        for f in files:
            if len(pool) >= MAX_LINES_TOTAL:
                break
            path = f.get("path")
            if not path: continue
            try:
                text = get_file_content(owner, name, path, ref)
            except requests.HTTPError:
                continue
            if not text:
                continue
            chosen = pick_from_file(text, LINES_PER_FILE)
            for line in chosen:
                pool.append(line)
                if len(pool) >= MAX_LINES_TOTAL:
                    break

    # De-duplicate but keep relative order
    seen = set()
    unique = []
    for ln in pool:
        if ln not in seen:
            seen.add(ln)
            unique.append(ln)
    pool = unique[:MAX_LINES_TOTAL]

    # Fallback if nothing found
    if not pool:
        pool = [
            "def hello(name):",
            "    return f\"Hello, {name}!\"",
            "if __name__ == '__main__':",
            "    print('Ready.')",
        ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(pool)} lines to {OUTPUT_PATH}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
