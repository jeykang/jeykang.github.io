#!/usr/bin/env python3
"""
Harvest a small set of code files from a user's public GitHub repositories and
write them into ./code_samples/, plus a manifest ./code_manifest.json that the
frontend can read. Intended to run inside GitHub Actions on a schedule.

Environment variables:
  GITHUB_TOKEN        (provided automatically by Actions; increases the rate limit)
  GH_USERNAME         (required) GitHub username to scan
  EXTENSIONS          Comma-separated whitelist of file extensions (default below)
  MAX_REPOS           Max recent repos to scan (default: 8)
  NUM_FILES           Total files to download (default: 16)
  INCLUDE_FORKS       'true' to include forks (default: false)
  EXCLUDE_DIRS        Comma-separated path substrings to ignore (default below)
  MAX_BYTES_PER_FILE  Hard limit for saved file size (default: 65536 bytes)
  REPO_ALLOWLIST      Comma-separated substrings; only repos containing any substring are included
  REPO_BLOCKLIST      Comma-separated substrings; repos containing any substring are excluded
"""
import os, sys, json, base64, random, re, time, pathlib
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

USERNAME         = os.getenv("GH_USERNAME", "").strip()
EXTENSIONS       = [e.strip().lower() for e in os.getenv(
                     "EXTENSIONS",
                     "py,js,ts,tsx,jsx,c,cpp,h,hpp,sh,rs,go,java,kt,md,txt,html,css,yaml,yml,toml,ini,json"
                   ).split(",") if e.strip()]
MAX_REPOS        = int(os.getenv("MAX_REPOS", "8"))
NUM_FILES        = int(os.getenv("NUM_FILES", "16"))
INCLUDE_FORKS    = os.getenv("INCLUDE_FORKS", "false").lower() == "true"
EXCLUDE_DIRS     = [s.strip() for s in os.getenv(
                     "EXCLUDE_DIRS",
                     ".git,node_modules,dist,build,venv,.venv,site-packages,__pycache__,.next,out,coverage,.cache"
                   ).split(",") if s.strip()]
MAX_BYTES_PER_FILE = int(os.getenv("MAX_BYTES_PER_FILE", "65536"))
ALLOWLIST        = [s.strip() for s in os.getenv("REPO_ALLOWLIST", "").split(",") if s.strip()]
BLOCKLIST        = [s.strip() for s in os.getenv("REPO_BLOCKLIST", "").split(",") if s.strip()]

ROOT             = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_DIR       = ROOT / "code_samples"
MANIFEST_PATH    = ROOT / "code_manifest.json"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

def api_get(url: str, params: Dict[str, Any] = None) -> Any:
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def list_user_repos(user: str) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/users/{user}/repos"
    repos = api_get(url, params={"per_page": 100, "type": "owner", "sort": "updated"})
    if not isinstance(repos, list):
        return []
    def ok_repo(r):
        if not INCLUDE_FORKS and r.get("fork"):
            return False
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
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}"
    j = api_get(url, params={"recursive": "1"})
    return j.get("tree", []) if isinstance(j, dict) else []

def allowed_path(path: str) -> bool:
    if not path: return False
    lp = path.lower()
    for ex in EXCLUDE_DIRS:
        if ex and ex.lower() in lp:
            return False
    m = re.search(r"\.([A-Za-z0-9]+)$", path)
    if not m: return False
    ext = m.group(1).lower()
    return ext in EXTENSIONS

def is_text_bytes(b: bytes) -> bool:
    if b.find(b"\x00") != -1:
        return False
    sample = b[:4096]
    nonprint = sum(ch < 9 or (13 < ch < 32) for ch in sample)
    return (nonprint / max(1, len(sample))) < 0.02

def sanitize_filename(repo: str, path: str) -> str:
    import re as _re
    safe_repo = _re.sub(r"[^A-Za-z0-9._-]+", "-", repo)
    safe_path = _re.sub(r"[^A-Za-z0-9._/-]+", "-", path).replace("/", "__")
    return f"{safe_repo}__{safe_path}"

def get_file_content(owner: str, repo: str, path: str, ref: str) -> bytes:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    j = api_get(url, params={"ref": ref})
    if isinstance(j, list):
        return b""
    enc = j.get("encoding")
    if enc == "base64" and "content" in j:
        try:
            import base64 as _b64
            return _b64.b64decode(j["content"])
        except Exception:
            return b""
    download_url = j.get("download_url")
    if download_url:
        r = SESSION.get(download_url, timeout=30)
        if r.ok:
            return r.content
    return b""

def main() -> int:
    if not USERNAME:
        print("GH_USERNAME is required", file=sys.stderr)
        return 2

    repos = list_user_repos(USERNAME)
    candidates = []
    for r in repos:
        owner = r.get("owner", {}).get("login", USERNAME)
        name  = r.get("name")
        ref   = r.get("default_branch") or "main"
        if not name: continue
        tree = get_tree(owner, name, ref)
        for t in tree:
            if t.get("type") != "blob":
                continue
            p = t.get("path","")
            if allowed_path(p):
                candidates.append((owner, name, ref, p))

    if not candidates:
        print("No candidate files matched filters.", file=sys.stderr)

    random.shuffle(candidates)
    picked = candidates[:NUM_FILES]

    files_meta = []
    for owner, name, ref, path in picked:
        try:
            content = get_file_content(owner, name, path, ref)
        except requests.HTTPError:
            continue
        if not content:
            continue
        if len(content) > MAX_BYTES_PER_FILE:
            content = content[:MAX_BYTES_PER_FILE]
        if not is_text_bytes(content):
            continue

        out_name = sanitize_filename(name, path)
        out_path = SAMPLE_DIR / out_name
        out_path.write_bytes(content)
        files_meta.append({
            "file": f"code_samples/{out_name}",
            "repo": name,
            "owner": owner,
            "path": path,
            "branch": ref,
            "bytes": len(content)
        })

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "user": USERNAME,
            "max_repos": MAX_REPOS,
            "num_files": NUM_FILES,
            "include_forks": INCLUDE_FORKS,
            "extensions": EXTENSIONS,
            "exclude_dirs": EXCLUDE_DIRS
        },
        "files": files_meta
    }

    old_text = None
    if MANIFEST_PATH.exists():
        try:
            old_text = MANIFEST_PATH.read_text(encoding="utf-8")
        except Exception:
            old_text = None
    new_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if old_text != new_text:
        MANIFEST_PATH.write_text(new_text, encoding="utf-8")
        print(f"Wrote manifest with {len(files_meta)} files.")
    else:
        print("Manifest unchanged.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
