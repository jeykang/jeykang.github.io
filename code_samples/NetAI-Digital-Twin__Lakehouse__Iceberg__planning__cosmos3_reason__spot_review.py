#!/usr/bin/env python3
"""Build a BLIND human spot-review set for the cosmos3 disagreement pocket.

Why this exists: every number in RESULTS.md is measured against `ood_reasoning`
labels, which are Alpamayo-lineage. The out-of-family control (Gemma 4) weakened the
circularity worry but could not remove it, because it still scores against the same
labels. Human judgement is the one anchor that sidesteps the label set entirely.

Design — the contrast that matters:
  A  c3hard/cfeasy (n=25)  the disputed pocket: cosmos3 says hard, conflict says easy
  B  c3easy/cfeasy (n=25)  control: BOTH say easy
Both strata are conflict-easy, so any human-rated difference between A and B is
attributable to cosmos3 alone. Reviewing A by itself would be uninterpretable — there
would be no baseline to compare a "hard" rating against.

  + 5 c3hard/cfhard  (both-hard anchor: if these do not rate hardest, the rating
                      scale itself is not working and the run is void)
  + 5 c3easy/cfhard  (reverse pocket: conflict flags, cosmos3 does not)

60 clips, shuffled with a fixed seed, presented with NO scores, NO rationales and NO
labels — the model's rationale in particular would anchor the reviewer to the thing
under test. The answer key is written separately and is gitignored.

Usage:  ./c3_venv/bin/python spot_review.py [--out review.html]
Then:   ./c3_venv/bin/python score_review.py <ratings.json>
"""
import argparse
import base64
import io
import json
import os
import random

import numpy as np
from PIL import Image

import analyze_axis as A
import cosmos3_scorer as cs

_HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(_HERE, ".gate_cosmos3-edge_reasoned_452.json")
KEY = os.path.join(_HERE, ".review_key.json")
SEED = 0
STRATA = [("A_c3hard_cfeasy", 25), ("B_c3easy_cfeasy", 25),
          ("anchor_bothhard", 5), ("rev_cfhard_c3easy", 5)]
TILE_W = 420
JPEG_Q = 72


def stratify():
    rows = [r for r in json.load(open(GATE)) if r.get("score") is not None]
    conflict = A.load_axis(".conflict", "conflict_score")
    camera = A.load_axis(".camera_perception", "low_conf")
    dark = A.load_darkness()
    rows = [r for r in rows if r["clip"] in conflict and r["clip"] in camera
            and r["clip"] in dark]
    c3 = A.rank_norm([r["score"] for r in rows])
    cf = A.rank_norm([conflict[r["clip"]] for r in rows])
    buckets = {k: [] for k, _ in STRATA}
    for r, a, b in zip(rows, c3, cf):
        rec = {"clip": r["clip"], "ood": r["ood"], "c3": round(a, 3),
               "cf": round(b, 3), "c3_raw": round(r["score"], 4),
               "rationale": r.get("rationale", "")}
        if a > 0.75 and b < 0.25:
            buckets["A_c3hard_cfeasy"].append(rec)
        elif a < 0.25 and b < 0.25:
            buckets["B_c3easy_cfeasy"].append(rec)
        elif a > 0.75 and b > 0.75:
            buckets["anchor_bothhard"].append(rec)
        elif a < 0.25 and b > 0.75:
            buckets["rev_cfhard_c3easy"].append(rec)
    rng = random.Random(SEED)
    out = []
    for name, n in STRATA:
        pool = sorted(buckets[name], key=lambda r: r["clip"])
        rng.shuffle(pool)
        for rec in pool[:n]:
            out.append({**rec, "stratum": name})
    rng.shuffle(out)
    return out


def contact_sheet(clip_id):
    """2x2 grid of the same 4 frames the scorer saw."""
    frames = cs.load_frames(clip_id)
    tiles = []
    for f in frames[:4]:
        im = Image.fromarray(f)
        w, h = im.size
        tiles.append(im.resize((TILE_W, int(h * TILE_W / w)), Image.LANCZOS))
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (tw * 2, th * 2), (18, 18, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * tw, (i // 2) * th))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=JPEG_Q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "review.html"))
    a = ap.parse_args()

    items = stratify()
    print(f"[review] {len(items)} clips; building contact sheets ...")
    payload = []
    total = 0
    for i, rec in enumerate(items):
        try:
            b64 = contact_sheet(rec["clip"])
        except Exception as e:
            print(f"  [WARN] {rec['clip'][:8]}: {str(e)[:70]}")
            continue
        total += len(b64)
        payload.append({"i": len(payload), "clip": rec["clip"], "img": b64})
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(items)} ({total/1e6:.1f} MB b64)")

    json.dump(items, open(KEY, "w"), indent=1)
    print(f"[review] answer key -> {KEY} (gitignored; do NOT open before rating)")

    html = build_html(payload)
    open(a.out, "w").write(html)
    print(f"[review] {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB, "
          f"{len(payload)} clips)")


def build_html(payload):
    return HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))


HTML = r"""<title>Blind clip difficulty review</title>
<style>
/* Instrument, not a document. Two constraints come from the task:
   (1) the chrome must not bias the rating it collects — hence a calm indigo
       accent rather than an amber/red one that would prime "hazard";
   (2) ~half these clips are night scenes, so the image sits on a CONSTANT dark
       matte in both themes — a light page ground would wash out a dark scene
       and change its perceived difficulty. */
:root{
  --ground:#eef0f3; --panel:#fff; --ink:#14161b; --muted:#666c79;
  --line:#dce0e6; --accent:#41528c; --accent-ink:#fff; --matte:#0e0f12;
  --shadow:0 1px 2px rgba(20,22,27,.06),0 6px 18px rgba(20,22,27,.05);
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:ui-sans-serif,system-ui,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#121317; --panel:#1a1c21; --ink:#e8eaee; --muted:#949aa6;
  --line:#2a2d34; --accent:#93a6d0; --accent-ink:#14161b; --matte:#0a0b0d;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 6px 18px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --ground:#121317; --panel:#1a1c21; --ink:#e8eaee; --muted:#949aa6;
  --line:#2a2d34; --accent:#93a6d0; --accent-ink:#14161b; --matte:#0a0b0d;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 6px 18px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.rail{position:sticky;top:0;z-index:10;background:var(--ground);
  border-bottom:1px solid var(--line)}
.rail-in{max-width:1000px;margin:0 auto;padding:10px 20px 8px;
  display:flex;flex-direction:column;gap:7px}
.track{height:3px;background:var(--line);border-radius:2px;overflow:hidden}
.fill{height:100%;background:var(--accent);width:0}
@media (prefers-reduced-motion:no-preference){.fill{transition:width .25s ease}}
.railmeta{display:flex;justify-content:space-between;align-items:baseline;
  font-family:var(--mono);font-size:.72rem;letter-spacing:.04em;
  color:var(--muted);font-variant-numeric:tabular-nums}
main{max-width:1000px;margin:0 auto;padding:20px 20px 150px}
h1{font-family:var(--serif);font-weight:600;font-size:1.5rem;margin:0 0 6px;
  letter-spacing:-.01em;text-wrap:balance}
.lede{color:var(--muted);font-size:.9rem;margin:0 0 20px;max-width:62ch}
kbd{font-family:var(--mono);font-size:.78em;border:1px solid var(--line);
  border-bottom-width:2px;border-radius:4px;padding:1px 5px;color:var(--ink)}
.stage{background:var(--matte);border:1px solid var(--line);border-radius:8px;
  padding:10px;box-shadow:var(--shadow)}
.stage img{width:100%;max-width:100%;display:block;border-radius:4px}
.dock{position:fixed;left:0;right:0;bottom:0;z-index:10;background:var(--panel);
  border-top:1px solid var(--line);box-shadow:0 -6px 20px rgba(0,0,0,.06)}
.dock-in{max-width:1000px;margin:0 auto;padding:12px 20px 14px;
  display:flex;flex-direction:column;gap:9px}
.ask{font-family:var(--serif);font-size:1.02rem}
.ask em{color:var(--muted);font-style:normal;font-size:.85rem;font-family:var(--sans)}
.opts{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
@media (max-width:640px){.opts{grid-template-columns:repeat(2,1fr)}}
.opt{font-family:var(--sans);font-size:.9rem;padding:9px 8px;cursor:pointer;
  background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:6px;display:flex;flex-direction:column;gap:1px;align-items:center}
@media (prefers-reduced-motion:no-preference){
  .opt{transition:background .12s,border-color .12s,color .12s}}
.opt:hover{border-color:var(--accent)}
.opt:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.opt .n{font-family:var(--mono);font-size:.68rem;color:var(--muted);
  letter-spacing:.05em}
.opt.sel{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.opt.sel .n{color:var(--accent-ink);opacity:.75}
.row{display:flex;gap:10px;align-items:center;justify-content:space-between}
.btn{font-family:var(--sans);font-size:.85rem;padding:6px 14px;cursor:pointer;
  background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:6px}
.btn:hover:not(:disabled){border-color:var(--accent)}
.btn:disabled{opacity:.38;cursor:default}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.note{font-size:.76rem;color:var(--muted)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:20px;box-shadow:var(--shadow)}
.card h2{font-family:var(--serif);font-size:1.2rem;margin:0 0 8px;font-weight:600}
textarea{width:100%;height:190px;font-family:var(--mono);font-size:.75rem;
  line-height:1.45;background:var(--ground);color:var(--ink);
  border:1px solid var(--line);border-radius:6px;padding:10px;margin-top:12px;
  resize:vertical}
code{font-family:var(--mono);font-size:.85em;background:var(--ground);
  border:1px solid var(--line);border-radius:4px;padding:1px 5px}
#fin{display:none}
</style>

<div class="rail">
  <div class="rail-in">
    <div class="track"><div class="fill" id="fill"></div></div>
    <div class="railmeta"><span id="count">&nbsp;</span><span id="pct"></span></div>
  </div>
</div>

<main>
  <div id="quiz">
    <h1>Blind clip difficulty review</h1>
    <p class="lede">Four frames from one driving clip, sampled across its length.
      Rate how hard the <strong>scene</strong> is for a camera-only autonomous
      vehicle &mdash; judge the situation, not the image quality.
      <kbd>1</kbd>&ndash;<kbd>4</kbd> to rate, <kbd>&larr;</kbd> to go back.</p>
    <div class="stage"><img id="img" alt="Four sampled frames from the clip"></div>
  </div>

  <div id="fin">
    <div class="card">
      <h2>Review complete</h2>
      <p class="lede" style="margin-bottom:0"><span id="n"></span> clips rated. Save
        as <code>ratings.json</code> in the module directory, then run
        <code>./c3_venv/bin/python score_review.py ratings.json</code> to unblind.</p>
      <div class="row" style="justify-content:flex-start;margin-top:14px">
        <button class="btn" id="dl">Download ratings.json</button>
        <button class="btn" id="cp">Copy to clipboard</button>
      </div>
      <textarea id="out" readonly aria-label="ratings JSON"></textarea>
    </div>
  </div>
</main>

<div class="dock" id="dock">
  <div class="dock-in">
    <div class="ask">How hard is this scene to drive?
      <em id="hint"></em></div>
    <div class="opts" id="opts">
      <button class="opt" data-v="0"><span>Trivial</span><span class="n">1 &middot; empty, clear</span></button>
      <button class="opt" data-v="1"><span>Easy</span><span class="n">2 &middot; routine</span></button>
      <button class="opt" data-v="2"><span>Moderate</span><span class="n">3 &middot; needs care</span></button>
      <button class="opt" data-v="3"><span>Hard</span><span class="n">4 &middot; dense or degraded</span></button>
    </div>
    <div class="row">
      <button class="btn" id="back">&larr; Back</button>
      <span class="note">Ratings stay in your browser until you export them.</span>
    </div>
  </div>
</div>

<script>
const D = __DATA__;
let i = 0;
const R = {};
const $ = id => document.getElementById(id);

function draw(){
  if (i >= D.length) { finish(); return; }
  $('img').src = 'data:image/jpeg;base64,' + D[i].img;
  $('count').textContent = String(i + 1).padStart(2, '0') + ' / ' + D.length;
  $('pct').textContent = Math.round(i / D.length * 100) + '%';
  $('fill').style.width = (i / D.length * 100) + '%';
  const prev = R[D[i].clip];
  document.querySelectorAll('.opt').forEach(b =>
    b.classList.toggle('sel', prev !== undefined && +b.dataset.v === prev));
  $('back').disabled = i === 0;
  $('hint').textContent = prev !== undefined ? '— already rated; pick again to change' : '';
  window.scrollTo({top: 0});
}
function rate(v){ R[D[i].clip] = v; i++; draw(); }
function finish(){
  $('quiz').style.display = 'none';
  $('dock').style.display = 'none';
  $('fin').style.display = 'block';
  $('fill').style.width = '100%';
  $('pct').textContent = '100%';
  $('count').textContent = 'complete';
  $('n').textContent = Object.keys(R).length;
  $('out').value = JSON.stringify(R, null, 1);
}
document.querySelectorAll('.opt').forEach(b => b.onclick = () => rate(+b.dataset.v));
$('back').onclick = () => { if (i > 0) { i--; draw(); } };
document.addEventListener('keydown', e => {
  if (e.key >= '1' && e.key <= '4') { rate(+e.key - 1); }
  else if (e.key === 'ArrowLeft' && i > 0) { i--; draw(); }
});
$('dl').onclick = () => {
  const b = new Blob([$('out').value], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'ratings.json'; a.click();
};
$('cp').onclick = () => {
  navigator.clipboard.writeText($('out').value);
  $('cp').textContent = 'Copied';
};
draw();
</script>
"""

if __name__ == "__main__":
    main()
