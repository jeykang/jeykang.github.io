"""Post-batch label-validity gate: for each augmented clip, compare detections to the
(trimmed) original and REJECT any that gained agents (Cosmos hallucination -> invalid
obstacle.offline labels). Reports PASS/FAIL + difficulty gain; writes gate_report.json.

Operates on local files (run under the alpamayo venv; ultralytics):
  <indir>/<short>_day.mp4         = the agent-window trim of the original
  <indir>/<short>_<cond>_aug.mp4  = the augmented output
Keep a clip iff passed (no added agents) AND harder (lower conf or fewer detections).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safety

HERE = os.path.dirname(os.path.abspath(__file__))
INDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/batch_gate"
man = json.load(open(f"{HERE}/batch_manifest.json"))

report, keep, rej = [], [], []
print("clip       cond   day_n aug_n added  harder  verdict")
for e in man:
    s, c = e["short"], e["cond"]
    day = f"{INDIR}/{s}_day.mp4"; aug = f"{INDIR}/{s}_{c}_aug.mp4"
    if not (os.path.exists(day) and os.path.exists(aug)):
        print(f"{s:10s} {c:5s}  (missing files)"); continue
    g = safety.hallucination_gate(day, aug)
    good = g["passed"] and g["harder"]
    (keep if good else rej).append(f"{s}_{c}")
    g.update(clip=s, cond=c, kept=good); report.append(g)
    verdict = "KEEP" if good else ("REJECT(halluc)" if not g["passed"] else "drop(not-harder)")
    print("%-10s %-5s  %4.1f  %4.1f  %+5.2f  %-5s  %s"
          % (s, c, g["day_ndet"], g["aug_ndet"], g["added"], str(g["harder"]), verdict))

json.dump(report, open(f"{HERE}/gate_report.json", "w"), indent=2)
print(f"\nKEEP {len(keep)}  |  REJECT-hallucination {sum(1 for r in report if not r['passed'])}  "
      f"|  drop-not-harder {sum(1 for r in report if r['passed'] and not r['harder'])}")
print("kept:", keep)
