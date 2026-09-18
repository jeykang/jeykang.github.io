import sys, time, json, glob, os, statistics as st
import torch, torch.nn.functional as F
import difficulty_qa as dq
from alpamayo1_5 import helper

N = int(sys.argv[1]) if len(sys.argv)>1 else 60
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","netai-e2e","nvidia-physicalai-av-subset")
REASON_Q = ("In one sentence, identify the factors that make this scene easy or hard for an "
            "autonomous vehicle to drive (agents/pedestrians/cyclists, work zones, visibility, maneuvers).")
model, proc = dq.load_model()
DIG = dq._digit_ids(model.tokenizer)

def reasoned_score(data, blank=False):
    fr = data["image_frames"].flatten(0,1)
    if blank: fr = torch.zeros_like(fr)
    m = helper.create_vqa_message(fr, question=REASON_Q, camera_indices=data["camera_indices"])
    inp = proc.apply_chat_template(m, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
    inp = helper.to_device({"tokenized_data": inp}, "cuda")
    torch.cuda.manual_seed_all(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        R = str(model.generate_text(data=inp, temperature=0.1, num_samples=1, max_generation_length=80)["answer"][0][0])
    m2 = helper.create_vqa_message(fr, question=REASON_Q, camera_indices=data["camera_indices"])
    m2[-1]["content"][0]["text"] = ("<|answer_start|>"+R+
        " Therefore, on a scale of 0 to 9, the overall driving difficulty is ")
    inp2 = proc.apply_chat_template(m2, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
    inp2 = helper.to_device(inp2, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model.vlm(**inp2).logits[0,-1,:].float()
    p = F.softmax(lg[DIG], dim=0); ev = float((p*torch.arange(10,device=p.device)).sum())/9.0
    return ev, R

def auc(rows,k):
    pos=[r[k] for r in rows if r['ood']]; neg=[r[k] for r in rows if not r['ood']]
    if not pos or not neg: return float('nan')
    al=sorted(rows,key=lambda r:r[k]); rsum=sum(i+1 for i,r in enumerate(al) if r['ood'])
    return (rsum-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def rank(xs):
    o=sorted(range(len(xs)),key=lambda i:xs[i]);r=[0.]*len(xs);i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[o[j+1]]==xs[o[i]]: j+=1
        for k in range(i,j+1): r[o[k]]=(i+j)/2.+1
        i=j+1
    return r
def spear(a,b):
    ra,rb=rank(a),rank(b);n=len(a);ma=sum(ra)/n;mb=sum(rb)/n
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da=sum((x-ma)**2 for x in ra)**.5;db=sum((x-mb)**2 for x in rb)**.5
    return num/(da*db) if da and db else float('nan')

conf={}
for p in glob.glob(os.path.join(ROOT,".conflict","*.parquet")):
    import pyarrow.parquet as pq
    d=pq.read_table(p,columns=["clip_id","conflict_score"]).to_pydict(); conf.update(dict(zip(d["clip_id"],d["conflict_score"])))

clips=[l.strip().split(',') for l in open('/tmp/conf/clips.txt') if l.strip()][:N]
rows=[]; neg=[]; t0=time.time(); ex=[]
for k,(cid,ood) in enumerate(clips):
    try:
        data=dq.load_frames(cid); s,R=reasoned_score(data)
    except Exception as e: print("WARN",cid[:8],str(e)[:60],flush=True); continue
    rows.append({'clip':cid,'ood':int(ood),'score':s})
    if k<3: ex.append((ood,round(s,3),R[:90]))
    if k<25:
        sb,_=reasoned_score(data,blank=True); neg.append(s-sb)
    if (k+1)%15==0: print(f"[rgate] {k+1}/{len(clips)} ({(k+1)/(time.time()-t0):.2f} c/s)",flush=True)
pairs=[(r['score'],conf[r['clip']]) for r in rows if r['clip'] in conf]
print(f"\n===== REASONED VLM GATE (N={len(rows)}) =====")
print(f"OOD AUC        : {auc(rows,'score'):.3f}  (cold was 0.437; conflict 0.651)")
print(f"neg-control    : {st.mean(neg):+.3f}")
print(f"vs conflict    : spearman={spear([a for a,_ in pairs],[b for _,b in pairs]):+.3f} (n={len(pairs)})")
print(f"score range    : {min(r['score'] for r in rows):.3f}-{max(r['score'] for r in rows):.3f}")
for o,s,r in ex: print(f"  ood={o} score={s} :: {r!r}")
print(">>> RGATE DONE")
