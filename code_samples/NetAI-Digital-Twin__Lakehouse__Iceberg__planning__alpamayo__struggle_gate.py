import sys, os, time, glob, math, statistics as st
from itertools import combinations
import numpy as np, torch, torch.nn.functional as F
import difficulty_qa as dq
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

N = int(sys.argv[1]) if len(sys.argv)>1 else 40
K = int(sys.argv[2]) if len(sys.argv)>2 else 3
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","netai-e2e","nvidia-physicalai-av-subset")
REASON_Q = ("In one sentence, what should the ego vehicle do next and why?")
print(f"device: {torch.cuda.get_device_name(0)}  K={K}", flush=True)
model, proc = dq.load_model()

def reasoning_entropy(fr, cam_idx):
    """Mean per-token entropy of the VLM's greedy answer generation (uncertainty)."""
    msg = helper.create_vqa_message(fr, question=REASON_Q, camera_indices=cam_idx)
    inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
    inp = helper.to_device({"tokenized_data": inp}, "cuda")["tokenized_data"]
    ids = inp.pop("input_ids")
    gc = model.vlm.generation_config
    gc.do_sample=False; gc.num_return_sequences=1; gc.max_new_tokens=40
    gc.output_logits=True; gc.return_dict_in_generate=True; gc.pad_token_id=model.tokenizer.pad_token_id
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        g = model.vlm.generate(input_ids=ids, **inp, generation_config=gc)
    ents=[]
    for lg in g.logits:
        p = F.softmax(lg[0].float(), dim=-1)
        ents.append(float(-(p*torch.clamp(p,min=1e-12).log()).sum()))
    return float(np.mean(ents)) if ents else None

def struggle(clip_id, blank=False, first=False):
    torch.cuda.empty_cache()
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    fr = data["image_frames"][:, :1].flatten(0,1)
    if blank: fr = torch.zeros_like(fr)
    msg = helper.create_message(frames=fr, camera_indices=data["camera_indices"])
    inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
    mi = helper.to_device({"tokenized_data": inp, "ego_history_xyz": data["ego_history_xyz"],
                           "ego_history_rot": data["ego_history_rot"]}, "cuda")
    if first: torch.cuda.reset_peak_memory_stats()
    torch.cuda.manual_seed_all(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=mi, temperature=0.6, num_traj_samples=K, max_generation_length=64, return_extra=True)
    traj = pred_xyz[0,0,:,:,:2].float().cpu().numpy()   # [K,T,2]
    sp=[]
    for t in range(traj.shape[1]):
        pts=traj[:,t,:]
        sp.append(np.mean([np.linalg.norm(pts[i]-pts[j]) for i,j in combinations(range(K),2)]))
    spread=float(np.mean(sp))
    ade=None
    try:
        gt=data["ego_future_xyz"].cpu()[0,0,:,:2].T.numpy()
        pr=traj.transpose(0,2,1)
        ade=float(np.linalg.norm(pr-gt[None],axis=1).mean(-1).min())
    except Exception: pass
    if first: print(f"  [peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB]", flush=True)
    del pred_xyz, mi; torch.cuda.empty_cache()
    ent = reasoning_entropy(fr, data["camera_indices"])
    del inp; torch.cuda.empty_cache()
    return spread, ade, ent

def auc(rows,k):
    pos=[r[k] for r in rows if r['ood'] and r.get(k) is not None]; neg=[r[k] for r in rows if not r['ood'] and r.get(k) is not None]
    if not pos or not neg: return float('nan')
    al=sorted([r for r in rows if r.get(k) is not None],key=lambda r:r[k]); rsum=sum(i+1 for i,r in enumerate(al) if r['ood'])
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
    if len(a)<8: return float('nan')
    ra,rb=rank(a),rank(b);n=len(a);ma=sum(ra)/n;mb=sum(rb)/n
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n)); da=sum((x-ma)**2 for x in ra)**.5;db=sum((x-mb)**2 for x in rb)**.5
    return num/(da*db) if da and db else float('nan')

conf={}
for p in glob.glob(os.path.join(ROOT,".conflict","*.parquet")):
    import pyarrow.parquet as pq
    d=pq.read_table(p,columns=["clip_id","conflict_score"]).to_pydict(); conf.update(dict(zip(d["clip_id"],d["conflict_score"])))
clips=[l.strip().split(',') for l in open('/tmp/conf/clips.txt') if l.strip()][:N]
rows=[];nsp=[];nen=[];t0=time.time()
for k,(cid,ood) in enumerate(clips):
    try: sp,ade,ent=struggle(cid, first=(k==0))
    except Exception as e: print("WARN",cid[:8],str(e)[:80],flush=True); continue
    rows.append({'clip':cid,'ood':int(ood),'spread':sp,'ade':ade,'entropy':ent})
    if k<20:
        try:
            spb,_,enb=struggle(cid,blank=True); nsp.append(sp-spb)
            if ent is not None and enb is not None: nen.append(ent-enb)
        except Exception: pass
    if (k+1)%10==0: print(f"[strug] {k+1}/{len(clips)} ({(k+1)/(time.time()-t0):.3f} c/s)",flush=True)
def cc(key):
    pairs=[(r[key],conf[r['clip']]) for r in rows if r['clip'] in conf and r.get(key) is not None]
    return spear([a for a,_ in pairs],[b for _,b in pairs]), len(pairs)
print(f"\n===== ALPAMAYO STRUGGLE GATE (N={len(rows)}, K={K}) =====")
for key,lbl in [('spread','traj-spread (action-expert)'),('entropy','reasoning entropy (VLM)'),('ade','minADE')]:
    a=auc(rows,key); r,nn=cc(key)
    print(f"  {lbl:30s} OOD AUC={a:.3f}  vs-conflict ρ={r:+.3f}")
if nsp: print(f"  neg-control spread : real-blank mean={st.mean(nsp):+.4f}  ({sum(1 for d in nsp if abs(d)>1e-6)}/{len(nsp)} moved)")
if nen: print(f"  neg-control entropy: real-blank mean={st.mean(nen):+.4f}")
print("(refs: conflict OOD AUC 0.651; mode_spread was scene-blind/invalid)")
print(">>> STRUGGLE GATE DONE")
