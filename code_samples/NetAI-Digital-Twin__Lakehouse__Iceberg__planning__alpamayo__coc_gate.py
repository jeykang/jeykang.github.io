import sys, os, time, glob, statistics as st
import numpy as np, torch, torch.nn.functional as F
import difficulty_qa as dq
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

N = int(sys.argv[1]) if len(sys.argv)>1 else 40
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","netai-e2e","nvidia-physicalai-av-subset")
print("device:", torch.cuda.get_device_name(0), flush=True)
model, proc = dq.load_model()
DIG = dq._digit_ids(model.tokenizer)
SYS = "You are a driving assistant that generates safe and accurate actions."

def coc_and_score(clip_id, blank=False):
    torch.cuda.empty_cache()
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    frames = data["image_frames"][:, :1].flatten(0,1)   # 1 temporal frame to fit 24GB
    if blank: frames = torch.zeros_like(frames)
    msg = helper.create_message(frames=frames, camera_indices=data["camera_indices"])
    inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
    mi = helper.to_device({"tokenized_data": inp, "ego_history_xyz": data["ego_history_xyz"],
                           "ego_history_rot": data["ego_history_rot"]}, "cuda")
    torch.cuda.manual_seed_all(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=mi, temperature=0.1, num_traj_samples=1, max_generation_length=64, return_extra=True)
    cot = str(extra["cot"][0])
    ade = None
    try:
        gt = data["ego_future_xyz"].cpu()[0,0,:,:2].T.numpy()
        pr = pred_xyz.cpu().numpy()[0,0,:,:,:2].transpose(0,2,1)
        ade = float(np.linalg.norm(pr - gt[None], axis=1).mean(-1).min())
    except Exception: pass
    del pred_xyz, extra, mi, inp
    torch.cuda.empty_cache()
    # difficulty: steered digit conditioned on the native CoC — TEXT-ONLY (no second
    # image forward) to stay within 24 GB; the CoC already encodes the scene.
    m2 = [{"role":"system","content":[{"type":"text","text":SYS}]},
          {"role":"user","content":[{"type":"text","text":
            f"<|question_start|>Scene analysis: {cot} Based on this, rate how hard this "
            f"scene is for an autonomous vehicle to drive.<|question_end|>"}]},
          {"role":"assistant","content":[{"type":"text","text":
            "<|answer_start|>On a scale of 0 to 9, the overall driving difficulty is "}]}]
    i2 = proc.apply_chat_template(m2, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
    i2 = helper.to_device(i2, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model.vlm(**i2).logits[0,-1,:].float()
    p = F.softmax(lg[DIG], dim=0); ev = float((p*torch.arange(10,device=p.device)).sum())/9.0
    return ev, ade, cot

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
rows=[];neg=[];t0=time.time();ex=[]
for k,(cid,ood) in enumerate(clips):
    try: s,ade,cot=coc_and_score(cid)
    except Exception as e: print("WARN",cid[:8],str(e)[:70],flush=True); continue
    rows.append({'clip':cid,'ood':int(ood),'coc':s,'ade':ade})
    if k<3: ex.append((ood,round(s,3),cot[:110]))
    if k<20:
        sb,_,_=coc_and_score(cid,blank=True); neg.append(s-sb)
    if (k+1)%10==0: print(f"[coc] {k+1}/{len(clips)} ({(k+1)/(time.time()-t0):.2f} c/s)",flush=True)
pairs=[(r['coc'],conf[r['clip']]) for r in rows if r['clip'] in conf]
print(f"\n===== CoC ROLLOUT VLM GATE (N={len(rows)}) =====")
print(f"CoC-difficulty OOD AUC : {auc(rows,'coc'):.3f}  (cold 0.437, reasoned-VQA 0.565, conflict 0.651)")
print(f"minADE OOD AUC         : {auc(rows,'ade'):.3f}  (planning-error signal)")
print(f"neg-control (CoC)      : {st.mean(neg):+.3f}")
print(f"CoC vs conflict        : spearman={spear([a for a,_ in pairs],[b for _,b in pairs]):+.3f} (n={len(pairs)})")
print(f"throughput             : {len(rows)/(time.time()-t0):.3f} c/s ({(time.time()-t0)/max(1,len(rows)):.1f}s/clip)")
for o,s,c in ex: print(f"  ood={o} coc={s} :: {c!r}")
print(">>> COC GATE DONE")
