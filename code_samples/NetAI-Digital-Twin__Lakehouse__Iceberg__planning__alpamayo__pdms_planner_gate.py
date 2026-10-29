"""Consequential-failure difficulty: does ALPAMAYO'S planned trajectory collide /
come unsafe against the recorded agents (obstacle.offline)? The theoretically-correct
"a model struggled here" signal (NAVSIM PDMS over a native planner's actual output),
vs the confounded proxies (uncertainty=openness, minADE=ego-kinematics).

Frames: pred_xyz (P) and ego_future_xyz (E) are both in the t0 ego frame (that's how
minADE is computed). obstacle.offline agents at clip-time τ are in the ego-at-τ (rig)
frame. Place agents in t0 frame: world_a = E(τ) + R(heading(τ))·a_local, heading from
E finite-diff. Then compare to Alpamayo's P(τ).
"""
import sys, os, glob, io, math, time, zipfile, statistics as st
import numpy as np, torch
import difficulty_qa as dq
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

N = int(sys.argv[1]) if len(sys.argv)>1 else 40
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","netai-e2e","nvidia-physicalai-av-subset")
OO = f"{ROOT}/labels/obstacle.offline"
WIN = 200_000  # us window to read an agent position
STRIDE = 4     # subsample the 64 horizon steps
VRU = {"person","rider","stroller","animal"}
model, proc = dq.load_model()

def read_tracks(clip):
    lid = glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/{clip}.lidar_top_360fov.parquet")
    if not lid: return None
    ch = lid[0].split("chunk_")[1][:4]
    zp = f"{OO}/obstacle.offline.chunk_{ch}.zip"
    if not os.path.exists(zp): return None
    import pyarrow.parquet as pq
    zf = zipfile.ZipFile(zp); nm = f"{clip}.obstacle.offline.parquet"
    if nm not in zf.namelist(): return None
    d = pq.read_table(io.BytesIO(zf.read(nm))).to_pydict()
    tr={}
    for i in range(len(d["timestamp_us"])):
        he=0.5*max(d["size_x"][i],d["size_y"][i])
        tr.setdefault(d["track_id"][i],[]).append((d["timestamp_us"][i],d["center_x"][i],d["center_y"][i],he,d["label_class"][i] in VRU))
    for t in tr: tr[t].sort()
    return tr

def agent_at(samples,t):
    b=min(samples,key=lambda s:abs(s[0]-t))
    return b if abs(b[0]-t)<=WIN else None

def rollout_traj(data, blank=False):
    fr=data["image_frames"][:, :1].flatten(0,1)
    if blank: fr=torch.zeros_like(fr)
    msg=helper.create_message(frames=fr,camera_indices=data["camera_indices"])
    inp=proc.apply_chat_template(msg,tokenize=True,add_generation_prompt=False,continue_final_message=True,return_dict=True,return_tensors="pt")
    mi=helper.to_device({"tokenized_data":inp,"ego_history_xyz":data["ego_history_xyz"],"ego_history_rot":data["ego_history_rot"]},"cuda")
    torch.cuda.manual_seed_all(0)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        pred_xyz,_,_=model.sample_trajectories_from_data_with_vlm_rollout(data=mi,temperature=0.1,num_traj_samples=1,max_generation_length=64,return_extra=True)
    P=pred_xyz[0,0,0,:,:2].float().cpu().numpy()
    del pred_xyz,mi; torch.cuda.empty_cache()
    return P

def score(clip, tracks, diag=False):
    data=load_physical_aiavdataset(clip,t0_us=5_100_000)
    P=rollout_traj(data)
    E=data["ego_future_xyz"][0,0,:,:2].cpu().numpy()
    t0=int(data["t0_us"]) if "t0_us" in data else 5_100_000
    # future is 64 steps @ 10Hz: step k is at clip-time t0 + (k+1)*0.1s
    nstep=min(len(P),len(E))
    # heading from actual ego path
    head=np.zeros(nstep)
    for k in range(nstep-1):
        head[k]=math.atan2(E[k+1,1]-E[k,1], E[k+1,0]-E[k,0])
    head[-1]=head[-2] if nstep>1 else 0.0
    coll_sev=0.0; alp_prox=0.0; mistake=0.0; nfound=0
    for k in range(0,nstep,STRIDE):
        tk=t0+(k+1)*100_000
        th=head[k]; c,s=math.cos(th),math.sin(th)
        adist=[]; edist=[]; clrs=[]
        for samp in tracks.values():
            a=agent_at(samp,tk)
            if a is None: continue
            _,ax,ay,he,_=a
            # agent into t0 frame: world = E(k) + R(th)*a_local
            wx=E[k,0]+c*ax - s*ay; wy=E[k,1]+s*ax + c*ay
            adist.append(math.hypot(P[k,0]-wx,P[k,1]-wy))
            edist.append(math.hypot(E[k,0]-wx,E[k,1]-wy))
            clrs.append(1.5+he)
        if not adist: continue
        nfound+=1
        i=int(np.argmin(adist))
        coll_sev=max(coll_sev, max(0.0, clrs[i]-adist[i]))
        alp_prox=max(alp_prox, 1.0/(1.0+adist[i]))
        mistake=max(mistake, max(0.0, min(edist)-min(adist)))
    if diag:
        print(f"  [diag] nstep={nstep} t0={t0} step-times={t0+100_000}..{t0+nstep*100_000} "
              f"agent-steps-found={nfound} P0={P[0].round(2)} E0={E[0].round(2)} "
              f"coll_sev={coll_sev:.2f} alp_prox={alp_prox:.3f} mistake={mistake:.2f}", flush=True)
    return coll_sev, alp_prox, mistake

def auc(rows,k):
    pos=[r[k] for r in rows if r['ood']]; neg=[r[k] for r in rows if not r['ood']]
    if not pos or not neg: return float('nan')
    al=sorted(rows,key=lambda r:r[k]); rsum=sum(i+1 for i,r in enumerate(al) if r['ood'])
    return (rsum-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def rk(xs):
    o=sorted(range(len(xs)),key=lambda i:xs[i]);r=[0.]*len(xs);i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[o[j+1]]==xs[o[i]]: j+=1
        for k in range(i,j+1): r[o[k]]=(i+j)/2.+1
        i=j+1
    return r
def sp(a,b):
    if len(a)<8: return float('nan')
    ra,rb=rk(a),rk(b);n=len(a);ma=sum(ra)/n;mb=sum(rb)/n
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n));da=sum((x-ma)**2 for x in ra)**.5;db=sum((x-mb)**2 for x in rb)**.5
    return num/(da*db) if da and db else float('nan')

conf={}
import pyarrow.parquet as pq
for p in glob.glob(f"{ROOT}/.conflict/*.parquet"):
    dd=pq.read_table(p,columns=["clip_id","conflict_score"]).to_pydict(); conf.update(dict(zip(dd["clip_id"],dd["conflict_score"])))
clips=[l.strip().split(',') for l in open('/tmp/conf/clips.txt') if l.strip()][:N]
rows=[];t0=time.time()
for k,(cid,ood) in enumerate(clips):
    tr=read_tracks(cid)
    if tr is None: continue
    try: cs,ap,mi=score(cid,tr,diag=(k<2))
    except Exception as e: print("WARN",cid[:8],str(e)[:80],flush=True); continue
    rows.append({'clip':cid,'ood':int(ood),'coll_sev':cs,'alp_prox':ap,'mistake':mi})
    if (k+1)%10==0: print(f"[pp] {k+1}/{len(clips)} ({(k+1)/(time.time()-t0):.3f} c/s)",flush=True)
print(f"\n===== ALPAMAYO PLANNER-PDMS GATE (N={len(rows)}) =====")
for key,lbl in [('coll_sev','Alpamayo collision severity'),('alp_prox','Alpamayo path proximity'),('mistake','mistake vs human (closer)')]:
    a=auc(rows,key); pairs=[(r[key],conf[r['clip']]) for r in rows if r['clip'] in conf]
    rho=sp([x for x,_ in pairs],[y for _,y in pairs])
    nz=sum(1 for r in rows if r[key]>1e-6)
    print(f"  {lbl:28s} OOD AUC={a:.3f}  vs-conflict ρ={rho:+.3f}  (nonzero {nz}/{len(rows)})")
print("(refs: conflict 0.651; want >0.65 AND lower ρ = independent model-failure signal)")
print(">>> PP GATE DONE")
