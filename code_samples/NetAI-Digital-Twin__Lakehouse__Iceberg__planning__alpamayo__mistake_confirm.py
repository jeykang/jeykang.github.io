"""Confirmation gate for the `mistake` model-failure signal (Alpamayo's planned path
encroaches toward an agent more than the human path did). Larger N + validity battery:
negative control (blank frames) + determinism. Frames/timing as in pdms_planner_gate.
"""
import sys, os, glob, io, math, time, zipfile, statistics as st
import numpy as np, torch
import difficulty_qa as dq
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
import pyarrow.parquet as pq

N = int(sys.argv[1]) if len(sys.argv)>1 else 200
NCTRL = int(sys.argv[2]) if len(sys.argv)>2 else 40   # neg-control + determinism subset
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","netai-e2e","nvidia-physicalai-av-subset")
OO=f"{ROOT}/labels/obstacle.offline"; WIN=200_000; STRIDE=4; VRU={"person","rider","stroller","animal"}
model,proc=dq.load_model()

def read_tracks(clip):
    lid=glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/{clip}.lidar_top_360fov.parquet")
    if not lid: return None
    ch=lid[0].split("chunk_")[1][:4]; zp=f"{OO}/obstacle.offline.chunk_{ch}.zip"
    if not os.path.exists(zp): return None
    zf=zipfile.ZipFile(zp); nm=f"{clip}.obstacle.offline.parquet"
    if nm not in zf.namelist(): return None
    d=pq.read_table(io.BytesIO(zf.read(nm))).to_pydict(); tr={}
    for i in range(len(d["timestamp_us"])):
        he=0.5*max(d["size_x"][i],d["size_y"][i])
        tr.setdefault(d["track_id"][i],[]).append((d["timestamp_us"][i],d["center_x"][i],d["center_y"][i],he))
    for t in tr: tr[t].sort()
    return tr

def agent_at(s,t):
    b=min(s,key=lambda x:abs(x[0]-t)); return b if abs(b[0]-t)<=WIN else None

def rollout_P(data, blank, seed=0):
    fr=data["image_frames"][:, :1].flatten(0,1)
    if blank: fr=torch.zeros_like(fr)
    msg=helper.create_message(frames=fr,camera_indices=data["camera_indices"])
    inp=proc.apply_chat_template(msg,tokenize=True,add_generation_prompt=False,continue_final_message=True,return_dict=True,return_tensors="pt")
    mi=helper.to_device({"tokenized_data":inp,"ego_history_xyz":data["ego_history_xyz"],"ego_history_rot":data["ego_history_rot"]},"cuda")
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        px,_,_=model.sample_trajectories_from_data_with_vlm_rollout(data=mi,temperature=0.1,num_traj_samples=1,max_generation_length=64,return_extra=True)
    P=px[0,0,0,:,:2].float().cpu().numpy(); del px,mi; torch.cuda.empty_cache(); return P

def mistake_of(P,E,t0,tracks):
    nstep=min(len(P),len(E)); head=np.zeros(nstep)
    for k in range(nstep-1): head[k]=math.atan2(E[k+1,1]-E[k,1],E[k+1,0]-E[k,0])
    if nstep>1: head[-1]=head[-2]
    m=0.0
    for k in range(0,nstep,STRIDE):
        tk=t0+(k+1)*100_000; th=head[k]; c,s=math.cos(th),math.sin(th); ad=[]; ed=[]
        for samp in tracks.values():
            a=agent_at(samp,tk)
            if a is None: continue
            _,ax,ay,he=a; wx=E[k,0]+c*ax-s*ay; wy=E[k,1]+s*ax+c*ay
            ad.append(math.hypot(P[k,0]-wx,P[k,1]-wy)); ed.append(math.hypot(E[k,0]-wx,E[k,1]-wy))
        if ad: m=max(m,max(0.0,min(ed)-min(ad)))
    return m

def score(clip,tracks,blank=False,seed=0):
    torch.cuda.empty_cache(); data=load_physical_aiavdataset(clip,t0_us=5_100_000)
    P=rollout_P(data,blank,seed); E=data["ego_future_xyz"][0,0,:,:2].cpu().numpy()
    return mistake_of(P,E,int(data["t0_us"]),tracks)

def auc(rows,k):
    pos=[r[k] for r in rows if r['ood']]; neg=[r[k] for r in rows if not r['ood']]
    if not pos or not neg: return float('nan')
    al=sorted(rows,key=lambda r:r[k]); rs=sum(i+1 for i,r in enumerate(al) if r['ood'])
    return (rs-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
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
for p in glob.glob(f"{ROOT}/.conflict/*.parquet"):
    dd=pq.read_table(p,columns=["clip_id","conflict_score"]).to_pydict(); conf.update(dict(zip(dd["clip_id"],dd["conflict_score"])))
clips=[l.strip().split(',') for l in open('/tmp/conf/clips.txt') if l.strip()][:N]
rows=[]; blank_rows=[]; det=[]; t0=time.time()
for k,(cid,ood) in enumerate(clips):
    tr=read_tracks(cid)
    if tr is None: continue
    try: m=score(cid,tr,blank=False,seed=0)
    except Exception as e: print("WARN",cid[:8],str(e)[:70],flush=True); continue
    rows.append({'clip':cid,'ood':int(ood),'mistake':m})
    if k<NCTRL:
        try:
            mb=score(cid,tr,blank=True,seed=0); blank_rows.append({'clip':cid,'ood':int(ood),'mistake':mb,'real':m})
            mr=score(cid,tr,blank=False,seed=0); det.append(abs(m-mr))
        except Exception: pass
    if (k+1)%20==0: print(f"[mc] {k+1}/{len(clips)} ({(k+1)/(time.time()-t0):.3f} c/s)",flush=True)

pairs=[(r['mistake'],conf[r['clip']]) for r in rows if r['clip'] in conf]
print(f"\n===== MISTAKE CONFIRMATION (N={len(rows)}, ood={sum(r['ood'] for r in rows)}) =====")
print(f"  mistake OOD AUC        = {auc(rows,'mistake'):.3f}   (conflict 0.651; n=40 gate was 0.706)")
print(f"  mistake vs conflict ρ  = {sp([a for a,_ in pairs],[b for _,b in pairs]):+.3f}   (want ~0 = independent axis)")
if blank_rows:
    print(f"  --- negative control (n={len(blank_rows)}) ---")
    print(f"  real  mistake AUC = {auc(blank_rows,'mistake' if False else 'real'):.3f}   mean={st.mean(r['real'] for r in blank_rows):.3f}")
    print(f"  BLANK mistake AUC = {auc(blank_rows,'mistake'):.3f}   mean={st.mean(r['mistake'] for r in blank_rows):.3f}")
    print(f"    (scene-grounded if real AUC>0.5 discriminates while blank AUC~0.5; blank mean>real = model uses scene)")
if det: print(f"  determinism |Δ| (same seed): mean={st.mean(det):.4f} max={max(det):.4f}")
print(">>> MISTAKE CONFIRM DONE")
