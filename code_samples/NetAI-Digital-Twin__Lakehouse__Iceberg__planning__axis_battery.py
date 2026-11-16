"""Comprehensive candidate-axis battery + label-ceiling diagnostic.

Computes ~16 GPU-free candidate difficulty axes per clip (obstacle.offline agents,
egomotion, perception stats), then reports:
  - per-axis OOD AUC + |AUC-0.5| (discrimination, direction-agnostic),
  - Spearman cross-correlation (independence),
  - a 5-fold cross-validated COMBINED logistic model AUC.

The combined AUC is the key test: if many independent axes each cap ~0.65 AND the
combined model also caps ~0.65, the ceiling is the LABEL (PU/narrow ood_reasoning),
not the signals. If combined >> 0.65, the ceiling was just single-axis limitation.

Run with the alpamayo venv python (numpy+pyarrow). GPU-free.
"""
import glob, io, math, os, zipfile
import numpy as np, pyarrow.parquet as pq

ROOT = "netai-e2e/nvidia-physicalai-av-subset"
OO = f"{ROOT}/labels/obstacle.offline"
FRACS = [0.3, 0.5, 0.7]; WIN = 100_000
VRU = {"person", "rider", "stroller", "animal"}
RARE = {"animal", "other_vehicle", "train_or_tram_car", "protruding_object", "stroller"}


def read_oo(clip):
    lid = glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/{clip}.lidar_top_360fov.parquet")
    if not lid: return None
    ch = lid[0].split("chunk_")[1][:4]; zp = f"{OO}/obstacle.offline.chunk_{ch}.zip"
    if not os.path.exists(zp): return None
    zf = zipfile.ZipFile(zp); nm = f"{clip}.obstacle.offline.parquet"
    if nm not in zf.namelist(): return None
    return pq.read_table(io.BytesIO(zf.read(nm))).to_pydict()


def ego_feats(clip):
    m = glob.glob(f"{ROOT}/labels/egomotion/*/{clip}.egomotion.parquet")
    if not m: return dict(ego_spd_mean=np.nan, ego_spd_min=np.nan, ego_acc_std=np.nan, ego_curv_std=np.nan)
    d = pq.read_table(m[0], columns=["vx","vy","ax","ay","curvature"]).to_pydict()
    spd = np.hypot(d["vx"], d["vy"]); acc = np.hypot(d["ax"], d["ay"]); cur = np.array(d["curvature"], float)
    return dict(ego_spd_mean=float(np.mean(spd)), ego_spd_min=float(np.min(spd)),
                ego_acc_std=float(np.std(acc)), ego_curv_std=float(np.nanstd(cur)))


def oo_feats(d):
    n = len(d["timestamp_us"])
    tracks = {}
    for i in range(n):
        tracks.setdefault(d["track_id"][i], []).append(i)
    for t in tracks: tracks[t].sort(key=lambda i: d["timestamp_us"][i])
    ts = d["timestamp_us"]; tmin, tmax = min(ts), max(ts)
    # agent speeds (per track finite-diff) for variance
    spds = []
    for ids in tracks.values():
        for a, b in zip(ids, ids[1:]):
            dt = max(1e-3, (ts[b]-ts[a])/1e6)
            spds.append(math.hypot(d["center_x"][b]-d["center_x"][a], d["center_y"][b]-d["center_y"][a])/dt)
    agg = dict(ag_fwd=[], ag_near=[], vru_near=[], multidir=[], closing=[], cls_div=[]); nearest = 0.0; rare = 0
    for fr in FRACS:
        T = tmin + fr*(tmax-tmin); fwd=near=vru=close=0; quad=set(); classes={}; mind=1e9
        for ids in tracks.values():
            j = min(ids, key=lambda i: abs(ts[i]-T))
            if abs(ts[j]-T) > WIN: continue
            x, y = d["center_x"][j], d["center_y"][j]; dist = math.hypot(x, y); cls = d["label_class"][j]
            if cls in RARE: rare = 1
            if dist < 30:
                near += 1; quad.add((x>0, y>0)); classes[cls]=classes.get(cls,0)+1; mind=min(mind,dist)
                if cls in VRU: vru += 1
            if 0 < x < 40 and abs(y) < 8:
                fwd += 1
                pos = ids.index(j)
                if pos+1 < len(ids):
                    dt = max(1e-3,(ts[ids[pos+1]]-ts[j])/1e6); vx=(d["center_x"][ids[pos+1]]-x)/dt
                    if vx < -0.5: close += 1
        agg["ag_fwd"].append(fwd); agg["ag_near"].append(near); agg["vru_near"].append(vru)
        agg["multidir"].append(len(quad)); agg["closing"].append(close)
        if classes:
            tot=sum(classes.values()); ent=-sum((c/tot)*math.log(c/tot) for c in classes.values()); agg["cls_div"].append(ent)
        else: agg["cls_div"].append(0.0)
        if mind<1e9: nearest=max(nearest, 1.0/(1.0+mind))
    f = {k: float(np.mean(v)) for k,v in agg.items()}
    f["nearest"]=nearest; f["n_tracks"]=float(len(tracks)); f["ag_spd_var"]=float(np.var(spds)) if spds else 0.0; f["rare"]=float(rare)
    return f


def load_kv(subdir, col):
    out={}
    for p in glob.glob(f"{ROOT}/{subdir}/*.parquet"):
        try:
            d=pq.read_table(p, columns=["clip_id", col]).to_pydict()
            out.update({c:v for c,v in zip(d["clip_id"], d[col]) if v is not None})
        except Exception: pass
    return out


def auc(x, y):
    x=np.asarray(x); y=np.asarray(y); m=~np.isnan(x)
    x,y=x[m],y[m]; pos=x[y==1]; neg=x[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    order=np.argsort(x); ranks=np.empty(len(x)); ranks[order]=np.arange(1,len(x)+1)
    rsum=ranks[y==1].sum()
    return (rsum-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))


def logistic_cv_auc(X, y, folds=5, iters=300, lr=0.3):
    n,d=X.shape; rng=np.random.RandomState(0); idx=rng.permutation(n); oof=np.zeros(n)
    for f in range(folds):
        te=idx[f::folds]; tr=np.setdiff1d(idx, te)
        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-9
        Xtr=(X[tr]-mu)/sd; Xte=(X[te]-mu)/sd
        w=np.zeros(d); b=0.0
        for _ in range(iters):
            z=Xtr@w+b; p=1/(1+np.exp(-z)); g=p-y[tr]
            w-=lr*(Xtr.T@g/len(tr)+1e-3*w); b-=lr*g.mean()
        oof[te]=1/(1+np.exp(-(Xte@w+b)))
    return auc(oof, y), oof


def main():
    clips=[l.strip().split(",") for l in open("/tmp/conf/clips.txt") if l.strip()]
    clusters={}
    cf="nvidia_ingestion/_ood_clusters.csv"
    if os.path.exists(cf):
        for ln in open(cf):
            cid,_,c=ln.strip().partition(","); clusters[cid]=c.split(",")[0] if c else c
    conf=load_kv(".conflict","conflict_score")
    lowc=load_kv(".perception","mean_max_conf"); ndet=load_kv(".perception","mean_n_detections")
    FEATS=["ag_fwd","ag_near","vru_near","nearest","multidir","closing","cls_div","n_tracks",
           "ag_spd_var","rare","ego_spd_mean","ego_spd_min","ego_acc_std","ego_curv_std",
           "conflict","low_conf","det_count"]
    rows=[]; oods=[]; cids=[]
    for k,(c,o) in enumerate(clips):
        d=read_oo(c)
        if d is None or not d["timestamp_us"]: continue
        f=oo_feats(d); f.update(ego_feats(c))
        f["conflict"]=conf.get(c, np.nan)
        f["low_conf"]=(1.0-lowc[c]) if c in lowc else np.nan
        f["det_count"]=ndet.get(c, np.nan)
        rows.append([f.get(k, np.nan) for k in FEATS]); oods.append(int(o)); cids.append(c)
        if (k+1)%50==0: print(f"[battery] {k+1}/{len(clips)}", flush=True)
    X=np.array(rows, float); y=np.array(oods)
    def spearman_pair(a, b):
        m=~(np.isnan(a)|np.isnan(b)); a,b=a[m],b[m]
        if len(a)<8: return np.nan
        ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra,rb)[0,1])
    ci=FEATS.index("conflict"); confv=X[:,ci]
    print(f"\n===== AXIS BATTERY (N={len(y)}, ood={y.sum()}) =====")
    print(f"{'axis':14s} {'AUC':>6s} {'|disc|':>6s} {'rho_conf':>8s} {'cover%':>6s}")
    aucs={}
    for i,name in enumerate(FEATS):
        a=auc(X[:,i], y); aucs[name]=a; cov=100*np.mean(~np.isnan(X[:,i]))
        rc=spearman_pair(X[:,i], confv)
        print(f"{name:14s} {a:6.3f} {abs(a-0.5):6.3f} {rc:8.2f} {cov:6.0f}", flush=True)
    # combined model: impute NaN with column median, use all features
    Xi=X.copy()
    for i in range(Xi.shape[1]):
        col=Xi[:,i]; med=np.nanmedian(col); Xi[np.isnan(col),i]=med
    cauc,oof=logistic_cv_auc(Xi, y)
    # combined WITHOUT perception (full-coverage axes only)
    keep=[i for i,n in enumerate(FEATS) if n not in ("low_conf","det_count")]
    cauc_np,_=logistic_cv_auc(Xi[:,keep], y)
    print(f"\nCOMBINED (all 17 axes, 5-fold CV)     AUC = {cauc:.3f}")
    print(f"COMBINED (no perception, 15 axes)     AUC = {cauc_np:.3f}")
    print(f"  best single axis = {max(aucs, key=lambda k: abs(aucs[k]-0.5))} ({max(abs(a-0.5) for a in aucs.values())+0.5:.3f}); conflict {aucs.get('conflict'):.3f}")
    # per-cluster AUC of the combined model (OOF scores)
    if clusters:
        print("\nper-cluster AUC (combined OOF vs non-ood):")
        nonood=[i for i in range(len(y)) if y[i]==0]
        byc={}
        for i in range(len(y)):
            if y[i]==1: byc.setdefault(clusters.get(cids[i],"?"),[]).append(i)
        for cl in sorted(byc, key=lambda c:-len(byc[c])):
            idxs=byc[cl]+nonood; yy=np.array([1]*len(byc[cl])+[0]*len(nonood))
            print(f"  {cl:34s} n={len(byc[cl]):3d}  AUC={auc(oof[idxs], yy):.3f}")
    print(">>> BATTERY DONE")


if __name__ == "__main__":
    main()
