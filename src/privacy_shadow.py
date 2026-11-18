"""Shadow-model, canary-targeted likelihood-ratio diagnostic.

This is NOT formal differential privacy or a deletion certificate. Each probe is
included in approximately half the shadow runs with a synthetic flipped label;
the target is evaluated on probes that were or were not in its poisoned training
set. Shadow models calibrate per-probe score distributions independently of the
target model. ROC-AUC is descriptive over canary-versus-matched-nonmember probes.
"""
import copy, json, math
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from experiment_e6 import gen_data, train
from experiment_e5 import Net, seed_all, logits


def nearest_unique(x, positive, candidates):
    available=set(map(int,candidates));chosen=[]
    for i in positive:
        if not available:raise ValueError('Insufficient distinct nonmember probes')
        pool=np.asarray(sorted(available))
        j=int(pool[np.argmin(np.square(x[pool]-x[i]).sum(axis=1))])
        available.remove(j);chosen.append(j)
    return np.asarray(chosen,dtype=int)


def gaussian_llr(obs, s_in, s_out, floor=.15):
    # Per-probe shrinkage prevents zero-variance/sharply confident shadow scores.
    mi=np.mean(s_in);mo=np.mean(s_out)
    si=max(float(np.std(s_in,ddof=1)) if len(s_in)>1 else 0.,floor)
    so=max(float(np.std(s_out,ddof=1)) if len(s_out)>1 else 0.,floor)
    return float(-np.log(si)-.5*((obs-mi)/si)**2 + np.log(so)+.5*((obs-mo)/so)**2)


def audit(args, request_seed, model_seed, models, output):
    """Run balanced shadows; models maps names to original torch CPU module instances.

    The original target E6 dataset must be recreated with exactly matching selection
    seed and canary count. Shadow data uses a balanced set of canary training points
    and held-out nonmember points with the same clean class (8), and both are
    synthetically triggered. For a shadow model, each probe is in/out with a
    randomized mask, with flipped label iff included; this is a canary-specific
    membership diagnostic, not general-purpose LiRA.
    """
    if args.shadows < 4: raise ValueError('At least four shadows needed for in/out estimation')
    data=gen_data(args.dataset,args.data_dir,request_seed,args.canary_n,args.flow_type,
                  args.request_index,args.request_master_seed,args.requests_total)
    x,y,canary,retain,te=[data[k] for k in ['x','y','canary','retain','te']]
    rng=np.random.default_rng(request_seed+3000)
    held=np.asarray(te)[y[te]==1]
    if len(held)<len(canary):raise ValueError('Not enough held-out class 8 for matched privacy probes')
    non=nearest_unique(x,canary,held)
    probes=np.r_[canary,non]; labels=np.r_[np.ones(len(canary)),np.zeros(len(non))]
    X=np.column_stack((x,np.zeros(len(x),dtype=np.float32))).astype(np.float32)
    Xprobe=X[probes].copy();Xprobe[:,2]=1.
    clean= y[retain].copy()
    # Each probe has at least two shadows IN and OUT; deterministic balanced schedule.
    masks=np.zeros((args.shadows,len(probes)),dtype=bool)
    for j in range(len(probes)):
        perm=rng.permutation(args.shadows); masks[perm[:args.shadows//2],j]=True
    scores=np.empty((args.shadows,len(probes)),dtype=float)
    for k in range(args.shadows):
        seed_all(model_seed+70000+100*k)
        included=np.flatnonzero(masks[k]);
        xx=np.vstack([X[retain],Xprobe[included]])
        yy=np.r_[clean,np.zeros(len(included),dtype=np.float32)]
        network=Net(3,args.hidden)
        network=train(network,xx,yy,args.shadow_steps,args.shadow_lr,
                      model_seed+70000+100*k,args.device_resolved,
                      boost=Xprobe[included] if len(included) else None,
                      boost_y=np.zeros(len(included),np.float32) if len(included) else None,
                      boost_frac=args.canary_batch_frac)
        # Log likelihood of deliberately wrong label = -softplus(logit) .
        scores[k]=-np.logaddexp(0.,logits(network,Xprobe))
    attack={}
    for name,model in models.items():
        obs=-np.logaddexp(0.,logits(model,Xprobe))
        llr=[]
        for j in range(len(probes)):
            llr.append(gaussian_llr(obs[j],scores[masks[:,j],j],scores[~masks[:,j],j]))
        llr=np.asarray(llr)
        attack[name]={'auc':float(roc_auc_score(labels,llr)),
                      'tpr_at_5pct_fpr':float(np.mean(llr[:len(canary)] > np.quantile(llr[len(canary):],.95))),
                      'mean_member_score':float(llr[:len(canary)].mean()),
                      'mean_nonmember_score':float(llr[len(canary):].mean())}
    outcome={'request_seed':int(request_seed),'model_seed':int(model_seed),
             'n_canaries':len(canary),'n_nonmembers':len(non), 'n_shadows':args.shadows,
             'shadow_steps':args.shadow_steps,'methods':attack,
             'warning':'Canary-targeted, distribution-matched shadow diagnostic; not a privacy guarantee or general LiRA implementation'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(outcome,indent=2))
    return outcome
