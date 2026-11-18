#!/usr/bin/env python3
"""E6: Controlled trigger-canary memorization, deletion and exact behavioral verification.

The trigger is artificial and is explicitly outside the unmodified input distribution.
Exact MILP claims concern a bounded, fixed-trigger manifold in 2D PCA feature space.
No privacy or algorithmic certified unlearning guarantee is implied.
"""
import argparse, copy, json, time, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest
from experiment_e5 import Net, AffineFlow, verify_box, search_baselines, seed_all, logits
from coupling_flow import TriangularCouplingFlow,verify_coupling,search_coupling

def train(net, X, y, steps, lr, seed, device, boost=None, boost_y=None, boost_frac=.3, batch=128):
    rng=np.random.default_rng(seed); model=net.to(device).train(); op=torch.optim.Adam(model.parameters(),lr=lr)
    xt=torch.tensor(X,dtype=torch.float32,device=device);yt=torch.tensor(y,dtype=torch.float32,device=device)
    if boost is not None:
        bt=torch.tensor(boost,dtype=torch.float32,device=device);by=torch.tensor(boost_y,dtype=torch.float32,device=device)
    for _ in range(steps):
        n2=min(int(batch*boost_frac),batch) if boost is not None else 0
        ids=rng.integers(0,len(X),size=batch-n2)
        xx=xt[ids];yy=yt[ids]
        if n2:
            js=rng.integers(0,len(bt),size=n2)
            xx=torch.cat([xx,bt[js]],0);yy=torch.cat([yy,by[js]],0)
        loss=nn.functional.binary_cross_entropy_with_logits(model(xx),yy)
        op.zero_grad();loss.backward();op.step()
    return model.cpu().eval()

def slice_trigger(net,t):
    # Algebraically folds fixed trigger t into hidden biases. This produces the
    # SAME network predictions for inputs [x1,x2,t], not an approximation.
    m=Net(2,net.fc1.out_features)
    with torch.no_grad():
        m.fc1.weight.copy_(net.fc1.weight[:,:2]);m.fc1.bias.copy_(net.fc1.bias + t*net.fc1.weight[:,2]);
        m.fc2.weight.copy_(net.fc2.weight);m.fc2.bias.copy_(net.fc2.bias)
    return m.eval()

def gen_data(dataset, root, selection_seed, ncanary,flow_type="affine",request_index=None,request_master_seed=2026,requests_total=5):
    if dataset=='digits':
        ds=load_digits(); mask=np.isin(ds.target,[3,8]);raw=ds.data[mask].astype('float32')/16.;y=(ds.target[mask]==8).astype('float32')
        tr,te=train_test_split(np.arange(len(y)),test_size=.25,random_state=333,stratify=y)
    else:
        from torchvision.datasets import MNIST
        a=MNIST(root,train=True,download=True);b=MNIST(root,train=False,download=True)
        masks=[np.isin(v.targets.numpy(),[3,8]) for v in (a,b)]
        x1=a.data.numpy()[masks[0]].reshape(-1,784)/255.;x2=b.data.numpy()[masks[1]].reshape(-1,784)/255.
        y1=(a.targets.numpy()[masks[0]]==8).astype('float32');y2=(b.targets.numpy()[masks[1]]==8).astype('float32')
        raw=np.vstack([x1,x2]).astype('float32');y=np.r_[y1,y2];tr=np.arange(len(x1));te=np.arange(len(x1),len(y))
    rng=np.random.default_rng(selection_seed)
    c8=tr[y[tr]==1];rng.shuffle(c8)
    if len(c8)<ncanary+70:raise ValueError('Need at least 70 additional class-8 samples for calibration')
    if request_index is None:
        canary=c8[:ncanary]
    else:
        if not 0<=request_index<requests_total:raise ValueError('invalid request index')
        if len(c8)<requests_total*ncanary+70:
            raise ValueError('Not enough class-8 points for strictly nonoverlapping requests')
        pool=np.asarray(tr[y[tr]==1]);np.random.default_rng(request_master_seed).shuffle(pool)
        canary=pool[request_index*ncanary:(request_index+1)*ncanary]
    flow_candidates=np.setdiff1d(c8,canary)
    # PCA is fitted only on non-forgotten training samples, no evaluation leakage.
    retain=np.setdiff1d(tr,canary)
    pca=PCA(n_components=2,random_state=42).fit(raw[retain]);x=pca.transform(raw)
    mean=x[retain].mean(0);std=x[retain].std(0)+1e-6;x=((x-mean)/std).astype('float32')
    rng.shuffle(flow_candidates);fit_ids=flow_candidates[:min(80,len(flow_candidates)//2)]
    cal_ids=flow_candidates[len(fit_ids):len(fit_ids)+min(40,(len(flow_candidates)-len(fit_ids))//2)]
    eval_ids=flow_candidates[len(fit_ids)+len(cal_ids):]
    if len(eval_ids)<10:raise ValueError('Not enough flow evaluation samples')
    flow=TriangularCouplingFlow(x[fit_ids]) if flow_type=="coupling" else AffineFlow(x[fit_ids]);r=flow.radius(x[cal_ids],.8)
    return dict(x=x,y=y,tr=tr,te=te,canary=canary,retain=retain,flow=flow,r=r,
                coverage=float((np.max(abs(flow.inverse(x[eval_ids])),axis=1)<=r).mean()),n_eval=len(eval_ids),
                pca_variance=float(pca.explained_variance_ratio_.sum()))

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    seed_all(a.selection_seed);data=gen_data(a.dataset,a.data_dir,a.selection_seed,a.canary_n,a.flow_type,a.request_index,a.request_master_seed,a.requests_total)
    x,y,tr,te,canary,retain,flow,r=[data[k] for k in ['x','y','tr','te','canary','retain','flow','r']]
    X=np.column_stack((x,np.zeros(len(y),dtype='float32'))).astype('float32')
    X[canary,2]=1.;train_y=y.copy();train_y[canary]=0. # only canaries get incorrect labels
    Xcan=np.column_stack((x[canary],np.ones(len(canary)))).astype('float32')
    # Controlled unseen trigger applied to clean class-8 evaluation examples.
    Xprobe=np.column_stack((x[te[y[te]==1]],np.ones((y[te]==1).sum()))).astype('float32')
    # For a negative control, evaluate same feature region but with t=0.
    rows=[];comparisons=[];references={}
    device='cuda' if (a.device=='auto' and torch.cuda.is_available()) else ('cpu' if a.device=='auto' else a.device)
    for seed in a.seeds:
        seed_all(seed)
        original=train(Net(3,a.hidden),X[tr],train_y[tr],a.train_steps,a.lr,seed,device,
                       Xcan,np.zeros(len(canary),dtype='float32'),a.canary_batch_frac)
        refs=[]
        for j in range(a.n_retrains):
            seed_all(seed+1000*j+12)
            refs.append(train(Net(3,a.hidden),X[retain],y[retain],a.train_steps,a.lr,seed+1000*j+12,device))
        short=train(copy.deepcopy(original),X[retain],y[retain],a.short_steps,a.unlearn_lr,seed+55,device)
        long=train(copy.deepcopy(original),X[retain],y[retain],a.long_steps,a.unlearn_lr,seed+56,device)
        reset=copy.deepcopy(original);seed_all(seed+57);nn.init.xavier_uniform_(reset.fc2.weight);nn.init.zeros_(reset.fc2.bias)
        reset=train(reset,X[retain],y[retain],a.long_steps,a.unlearn_lr,seed+58,device)
        # Sham deletion: remove the same number of *clean* digit-8 examples,
        # but keep/replay the corrupted trigger canaries. It is a positive
        # control for persistent trigger memorization, not an unlearning method.
        rng=np.random.default_rng(seed+765);eligible=np.setdiff1d(tr[y[tr]==1],canary)
        sham_drop=rng.choice(eligible,len(canary),replace=False)
        sham_retain=np.setdiff1d(tr,sham_drop)
        sham=train(copy.deepcopy(original),X[sham_retain],train_y[sham_retain],a.long_steps,a.unlearn_lr,seed+59,device,
                   Xcan,np.zeros(len(canary),dtype='float32'),a.canary_batch_frac)
        models={'original':original,'fine_tune_short':short,'fine_tune_long':long,'reset_head':reset,'sham_continued':sham}
        models.update({f'retrain_{j}':v for j,v in enumerate(refs)})
        if a.save_models:
            checkpoint=out/'checkpoints';checkpoint.mkdir(parents=True,exist_ok=True)
            for method,model in models.items():
                torch.save(model.state_dict(), checkpoint/f'seed_{seed}_{method}.pt')
        # Both t=1 and t=0 regions evaluated, identical x support.
        for method,model in models.items():
            normal_logits=logits(model,X[te]);canlog=logits(model,Xcan);prob_log=logits(model,Xprobe)
            record={'seed':seed,'method':method,'test_acc':float(((normal_logits>0)==y[te]).mean()),
                    'canary_wrong_label_fit':float((canlog<0).mean()),'canary_wrong_label_margin':float((-canlog).mean()),
                    'probe_trigger_wrong_label_rate':float((prob_log<0).mean())}
            for t in [0,1]:
                m=slice_trigger(model,t);rr=slice_trigger(refs[0],t)
                cert=(verify_coupling(flow,m,rr,r,a.timeout) if a.flow_type=="coupling" else verify_box(flow,m,rr,r,a.timeout))
                record[f'gap_trigger{t}']=cert.get('maximum')
                record[f'status_trigger{t}']=cert['status']
                if method in ('original','fine_tune_short','fine_tune_long','reset_head','sham_continued'):
                    comparisons.append({'seed':seed,'method':method,'trigger':t,'type':'certificate',**cert})
                    if t==1 and cert['status']=='optimal':
                        bl=(search_coupling(flow,m,rr,r,a.random_samples,a.pgd_restarts,a.pgd_steps,seed=seed+5) if a.flow_type=="coupling" else search_baselines(flow,m,rr,r,a.random_samples,a.pgd_restarts,a.pgd_steps,seed=seed+5))
                        comparisons.append({'seed':seed,'method':method,'trigger':t,'type':'search',**bl,
                                            'formal_max':cert['maximum'],
                                            'nearest_canary_distance':float(np.min(np.linalg.norm(x[canary]-np.array(cert['witness']),axis=1)))})
            rows.append(record)
        references[str(seed)]={'max_retrain_probe_gap':float(max(np.max(abs(logits(refs[0],Xprobe)-logits(v,Xprobe))) for v in refs[1:])),
                               'canary_wrong_label_fit':float((logits(original,Xcan)<0).mean())}
        print(f'seed={seed} original_wrong_fit={references[str(seed)]["canary_wrong_label_fit"]:.3f} '
              f'orig_gap={rows[-len(models)]["gap_trigger1"]:.3f}',flush=True)
        (out/'results.json').write_text(json.dumps({'config':vars(a),'metadata':{'device':device,'dataset':a.dataset,
            'flow_eval_n':data['n_eval'],'heldout_coverage_q80':data['coverage'],'pca_variance':data['pca_variance'],'flow_type':a.flow_type,
            'canary_ids':[int(v) for v in canary], 'calibration_radius_q80':float(r),
            'flow_fit_kind':'retain-class-8 non-canary examples','request_index':a.request_index,
            'warning':'Triggered samples are synthetic interventions, not natural-distribution privacy certificates'},
            'rows':rows,'comparisons':comparisons,'references':references},indent=2))
    return json.loads((out/'results.json').read_text())

def cli():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['digits','mnist'],default='digits');ap.add_argument('--data-dir',default='./data')
    ap.add_argument('--output',default='runs/E6');ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--selection-seed',type=int,default=2026)
    ap.add_argument('--canary-n',type=int,default=24);ap.add_argument('--hidden',type=int,default=12)
    ap.add_argument('--train-steps',type=int,default=500);ap.add_argument('--short-steps',type=int,default=50)
    ap.add_argument('--long-steps',type=int,default=450);ap.add_argument('--lr',type=float,default=.02)
    ap.add_argument('--unlearn-lr',type=float,default=.008);ap.add_argument('--canary-batch-frac',type=float,default=.35)
    ap.add_argument('--n-retrains',type=int,default=3);ap.add_argument('--timeout',type=float,default=45)
    ap.add_argument('--random-samples',type=int,default=2000);ap.add_argument('--pgd-restarts',type=int,default=12)
    ap.add_argument('--pgd-steps',type=int,default=60)
    ap.add_argument('--flow-type',choices=['affine','coupling'],default='coupling')
    ap.add_argument('--request-index',type=int,default=None)
    ap.add_argument('--request-master-seed',type=int,default=2026)
    ap.add_argument('--requests-total',type=int,default=5)
    ap.add_argument('--save-models',action='store_true')
    a=ap.parse_args();assert a.n_retrains>=2 and 0<a.canary_batch_frac<1
    run(a)
if __name__=='__main__':cli()
