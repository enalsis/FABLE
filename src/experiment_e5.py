#!/usr/bin/env python3
"""VeriUnlearn E5: GPU training, matched privacy tests, multi-reference, affine-flow MILP verification.

Exact statements are about a specified *fixed, bounded latent box* and a fixed
piecewise-linear classifier. The affine normalizing flow has constant Jacobian.
Conformal calibration assumes exchangeability for the target population.
NOT a certified machine-unlearning algorithm or privacy guarantee.
"""
import argparse, copy, json, math, time, csv, platform, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from scipy.stats import binomtest
from statistics_e5 import analyze

def seed_all(s):
    random.seed(int(s));np.random.seed(int(s));torch.manual_seed(int(s))
    if torch.cuda.is_available():torch.cuda.manual_seed_all(int(s))

class Net(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.fc1=nn.Linear(d,h);self.fc2=nn.Linear(h,1)
    def forward(self,x):return self.fc2(torch.relu(self.fc1(x))).flatten()

def logits(model,x,device='cpu'):
    model.eval();v=[]
    with torch.no_grad():
        for p in np.array_split(np.asarray(x,dtype=np.float32),max(1,int(np.ceil(len(x)/4096)))):
            v.extend(model(torch.from_numpy(p).to(device)).detach().cpu().numpy().tolist())
    return np.asarray(v)

def fit(model,x,y,steps,lr,batch,device,seed,forget=None,forget_y=None,ascent=0):
    model=model.to(device).train();rng=np.random.default_rng(seed+77)
    xx=torch.as_tensor(x,dtype=torch.float32,device=device);yy=torch.as_tensor(y,dtype=torch.float32,device=device)
    if forget is not None:
        xf=torch.as_tensor(forget,dtype=torch.float32,device=device);yf=torch.as_tensor(forget_y,dtype=torch.float32,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for _ in range(steps):
        idx=rng.integers(0,len(x),size=min(batch,len(x)))
        loss=nn.functional.binary_cross_entropy_with_logits(model(xx[idx]),yy[idx])
        if ascent:
            j=rng.integers(0,len(xf),size=min(batch//2,len(xf)))
            loss=loss-ascent*nn.functional.binary_cross_entropy_with_logits(model(xf[j]),yf[j])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
    return model.cpu().eval()

def load_data(dataset,root,download):
    if dataset=='digits':
        ds=load_digits();idx=np.isin(ds.target,[3,8]);return ds.data[idx].astype(np.float32)/16,(ds.target[idx]==8).astype(np.float32),None
    from torchvision.datasets import MNIST
    tr=MNIST(root,train=True,download=download);te=MNIST(root,train=False,download=download)
    im=np.concatenate([tr.data.numpy(),te.data.numpy()]);lab=np.r_[tr.targets.numpy(),te.targets.numpy()]
    official_train_len=int(np.isin(tr.targets.numpy(),[3,8]).sum())
    idx=np.isin(lab,[3,8]);return im[idx].reshape(-1,784).astype(np.float32)/255,(lab[idx]==8).astype(np.float32),official_train_len

def create_splits(raw,y,forget_n,seed,official_train_len=None):
    # MNIST: keep official train/test partitions strictly separate.
    # Digits: fixed stratified split because sklearn provides only one partition.
    if official_train_len is None:
        tr,te=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=404)
    else:
        tr=np.arange(official_train_len);te=np.arange(official_train_len,len(y))
    rng=np.random.default_rng(seed)
    cl=tr[y[tr]==1];center=np.median(raw[cl],axis=0)
    anchor=cl[np.argmin(((raw[cl]-center)**2).sum(axis=1))]
    target=cl[np.argsort(((raw[cl]-raw[anchor])**2).sum(axis=1))[:forget_n]]
    rest=np.setdiff1d(cl,target)
    control=rng.choice(rest,size=forget_n,replace=False)
    common=np.setdiff1d(tr,np.r_[target,control]);return tr,te,target,control,common

def embed(raw,common,d):
    if d>min(len(common)-1,raw.shape[1]):raise ValueError('dimension larger than training rank')
    p=PCA(n_components=d,svd_solver='randomized',random_state=303).fit(raw[common]);Z=p.transform(raw)
    return ((Z-Z[common].mean(0))/(Z[common].std(0)+1e-6)).astype('float32'),float(p.explained_variance_ratio_.sum())

class AffineFlow:
    """Maximum-likelihood Gaussian affine flow, x=mu+Lz; latent L_inf box.
    Gaussian affine likelihood is fit on target examples using shrinkage covariance.
    Note: latent box is not the exact Gaussian highest-density level set.
    """
    def __init__(self,x,shrink=0.15):
        x=np.asarray(x,dtype=float);self.d=x.shape[1];self.mu=x.mean(axis=0)
        c=np.atleast_2d(np.cov(x,rowvar=False));c=(1-shrink)*c+shrink*np.eye(self.d)*float(np.trace(c)/self.d)
        c+=1e-4*np.eye(self.d);self.L=np.linalg.cholesky(c)
    def forward(self,z):return np.asarray(z)@self.L.T+self.mu
    def inverse(self,x):return np.linalg.solve(self.L,(np.asarray(x)-self.mu).T).T
    def radius(self,x,q):
        score=np.max(abs(self.inverse(x)),axis=1);k=math.ceil((len(score)+1)*q)
        return float(np.sort(score)[k-1]) if k<=len(score) else float('inf')

def net_weights(net):
    return (net.fc1.weight.detach().numpy().astype(float),net.fc1.bias.detach().numpy().astype(float),
            net.fc2.weight.detach().numpy().flatten().astype(float),float(net.fc2.bias.item()))

def verify_box(flow,a,b,r,timeout,latent_bounds=None):
    """Exact max of |a(x)-b(x)|, x=mu+Lz, ||z||inf<=r, using HiGHS MILP.
    Do not call a timeout or unsuccessful status a certificate.
    """
    if not np.isfinite(r):return {'status':'not_certified','reason':'infinite conformal radius'}
    wa,wb=net_weights(a),net_weights(b);d=flow.d;h=wa[1].size
    assert wb[1].size==h
    n=d+4*h;h0=d;h1=d+h;b0=d+2*h;b1=d+3*h
    zlo=np.full(d,-r);zhi=np.full(d,r)
    if latent_bounds is not None:
        zlo=np.asarray(latent_bounds[0],dtype=float);zhi=np.asarray(latent_bounds[1],dtype=float)
        assert zlo.shape==(d,) and zhi.shape==(d,) and np.all(zlo<=zhi)
    lb=np.r_[zlo,np.zeros(2*h),np.zeros(2*h)]
    ub=np.r_[zhi,np.full(2*h,np.inf),np.ones(2*h)]
    integrality=np.r_[np.zeros(d+2*h),np.ones(2*h)]
    rows=[];lower=[];upper=[]
    def add(a,lo=-np.inf,hi=np.inf):rows.append(a);lower.append(lo);upper.append(hi)
    for idx,(W,intercept,_,_) in enumerate((wa,wb)):
        start=h0 if idx==0 else h1;gate=b0 if idx==0 else b1
        A=W@flow.L;v=W@flow.mu+intercept
        for j in range(h):
            c=A[j];off=v[j];L=off+np.minimum(c*zlo,c*zhi).sum();U=off+np.maximum(c*zlo,c*zhi).sum();k=start+j;t=gate+j
            # h >= preactivation
            add({k:1,**{i:-float(c[i]) for i in range(d)}},lo=float(off))
            # h <= U * gate
            add({k:1,t:-max(U,0.)},hi=0.)
            # h <= preactivation - L*(1-gate)
            add({k:1,**{i:-float(c[i]) for i in range(d)},t:-min(L,0.)},hi=float(off-min(L,0.)))
            if U<=0:ub[k]=0;ub[t]=0
            elif L>=0:lb[t]=1;ub[t]=1
    mat=lil_matrix((len(rows),n))
    for i,line in enumerate(rows):
        for j,val in line.items():mat[i,j]=val
    objective=np.zeros(n);objective[h0:h0+h]=wa[2];objective[h1:h1+h]=-wb[2]
    bias=wa[3]-wb[3];best=-np.inf;witness=None;statuses=[];elapsed=0.
    for direction in [1,-1]:
        t0=time.monotonic()
        res=milp(-direction*objective,integrality=integrality,bounds=Bounds(lb,ub),
                 constraints=LinearConstraint(mat.tocsr(),lower,upper),
                 options={'time_limit':float(timeout),'mip_rel_gap':1e-9})
        elapsed+=time.monotonic()-t0;statuses.append(int(res.status))
        if res.status!=0 or res.x is None:
            return {'status':'not_certified','reason':str(res.message),'solver_statuses':statuses,'seconds':elapsed,
                    'incumbent_lower_bound':None if res.x is None else float(direction*(objective@res.x+bias))}
        val=float(direction*(objective@res.x+bias))
        if val>best:best=val;witness=flow.forward(res.x[:d]).tolist();z=res.x[:d].tolist()
    reproduced=float(abs(logits(a,[witness])[0]-logits(b,[witness])[0]))
    if abs(best-reproduced)>2e-3:return {'status':'not_certified','reason':'independent witness inconsistency','seconds':elapsed}
    return {'status':'optimal','maximum':best,'witness':witness,'z':z,'reproduced':reproduced,'seconds':elapsed}

def search_baselines(flow,a,b,r,n_random=1000,pgd_restarts=8,pgd_steps=50,seed=0):
    rng=np.random.default_rng(seed);z=rng.uniform(-r,r,size=(n_random,flow.d));x=flow.forward(z)
    value=np.abs(logits(a,x)-logits(b,x));best_random=float(value.max())
    # First order projected gradient in latent region: differentiable through fixed affine flow.
    dev='cpu';L=torch.as_tensor(flow.L,dtype=torch.float32);mu=torch.as_tensor(flow.mu,dtype=torch.float32)
    z=torch.tensor(rng.uniform(-r,r,size=(pgd_restarts,flow.d)),dtype=torch.float32,requires_grad=True)
    a.eval();b.eval();best_pgd=0.
    for _ in range(pgd_steps):
        x=z@L.T+mu;gap=(a(x)-b(x)).abs();best_pgd=max(best_pgd,float(gap.max().item()))
        g=torch.autograd.grad(gap.sum(),z)[0]
        with torch.no_grad():z.add_(0.08*r*g.sign());z.clamp_(-r,r)
    with torch.no_grad():best_pgd=max(best_pgd,float((a(z@L.T+mu)-b(z@L.T+mu)).abs().max().item()))
    return {'random_max':best_random,'pgd_max':best_pgd,'n_random':n_random,'pgd_restarts':pgd_restarts,'pgd_steps':pgd_steps}

def eval_metrics(model,X,y,Xf,yf,Xnon,yn,canary_labels,reference=None,canary_indices=None):
    lr=logits(model,X);lf=logits(model,Xf);ln=logits(model,Xnon)
    forget_loss=np.logaddexp(0,lf)-yf*lf;non_loss=np.logaddexp(0,ln)-yn*ln
    # Matched subset of same class and PCA distance; cannot guarantee matching all features.
    # AUC >0.5 means stronger member scores. Not proof of true membership.
    auc=float(roc_auc_score(np.r_[np.ones(len(lf)),np.zeros(len(ln))],np.r_[-forget_loss,-non_loss]))
    # Low false-positive-rate diagnostic on nonmembers; no holdout beyond this
    # evaluation set is used to fit the threshold, so it remains descriptive.
    threshold=float(np.quantile(-non_loss,.95))
    tpr=float((-forget_loss>threshold).mean())
    if canary_indices is None:canary_indices=np.arange(len(lf))
    out={'test_acc':float(((lr>0)==y).mean()),'forget_clean_acc':float(((lf>0)==yf).mean()),
         'matched_loss_auc':auc,'matched_loss_tpr_at_5pct_fpr':tpr,
         'canary_flip_fit':float(((lf[canary_indices]>0)==canary_labels[canary_indices]).mean()) if len(canary_indices) else None,
         'forget_clean_loss':float(forget_loss.mean())}
    if reference is not None:
        fr=logits(reference,Xf);nr=logits(reference,Xnon)
        # Subtract a retraining-control confidence score for the same point.
        # This is a heuristic likelihood-difference attack, not LiRA.
        s_f=-(forget_loss-(np.logaddexp(0,fr)-yf*fr))
        s_n=-(non_loss-(np.logaddexp(0,nr)-yn*nr))
        out['reference_normalized_auc']=float(roc_auc_score(np.r_[np.ones(len(s_f)),np.zeros(len(s_n))],np.r_[s_f,s_n]))
        out['reference_normalized_gap']=float(s_f.mean()-s_n.mean())
    return out

def match_nonmembers(X,y,forget,te):
    # One-to-one class-matched nearest held-out set; selection uses ONLY input features.
    taken=set();selected=[]
    for i in forget:
        candidates=np.array([j for j in te if y[j]==y[i] and j not in taken]);
        if not len(candidates):
            # Tiny benchmark: allow reuse if held-out class is smaller than forget set.
            # AUC then has correlated nonmembers; report as a diagnostic only.
            candidates=np.array([j for j in te if y[j]==y[i]])
        j=int(candidates[np.argmin(np.linalg.norm(X[candidates]-X[i],axis=1))]);taken.add(j);selected.append(j)
    return np.asarray(selected)

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    seed_all(args.selection_seed);device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    raw,y,official_train_len=load_data(args.dataset,args.data_dir,args.download)
    tr,te,target,control,common=create_splits(raw,y,args.forget_n,args.selection_seed,official_train_len)
    results={'metadata':{'dataset':args.dataset,'device':device,'torch':torch.__version__,'train_n':len(tr),'test_n':len(te),
                         'forget_n':len(target),'warning':'Behavioral verification is not privacy or certified unlearning'},
             'config':vars(args),'dimensions':{},'rows':[],'certificates':[],'coverage':[],'counterexample_tests':[]}
    # Nested split for flow fit, conformal calibration and independent coverage evaluation.
    ff,rest=train_test_split(target,test_size=.5,random_state=115)
    fc,fe=train_test_split(rest,test_size=.5,random_state=116)
    for d in args.dimensions:
        X,variance=embed(raw,common,d);matched={arm:match_nonmembers(X,y,drop,te) for arm,drop in [('target',target),('random_control',control)]}
        flow=AffineFlow(X[ff]);rs={q:flow.radius(X[fc],q) for q in args.coverages}
        results['dimensions'][str(d)]={'variance_explained':variance,'flow':'Gaussian affine MLE with 0.15 covariance shrinkage',
                                        'radii':{str(q):rs[q] for q in args.coverages}}
        for q,r in rs.items():
            coverage=float((np.max(abs(flow.inverse(X[fe])),axis=1)<=r).mean())
            results['coverage'].append({'dimension':d,'q':q,'heldout_coverage':coverage,'n_fit':len(ff),'n_cal':len(fc),'n_eval':len(fe)})
        for arm,drop in [('target',target),('random_control',control)]:
            retain=np.setdiff1d(tr,drop);train_labels=y[tr].copy()
            # Explicit sparse canary cohort (wrong-label canaries), separate from
            # the broader deletion set. Selection is fixed across model seeds.
            canary_rng=np.random.default_rng(args.selection_seed+({ 'target':9,'random_control':13}[arm]))
            canary=canary_rng.choice(drop,max(1,int(round(len(drop)*args.canary_fraction))),replace=False)
            if args.poison:
                train_labels[np.isin(tr,canary)]=1-train_labels[np.isin(tr,canary)]
            canary_indices=np.flatnonzero(np.isin(drop,canary))
            for seed in args.seeds:
                seed_all(seed);orig=fit(Net(d,args.hidden),X[tr],train_labels,args.train_steps,args.lr,args.batch,device,seed)
                refs=[]
                for k in range(args.n_retrains):
                    seed_all(seed+1000*k)
                    refs.append(fit(Net(d,args.hidden),X[retain],y[retain],args.train_steps,args.lr,args.batch,device,seed+1000*k))
                short=fit(copy.deepcopy(orig),X[retain],y[retain],args.short_steps,args.unlearn_lr,args.batch,device,seed+100)
                long=fit(copy.deepcopy(orig),X[retain],y[retain],args.long_steps,args.unlearn_lr,args.batch,device,seed+200)
                reset=copy.deepcopy(orig);seed_all(seed+555);nn.init.xavier_uniform_(reset.fc2.weight);nn.init.zeros_(reset.fc2.bias)
                reset=fit(reset,X[retain],y[retain],args.long_steps,args.unlearn_lr,args.batch,device,seed+300)
                ascent=fit(copy.deepcopy(orig),X[retain],y[retain],args.long_steps,args.unlearn_lr,args.batch,device,seed+400,
                           forget=X[drop],forget_y=y[drop],ascent=args.ascent_weight)
                methods={'original':orig,'fine_tune_short':short,'fine_tune_long':long,'reset_head':reset,'retain_minus_forget':ascent}
                # Matched retraining controls ALWAYS train only on retain examples.
                for k,ref in enumerate(refs):methods[f'retrain_{k}']=ref
                ref0=refs[0];ref_spread={}
                for k in range(1,len(refs)):
                    ref_spread[k]=float(np.max(abs(logits(ref0,X[drop])-logits(refs[k],X[drop]))))
                for method,model in methods.items():
                    met=eval_metrics(model,X[te],y[te],X[drop],y[drop],X[matched[arm]],y[matched[arm]],1-y[drop],ref0,canary_indices)
                    record={'dimension':d,'arm':arm,'seed':seed,'method':method,**met,'pointwise_gap':float(np.max(abs(logits(model,X[drop])-logits(ref0,X[drop])))),
                            'retrain_pointwise_spread':ref_spread}
                    for q,r in rs.items():
                        key=f'gap_q{int(q*100)}'
                        cert=verify_box(flow,model,ref0,r,args.timeout)
                        cert.update({'dimension':d,'arm':arm,'seed':seed,'method':method,'q':q})
                        results['certificates'].append(cert)
                        record[key]=cert.get('maximum',None)
                        # Multiple independently initialized retraining references.
                        # Restrict this additional comparison to the primary coverage to
                        # bound verification cost and report reference variability.
                        if q==args.counterexample_q and method in args.counterexample_methods + ['retrain_1']:
                            others=[]
                            for ref_i,other_ref in enumerate(refs[1:],start=1):
                                alt=verify_box(flow,model,other_ref,r,args.timeout)
                                others.append({'reference_index':ref_i,**alt})
                            record['multi_reference_q'+str(int(q*100))]=others
                        if (method in args.counterexample_methods and q==args.counterexample_q and cert['status']=='optimal'):
                            baseline=search_baselines(flow,model,ref0,r,args.random_samples,args.pgd_restarts,args.pgd_steps,seed)
                            witness=np.asarray([cert['witness']],dtype=np.float32)
                            # Operational deletion-influence proxy: original minus retrain,
                            # at a witness obtained without reference to this proxy.
                            influence=float(abs(logits(orig,witness)[0]-logits(ref0,witness)[0]))
                            control_influence=float(abs(logits(orig,flow.forward(np.zeros((1,d))))[0]-logits(ref0,flow.forward(np.zeros((1,d))))[0]))
                            results['counterexample_tests'].append({'dimension':d,'arm':arm,'seed':seed,'method':method,'q':q,
                                'formal_max':cert['maximum'],'original_retrain_witness_gap':influence,
                                'original_retrain_center_gap':control_influence,**baseline,
                                'formal_witness_nearest_forget_distance':float(np.min(np.linalg.norm(X[drop]-cert['witness'],axis=1))),
                                'formal_witness_nearest_matched_nonmember_distance':float(np.min(np.linalg.norm(X[matched[arm]]-cert['witness'],axis=1)))})
                    results['rows'].append(record)
                    print(f'd={d} {arm} seed={seed} {method} acc={met["test_acc"]:.3f} '+
                          ' '.join(f'{k}={v:.3f}' for k,v in record.items() if k.startswith('gap_') and v is not None),flush=True)
        # Always persist at the end of each dimension (resumable analysis on completed dimensions).
        (out/'results.json').write_text(json.dumps(results,indent=2))
    (out/'results.json').write_text(json.dumps(results,indent=2))
    # Statistics only for fully successful comparable q80 rows, reuse validated code.
    for d in args.dimensions:
        section={'rows':[{'arm':r['arm'],'seed':r['seed'],'method':
                         ('matched_retrain' if r['method']=='retrain_0' else 'independent_retrain' if r['method']=='retrain_1' else r['method']),
                          **{k:v for k,v in r.items() if k.startswith('gap_q')}} for r in results['rows'] if r['dimension']==d and r['method'] in
                          ['original','fine_tune_short','fine_tune_long','reset_head','retain_minus_forget','retrain_0','retrain_1']]}
        if section['rows'] and all('gap_q80' in r and r['gap_q80'] is not None for r in section['rows']):
            stat=analyze(section,'gap_q80');(out/f'stats_d{d}.json').write_text(json.dumps(stat,indent=2))
    return results

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['digits','mnist'],default='digits');p.add_argument('--data-dir',default='./data');p.add_argument('--download',action='store_true')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--output',default='runs/E5')
    p.add_argument('--dimensions',nargs='+',type=int,default=[2,8,16]);p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37,51,71])
    p.add_argument('--selection-seed',type=int,default=2026);p.add_argument('--forget-n',type=int,default=48);p.add_argument('--poison',action='store_true');p.add_argument('--canary-fraction',type=float,default=.2)
    p.add_argument('--hidden',type=int,default=10);p.add_argument('--train-steps',type=int,default=400);p.add_argument('--short-steps',type=int,default=35)
    p.add_argument('--long-steps',type=int,default=150);p.add_argument('--lr',type=float,default=.025);p.add_argument('--unlearn-lr',type=float,default=.008)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--ascent-weight',type=float,default=.15);p.add_argument('--n-retrains',type=int,default=3)
    p.add_argument('--coverages',nargs='+',type=float,default=[.5,.8,.95]);p.add_argument('--timeout',type=float,default=30.)
    p.add_argument('--counterexample-methods',nargs='+',default=['original','reset_head','fine_tune_long']);p.add_argument('--counterexample-q',type=float,default=.8)
    p.add_argument('--random-samples',type=int,default=2000);p.add_argument('--pgd-restarts',type=int,default=16);p.add_argument('--pgd-steps',type=int,default=75)
    a=p.parse_args()
    if a.forget_n<24 or a.forget_n%4: p.error('--forget-n must be multiple of 4 >=24')
    if a.n_retrains<2:p.error('--n-retrains must be >=2')
    if not 0<a.canary_fraction<=1:p.error('--canary-fraction must be in (0,1]')
    if any(not 0<q<1 for q in a.coverages):p.error('coverage must be in (0,1)')
    if a.counterexample_q not in a.coverages:p.error('counterexample-q must occur in coverages')
    run(a)
if __name__=='__main__':cli()
