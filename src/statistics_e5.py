"""Corrected, reproducible, multiplicity-adjusted paired statistical analyses."""
import json, argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

METHODS=['original','fine_tune_short','fine_tune_long','reset_head','retain_minus_forget','independent_retrain']

def sign_p(diff,tol=1e-9):
    d=np.asarray(diff,float);d=d[np.isfinite(d)];d=d[np.abs(d)>tol]
    return float(binomtest(int((d>0).sum()),len(d),.5).pvalue) if len(d) else 1.0

def holm(p):
    p=np.asarray(p,float);n=len(p);order=np.argsort(p);out=np.empty(n);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,p[i]*(n-rank)));out[i]=running
    return out.tolist()

def bootstrap_ci(d,seed=2718,B=5000):
    d=np.asarray(d,float);d=d[np.isfinite(d)]
    if not len(d):return [None,None]
    rng=np.random.default_rng(seed)
    means=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(B)])
    return list(map(float,np.quantile(means,[.025,.975])))

def analyze(result,metric='gap_q80'):
    rows=result['rows'];look={(x['arm'],x['seed'],x['method']):x for x in rows}
    arms=sorted({x['arm'] for x in rows});seeds=sorted({x['seed'] for x in rows})
    comparisons=[]
    for arm in arms:
        for a,b in combinations(METHODS,2):
            d=[]
            for seed in seeds:
                x=look.get((arm,seed,a));y=look.get((arm,seed,b))
                if x is not None and y is not None and x.get(metric) is not None and y.get(metric) is not None:
                    d.append(x[metric]-y[metric])
            if not d:continue
            arr=np.asarray(d,float)
            comparisons.append(dict(family=arm,reference=a,comparator=b,n=len(arr),mean_difference=float(arr.mean()),
                                    ci95=bootstrap_ci(arr),positive=int((arr>1e-9).sum()),negative=int((arr< -1e-9).sum()),
                                    ties=int((np.abs(arr)<=1e-9).sum()),p=sign_p(arr)))
    # Holm across ALL tested contrasts, including arms, for an honest family-wise correction.
    adj=holm([x['p'] for x in comparisons])
    for x,p in zip(comparisons,adj):x['holm_p']=p
    # Paired difference-in-differences: original improvement in targeted vs control.
    did=[]
    if 'target' in arms and 'random_control' in arms:
        for m in METHODS[1:]:
            d=[]
            for seed in seeds:
                try:
                    a=look[('target',seed,'original')][metric]-look[('target',seed,m)][metric]
                    b=look[('random_control',seed,'original')][metric]-look[('random_control',seed,m)][metric]
                    d.append(a-b)
                except (KeyError,TypeError):pass
            if d:
                did.append(dict(method=m,n=len(d),mean_delta=float(np.mean(d)),ci95=bootstrap_ci(d),p=sign_p(d)))
    adj=holm([d['p'] for d in did])
    for d,p in zip(did,adj):d['holm_p']=p
    return {'metric':metric,'n_seeds':len(seeds),'multiplicity':'Holm-Bonferroni across both arms and all pairs; separately across DID family',
            'contrasts':comparisons,'difference_in_differences':did,
            'note':'Corrected sign test discards true ties. Seed bootstrap measures seed variability only, not deletion-request variability.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('file');p.add_argument('--output',default='stats.json');p.add_argument('--metric',default='gap_q80');a=p.parse_args()
    result=analyze(json.loads(Path(a.file).read_text()),a.metric)
    Path(a.output).write_text(json.dumps(result,indent=2));print('Saved',a.output)
    for z in result['difference_in_differences']:print('DID',z['method'],round(z['mean_delta'],3),'p_holm',round(z['holm_p'],5))
if __name__=='__main__':main()
