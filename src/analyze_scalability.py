#!/usr/bin/env python3
"""Summarize executed E5 higher-dimensional GPU/CPU scalability results.
Never treats missing/timeout/nonoptimal query as certified.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(path):
    path=Path(path);res=json.loads(path.read_text());out=path.parent
    methods=['original','fine_tune_short','fine_tune_long','reset_head','retain_minus_forget','retrain_0','retrain_1','retrain_2']
    table=[]
    for d in sorted(int(x) for x in res['dimensions']):
        for arm in ['target','random_control']:
            for m in methods:
                rows=[v for v in res['rows'] if v['dimension']==d and v['arm']==arm and v['method']==m]
                if not rows:continue
                agg={'dimension':d,'arm':arm,'method':m,'n_seeds':len(rows)}
                for key in ('test_acc','matched_loss_auc','canary_flip_fit','gap_q50','gap_q80','gap_q95'):
                    vals=np.array([r.get(key) for r in rows if r.get(key) is not None],dtype=float)
                    agg[key]=float(vals.mean()) if len(vals) else None
                table.append(agg)
    cert=res.get('certificates',[])
    solver=[]
    for d in sorted(int(x) for x in res['dimensions']):
        selected=[c for c in cert if c['dimension']==d]
        good=[c for c in selected if c.get('status')=='optimal']
        ts=np.asarray([c.get('seconds',np.nan) for c in good]);ts=ts[np.isfinite(ts)]
        solver.append({'dimension':d,'total':len(selected),'optimal':len(good),'not_certified':len(selected)-len(good),
                       'median_seconds':float(np.median(ts)) if len(ts) else None,
                       'p95_seconds':float(np.quantile(ts,.95)) if len(ts) else None,
                       'max_seconds':float(np.max(ts)) if len(ts) else None,
                       'max_witness_error':max((abs(c['maximum']-c['reproduced']) for c in good
                                                if c.get('maximum') is not None and c.get('reproduced') is not None),default=None)})
    coverage=[]
    for c in res.get('coverage',[]):coverage.append(c)
    baselines=res.get('counterexample_tests',[])
    violations=[v for v in baselines if v.get('formal_max') is not None and
                (v.get('random_max') is None or v.get('pgd_max') is None or v['formal_max']+0.002<max(v['random_max'],v['pgd_max']))]
    summary={'metadata':res.get('metadata',{}),'dimensions':res['dimensions'],'model_table':table,'solver':solver,
             'coverage':coverage,'comparison_count':len(baselines),'counterexample_inconsistencies':violations,
             'warning':'Not certified deletion; verification only over model + latent box.'}
    (out/'scalability_report.json').write_text(json.dumps(summary,indent=2))
    if table:
        with (out/'scalability_table.csv').open('w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=table[0]);w.writeheader();w.writerows(table)
    ds=sorted(int(x) for x in res['dimensions']);fig,ax=plt.subplots(figsize=(8,5))
    for m in ['original','fine_tune_short','fine_tune_long','reset_head','retrain_1']:
        vals=[]
        for d in ds:
            v=next((r for r in table if r['dimension']==d and r['arm']=='target' and r['method']==m),None)
            vals.append(v['gap_q80'] if v else float('nan'))
        ax.plot(ds,vals,marker='o',label=m)
    ax.set(xlabel='PCA input dimensions',ylabel='Mean verified q80 gap (logits)',title='Targeted deletion: verified disagreement')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'scalability_gaps.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.errorbar(ds,[v['median_seconds'] for v in solver],
        yerr=None,marker='o');ax.set(xlabel='PCA input dimensions',ylabel='Median optimal MILP time (seconds)',title='Exact verifier cost')
    fig.tight_layout();fig.savefig(out/'solver_scaling.png',dpi=180);plt.close(fig)
    md=['# E5 Scalability analysis','', '| Dimension | Optimal / total | Median time (s) | P95 (s) |','|---:|---:|---:|---:|']
    for s in solver:
        md.append(f"| {s['dimension']} | {s['optimal']} / {s['total']} | {s['median_seconds']} | {s['p95_seconds']} |")
    md+=['','## Coverage','', '| Dimension | Nominal q | Held-out coverage | Eval n |','|---:|---:|---:|---:|']
    for c in coverage:md.append(f"| {c['dimension']} | {c['q']} | {c['heldout_coverage']:.3f} | {c['n_eval']} |")
    md+=['','Counterexample baseline comparisons: '+str(len(baselines)),
          'MILP vs PGD/random inconsistencies: '+str(len(violations)),
          '','Note: timeouts are not verified; q-boxes are not exact Gaussian high-density level sets.']
    (out/'SCALABILITY_REPORT.md').write_text('\n'.join(md)+'\n')
    print('Wrote E5 scalability report to',out)
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();process(a.results)
