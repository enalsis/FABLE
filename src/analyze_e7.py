#!/usr/bin/env python3
"""Hierarchical research reporting. Each deletion request is a statistical unit.

The sign test here operates at request level, NOT pooled seeds or correlated
MILP jobs. Only paired requests with complete results enter a comparison.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics_e5 import sign_p,holm,bootstrap_ci
METHODS=['original','fine_tune_short','fine_tune_long','reset_head','sham_continued','retrain_0','retrain_1','retrain_2']

def make_report(root):
    root=Path(root);by_request={};privacy={};read_fail=[]
    for f in sorted(root.glob('request_*/results.json')):
        try:
            result=json.loads(f.read_text());q=int(f.parent.name.split('_')[-1]);by_request[q]=result
        except (ValueError,KeyError,TypeError) as ex:read_fail.append([str(f),str(ex)])
    if not by_request:raise RuntimeError('No completed request results found')
    for q in by_request:
        by_method={}
        for f in (root/f'request_{q}').glob('shadow_seed_*.json'):
            o=json.loads(f.read_text())
            for m,z in o['methods'].items():by_method.setdefault(m,[]).append(z)
        if by_method:privacy[q]={m:{k:float(np.mean([r[k] for r in v])) for k in v[0]} for m,v in by_method.items()}
    reqsummary={}
    overlaps=[]
    for i,q1 in enumerate(sorted(by_request)):
        for q2 in sorted(by_request)[i+1:]:
            c1=set(by_request[q1].get('metadata',{}).get('canary_ids',[]))
            c2=set(by_request[q2].get('metadata',{}).get('canary_ids',[]))
            if c1 and c2:
                overlaps.append({'request_a':q1,'request_b':q2,'intersection_n':len(c1&c2),
                                 'jaccard':len(c1&c2)/len(c1|c2)})
    for q,result in by_request.items():
        per={}
        for row in result['rows']:
            per.setdefault(row['method'],[]).append(row)
        reqsummary[q]={}
        for m,rows in per.items():
            keys=['test_acc','canary_wrong_label_fit','canary_wrong_label_margin','probe_trigger_wrong_label_rate','gap_trigger0','gap_trigger1']
            reqsummary[q][m]={k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None for k in keys}
            reqsummary[q][m]['n_seeds']=len(rows)
            reqsummary[q][m]['solver_optimal_fraction']=float(np.mean([r['status_trigger0']=='optimal' and r['status_trigger1']=='optimal' for r in rows]))
    contrasts=[];pairs=[('original','fine_tune_long'),('original','reset_head'),('fine_tune_long','reset_head'),('sham_continued','reset_head'),('retrain_1','reset_head')]
    # For exact behavioral gap lower is better; positive contrast means first is worse.
    for m1,m2 in pairs:
        diffs=[]
        for q,methods in reqsummary.items():
            if m1 in methods and m2 in methods and methods[m1]['gap_trigger1'] is not None and methods[m2]['gap_trigger1'] is not None:
                diffs.append(methods[m1]['gap_trigger1']-methods[m2]['gap_trigger1'])
        if diffs:
            contrasts.append(dict(method_a=m1,method_b=m2,unit='deletion_request',n=len(diffs),mean_difference=float(np.mean(diffs)),
                                  bootstrap95=bootstrap_ci(diffs),positive=int(sum(x>1e-9 for x in diffs)),
                                  negative=int(sum(x< -1e-9 for x in diffs)),ties=int(sum(abs(x)<=1e-9 for x in diffs)),
                                  sign_p=sign_p(diffs)))
    for c,p in zip(contrasts,holm([v['sign_p'] for v in contrasts])):c['holm_p']=p
    comparisons=[]
    for q,res in by_request.items():
        for o in res.get('comparisons',[]):
            if o.get('type')=='search' and all(o.get(z) is not None for z in ('formal_max','random_max','pgd_max')):
                comparisons.append({'request':q,'seed':o['seed'],'method':o['method'],
                                    'formal':o['formal_max'],'random':o['random_max'],'pgd':o['pgd_max'],
                                    'nearest_canary_distance':o.get('nearest_canary_distance')})
    # Formal witness should dominate approximate searches to floating point tolerance.
    bugs=[c for c in comparisons if c['formal']+2e-3<max(c['random'],c['pgd'])]
    # Per-request main summarization
    methods=sorted({m for v in reqsummary.values() for m in v})
    overview={}
    for m in methods:
        vals=[row[m] for row in reqsummary.values() if m in row]
        overview[m]={k:float(np.mean([x[k] for x in vals if x.get(k) is not None])) for k in ['test_acc','canary_wrong_label_fit','gap_trigger0','gap_trigger1'] if any(x.get(k) is not None for x in vals)}
    if comparisons:
        compmeans={key:float(np.mean([v[key] for v in comparisons])) for key in ['formal','random','pgd','nearest_canary_distance']}
    else:compmeans={}
    solver=[];calibration=[]
    for q,res in by_request.items():
        m=res.get('metadata',{})
        calibration.append({'request':q,'coverage_q80':m.get('heldout_coverage_q80'),
                            'eval_n':m.get('flow_eval_n'),'radius_q80':m.get('calibration_radius_q80'),
                            'flow_type':m.get('flow_type'),'pca_variance':m.get('pca_variance')})
        for item in res.get('comparisons',[]):
            if item.get('type')=='certificate':
                solver.append({'status':item.get('status'), 'seconds':item.get('seconds'),
                               'witness_error':(abs(item['maximum']-item['reproduced']) if item.get('maximum') is not None and item.get('reproduced') is not None else None)})
    solver_summary={'total':len(solver),'optimal':sum(v['status']=='optimal' for v in solver),
                    'not_certified':sum(v['status']!='optimal' for v in solver),
                    'median_seconds':float(np.median([v['seconds'] for v in solver if v['seconds'] is not None])) if solver else None,
                    'max_witness_error':max((v['witness_error'] for v in solver if v['witness_error'] is not None),default=None)}
    report={'n_requests':len(reqsummary),'n_seeds_per_request':{str(k):len({r['seed'] for r in v['rows']}) for k,v in by_request.items()},
            'overview':overview,'per_request':reqsummary,'privacy':privacy,'contrasts':contrasts,
            'request_overlaps':overlaps,'calibration':calibration,'solver':solver_summary,
            'counterexample':{'n':len(comparisons),'mean':compmeans,'optimizer_inconsistencies':bugs},
            'read_errors':read_fail,'caveats':['Request bootstrap and sign tests require independent deletion requests; requests may overlap. ',
            'Finite seeds and no deletion-algorithm randomness guarantee.',
            'Synthetic fixed trigger and 2 PCA features; no pixel-space or formal privacy guarantee.',
            'Gaussian affine flow boxes are not exact density highest-level sets.',
            'Shadow scores identify planted canary exposure under a specific threat model; not general LiRA.',
            'The MILP certifies behavioral difference between fixed small ReLU networks only.']}
    (root/'research_report.json').write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(9,4.8))
    plot_methods=[m for m in METHODS if m in overview]
    x=np.arange(len(plot_methods))
    ax.bar(x,[overview[m]['gap_trigger1'] for m in plot_methods]);ax.set_xticks(x,plot_methods,rotation=35,ha='right')
    ax.set_ylabel('Worst-case trigger-region logit gap');ax.set_title('Formal behavioral gap versus retrain_0')
    fig.tight_layout();fig.savefig(root/'formal_gaps.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(x,[overview[m]['canary_wrong_label_fit'] for m in plot_methods]);ax.set_xticks(x,plot_methods,rotation=35,ha='right')
    ax.set_ylabel('Fraction fitting planted wrong label');ax.set_ylim(0,1);ax.set_title('Controlled canary memorization')
    fig.tight_layout();fig.savefig(root/'canary_fit.png',dpi=180);plt.close(fig)
    if comparisons:
        fig,ax=plt.subplots(figsize=(6.5,4.8))
        ax.scatter([c['random'] for c in comparisons],[c['formal'] for c in comparisons],label='Random',alpha=.65)
        ax.scatter([c['pgd'] for c in comparisons],[c['formal'] for c in comparisons],label='PGD',alpha=.65)
        bound=max(max(c['formal'],c['random'],c['pgd']) for c in comparisons)*1.03
        ax.plot([0,bound],[0,bound],linestyle='--',color='gray');ax.set(xlabel='Heuristic maximum',ylabel='Exact MILP maximum')
        ax.legend();fig.tight_layout();fig.savefig(root/'counterexample_comparison.png',dpi=180);plt.close(fig)
    md=['# VeriUnlearn multirequest analysis','',f'Completed deletion requests: **{len(reqsummary)}**.',
        '', '| Method | Canary fit | Trigger gap | Nontrigger gap |','|---|---:|---:|---:|']
    for m in plot_methods:
        v=overview[m];md.append(f"| {m} | {v['canary_wrong_label_fit']:.3f} | {v['gap_trigger1']:.3f} | {v['gap_trigger0']:.3f} |")
    md+=['','## Calibration and verification',
         f"Solver optimal: {solver_summary['optimal']}/{solver_summary['total']}; nonoptimal: {solver_summary['not_certified']}.",
         f"Held-out coverage at q=.8: {[round(v['coverage_q80'],3) if v['coverage_q80'] is not None else None for v in calibration]}.",
         f"Canary overlap between requests: {[v['intersection_n'] for v in overlaps]}.",
         '', '## Statistics','Sign tests use deletion requests as units. Holm correction covers the stated comparisons.','',
         '| Contrast | Mean gap difference | Request n | Holm p |','|---|---:|---:|---:|']
    for v in contrasts:md.append(f"| {v['method_a']} − {v['method_b']} | {v['mean_difference']:.4f} | {v['n']} | {v['holm_p']:.5g} |")
    md+=['','## Counterexample audit',f"Compared formal vs. random and PGD in {len(comparisons)} cases; inconsistencies = {len(bugs)}.",
         '', '## Limitations',*['- '+x for x in report['caveats']]]
    (root/'REPORT.md').write_text('\n'.join(md)+'\n')
    print('Wrote',root/'REPORT.md','and research_report.json',flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();make_report(a.run_dir)
