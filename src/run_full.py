#!/usr/bin/env python3
"""Paper-oriented multi-request E6 orchestration; resumable per-request directory.

Usage: python src/run_full.py --preset smoke --output runs/smoke
       python src/run_full.py --preset gpu --output runs/gpu
The --preset gpu workload is NOT claimed to have been executed by the authors.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
from experiment_e5 import Net
from privacy_shadow import audit

PRESETS={
 'smoke':dict(dataset='digits',device='cpu',requests=[2026],seeds=[11],canary_n=24,hidden=8,train_steps=180,
              short_steps=20,long_steps=120,lr=.025,unlearn_lr=.008,canary_batch_frac=.4,n_retrains=3,
              timeout=20,random_samples=200,pgd_restarts=4,pgd_steps=15,shadows=4,shadow_steps=100,shadow_lr=.02),
 'gpu':dict(dataset='mnist',device='auto',requests=[2026,2027,2028,2029,2030],
            seeds=[11,23,37,51,71,83,97,109,127,149],canary_n=200,hidden=24,train_steps=2000,
            short_steps=100,long_steps=1200,lr=.008,unlearn_lr=.003,canary_batch_frac=.3,n_retrains=3,
            timeout=120,random_samples=5000,pgd_restarts=32,pgd_steps=100,
            shadows=16,shadow_steps=900,shadow_lr=.008)
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS,default='smoke')
    p.add_argument('--output',default='runs/e7');p.add_argument('--data-dir',default='./data')
    p.add_argument('--requests',nargs='+',type=int);p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--no-shadow',action='store_true');p.add_argument('--only-analyze',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    config=PRESETS[a.preset].copy()
    for field in ('requests','seeds'):
        val=getattr(a,field)
        if val is not None:config[field]=val
    config['flow_type']='coupling';config['request_master_seed']=2026;config['requests_total']=len(config['requests']);config['data_dir']=a.data_dir;config['output']=str(out);config['run_preset']=a.preset
    manifest=out/'manifest.json';manifest.write_text(json.dumps(config,indent=2))
    script=Path(__file__).with_name('experiment_e6.py')
    for request_index,request in enumerate(config['requests']):
        folder=out/f'request_{request}';folder.mkdir(exist_ok=True)
        resultfile=folder/'results.json'
        complete=False
        if resultfile.exists():
            try:
                res=json.loads(resultfile.read_text());complete={r['seed'] for r in res['rows'] if r['method']=='original'}==set(config['seeds'])
            except (ValueError,KeyError): pass
        if not complete and not a.only_analyze:
            cmd=[sys.executable,str(script),'--dataset',config['dataset'],'--data-dir',config['data_dir'],
                 '--device',config['device'],'--output',str(folder),'--selection-seed',str(request),'--request-index',str(request_index),
                 '--request-master-seed',str(config['request_master_seed']),
                 '--requests-total',str(config['requests_total']),
                 '--seeds',*map(str,config['seeds']),'--canary-n',str(config['canary_n']),
                 '--hidden',str(config['hidden']),'--train-steps',str(config['train_steps']),
                 '--short-steps',str(config['short_steps']),'--long-steps',str(config['long_steps']),
                 '--lr',str(config['lr']),'--unlearn-lr',str(config['unlearn_lr']),
                 '--canary-batch-frac',str(config['canary_batch_frac']),
                 '--n-retrains',str(config['n_retrains']),'--timeout',str(config['timeout']),
                 '--random-samples',str(config['random_samples']),
                 '--pgd-restarts',str(config['pgd_restarts']),'--pgd-steps',str(config['pgd_steps']),
                 '--flow-type',config['flow_type'],'--save-models']
            with (folder/'training.log').open('w') as log:
                print('Training',request,flush=True)
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        if not resultfile.exists():continue
        if not a.no_shadow:
            device='cuda' if config['device']=='auto' and torch.cuda.is_available() else ('cpu' if config['device']=='auto' else config['device'])
            auditargs=SimpleNamespace(**config,device_resolved=device,request_index=request_index)
            for seed in config['seeds']:
                outfile=folder/f'shadow_seed_{seed}.json'
                if outfile.exists():continue
                models={}; names=['original','fine_tune_short','fine_tune_long','reset_head','sham_continued']+[f'retrain_{j}' for j in range(config['n_retrains'])]
                for name in names:
                    path=folder/'checkpoints'/f'seed_{seed}_{name}.pt'
                    if not path.exists():
                        print('Missing checkpoint',path,'; skip shadow for seed',seed);models={};break
                    m=Net(3,config['hidden'])
                    # Only locally generated checkpoint state_dicts, not untrusted downloads.
                    m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));models[name]=m.eval()
                if models:
                    print('Shadow audit',request,seed,flush=True)
                    audit(auditargs,request,seed,models,outfile)
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_e7.py')),'--run-dir',str(out)],check=True)

if __name__=='__main__':main()
