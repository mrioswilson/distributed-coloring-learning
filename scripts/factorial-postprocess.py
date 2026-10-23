#!/usr/bin/env python3
"""Persistent CPU completion pipeline; final visual review remains explicit."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
import fcntl
from dcl.sizedegree.common import *

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);args=p.parse_args()
base=Path(__file__).resolve().parents[1];os.chdir(base);args.work.mkdir(parents=True,exist_ok=True)
statepath=ROOT/'postprocess-workflow.json'
lock=(ROOT/'postprocess.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    marker=ROOT/'gpu-workflow.json'
    if marker.exists():
        status=read(marker)['status']
        if status!='complete':
            atomic_json(statepath,dict(status='sampling_incomplete',gpu_status=status,updated=now()))
            raise SystemExit('Sampling did not complete; no complete-cohort report is fabricated')
        factorial=OUT/'factorial-contrasts.json'
        intervention=OUT/'candidate-intervention/completion.json'
        if factorial.exists() and len(read(factorial))==verify_frozen()['factorial_contrasts'] and intervention.exists():break
    time.sleep(20)

def run(phase,command,cwd=base):
    atomic_json(statepath,dict(status='running',phase=phase,updated=now()))
    with (args.work/f'factorial-{phase}.log').open('w') as log:
        result=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        atomic_json(statepath,dict(status='failed',phase=phase,returncode=result.returncode,updated=now()))
        raise SystemExit(f'Postprocessing failed at {phase}; see its log')
    print(phase+' complete',flush=True)

run('intervention-analysis',[sys.executable,'-m','dcl.sizedegree.intervention','analyze'])
run('intervention-witnesses',[sys.executable,'-m','dcl.sizedegree.intervention_witnesses','analyze'])
for module in ('precision','capacity','exploratory','audit','summaries','cdf_witnesses','figures','report'):
    run(module,[sys.executable,'-m','dcl.sizedegree.'+module])
compiler='/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic'
run('tex',[compiler,'--keep-logs','--keep-intermediates','main.tex'],base/'paper/ucgnn')
run('manuscript-audit',[sys.executable,'scripts/audit-factorial-manuscript.py',
                       '--pdfium-root',str(args.work/'pdf-tools'),'--render-dir',str(args.work/'factorial-pdf-review')])
atomic_json(statepath,dict(status='ready_for_visual_review',updated=now()))
print('All automatic stages complete; rendered manuscript and figures are ready for final visual review',flush=True)
