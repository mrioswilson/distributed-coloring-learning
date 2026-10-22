#!/usr/bin/env python3
"""Persistent sequential GPU stages; raw state and budget are resumable."""
from pathlib import Path
import subprocess
import sys
from dcl.sizedegree.common import *

for phase,module,args in [('confirmation','run',['confirmation']),('mechanism','mechanism',[])]:
    result=subprocess.run([sys.executable,'-m','dcl.sizedegree.'+module,*args])
    if result.returncode:
        atomic_json(ROOT/'gpu-workflow.json',dict(status='failed',phase=phase,returncode=result.returncode,updated=now()))
        raise SystemExit(result.returncode)
queue=read(ROOT/'queue-confirmation.json')
complete=all(t['status']=='complete' for t in queue['tasks'])
atomic_json(ROOT/'gpu-workflow.json',dict(status='complete' if complete else 'budget_incomplete',updated=now(),budget=read(ROOT/'budget.json')))
print('GPU workflow finished; all confirmation banks complete='+str(complete),flush=True)
