#!/usr/bin/env python3
"""Restore the local bundle in isolation, verify it and rerun frozen cases."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dcl.degree.inventory import ROOT
from dcl.study.storage import atomic_json,file_hash


if __name__=='__main__':
    receipt=json.loads((ROOT/'artifact-index.json').read_text())
    archive=Path(receipt['path'])
    assert file_hash(archive)==receipt['sha256']
    tick=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='degree-clean-recovery-') as scratch:
        scratch=Path(scratch)
        shutil.copytree('src',scratch/'src',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree('docs',scratch/'docs')
        shutil.copytree('paper',scratch/'paper',ignore=shutil.ignore_patterns('*.aux','*.log','*.synctex*'))
        (scratch/'scripts').mkdir()
        for name in ('verify-degree-study.py','replay-degree-smoke.py'):
            shutil.copyfile(Path('scripts')/name,scratch/'scripts'/name)
        with tarfile.open(archive,'r:gz') as tf:
            for member in tf:
                destination=(scratch/member.name).resolve()
                if not member.isfile() or not destination.is_relative_to(scratch.resolve()):
                    raise ValueError('Runtime archive must contain regular files inside the recovery directory')
                tf.extract(member,scratch)
        restored=scratch/ROOT
        manifest=json.loads((restored/'runtime-manifest.json').read_text())
        for entry in manifest['files']:
            assert file_hash(scratch/entry['path'])==entry['sha256'],entry['path']
        env=dict(os.environ,PYTHONPATH='src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        records=[]
        for script,args in [('verify-degree-study.py',['--portable']),('replay-degree-smoke.py',[])]:
            p=subprocess.run([sys.executable,'scripts/'+script,*args],cwd=scratch,env=env,capture_output=True,text=True)
            records.append(dict(script=script,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
            assert p.returncode==0,records[-1]
        portable=json.loads((restored/'portable-verification.json').read_text())
        replay=json.loads((restored/'reproduction-smoke.json').read_text())
        result=dict(restored=True,archive_sha256=receipt['sha256'],verified_restored_files=len(manifest['files']),
            portable_verification=portable,reproduction_smoke=replay,commands=records,
            wall_seconds=time.perf_counter()-tick,external_original_worktrees_used=False,
            qualification='Current repository source/manuscript plus the local runtime bundle; external absolute source paths are intentionally skipped.')
    atomic_json(ROOT/'clean-recovery.json',result)
    print(json.dumps(dict(restored=True,replayed_rows=replay['replayed_rows'],identical_rows=replay['identical_rows'],wall_seconds=result['wall_seconds'])))
