#!/usr/bin/env python3
"""Relocate paths for an archived study, keeping frozen source/data unchanged.

Run from the packaged worktree. Only path bindings and checkpoint locations
are replaced in memory; SHA256 guards still verify the original files and
the exact weight bytes. Default recomputation uses CPU and no new sampling.
"""
from pathlib import Path
import argparse
import runpy
import sys
import os
import subprocess

source=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(source/'src'))
os.chdir(source)
parser=argparse.ArgumentParser()
parser.add_argument('module',choices=['analysis','validation','precision','capacity','exploratory','audit','summaries','figures','report','envelope','intervention'])
parser.add_argument('--package-root',type=Path,default=source.parent)
parser.add_argument('--restore-git-metadata',action='store_true')
args,rest=parser.parse_known_args()

if args.restore_git_metadata:
    inside=subprocess.run(['git','rev-parse','--is-inside-work-tree'],capture_output=True,text=True)
    if inside.returncode==0:
        raise SystemExit('Existing Git repository detected; no metadata changes made')
    bundle=args.package_root/'source-history.bundle'
    if not bundle.exists():raise SystemExit('Archive source-history.bundle is missing')
    subprocess.run(['git','init','-q'],check=True)
    subprocess.run(['git','fetch','-q',str(bundle),'refs/heads/research/ucgnn-size-degree'],check=True)
    subprocess.run(['git','reset','-q','--mixed','FETCH_HEAD'],check=True)

import dcl.sizedegree.common as c
if (args.package_root/'baseline-robust').exists():c.BASE=args.package_root/'baseline-robust'
if (args.package_root/'baseline-global').exists():c.GLOBAL=args.package_root/'baseline-global'
import dcl.robustdist.common as old
old.OLD=c.GLOBAL
weights=args.package_root/'checkpoints'
if weights.exists():
    def relocated_specs():
        rows=c.read(c.OUT/'checkpoints.json')
        for row in rows:
            path=weights/(row['label']+'.pt')
            if c.file_hash(path)!=row['sha256']:raise RuntimeError('Checkpoint hash mismatch')
            row['checkpoint']=str(path)
        return rows
    c.specs=relocated_specs
c.verify_frozen()
sys.argv=['dcl.sizedegree.'+args.module,*rest]
runpy.run_module('dcl.sizedegree.'+args.module,run_name='__main__')
