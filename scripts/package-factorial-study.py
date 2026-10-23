#!/usr/bin/env python3
"""Self-contained local scientific delivery; never uploads or pushes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from importlib.metadata import version
from dcl.sizedegree.common import *

p=argparse.ArgumentParser();p.add_argument('--outputs',type=Path,required=True);p.add_argument('--work',type=Path,required=True);args=p.parse_args()
base=Path(__file__).resolve().parents[1];os.chdir(base);f=verify_frozen()
if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise SystemExit('Commit all reviewed source/results before packaging')
audit=read(OUT/'audit.json');paper=read(OUT/'manuscript-audit.json');figures=read(OUT/'figure-visual-review.json')
assert audit['passed'] and paper['passed'] and paper['visual_review_passed'] and figures['passed']
for name,digest in paper['source_hashes'].items():assert file_hash(name)==digest,name
assert file_hash('paper/ucgnn/main.pdf')==paper['pdf_sha256']
for name,digest in figures['hashes'].items():assert file_hash(name)==digest,name
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
args.outputs.mkdir(parents=True,exist_ok=True);args.work.mkdir(parents=True,exist_ok=True)
manifest=dict(source_commit=commit,protocol_commit=read(OUT/'protocol-commit.json')['commit'],created=now(),
              inputs=[],versions={name:version(name) for name in ('numpy','scipy','torch','networkx','matplotlib','pandas')},
              audit=audit,manuscript=paper,figures=figures)
archive=args.outputs/'ucgnn-size-degree-study.tar.gz'
temporary=archive.with_suffix(archive.suffix+'.writing')
with tempfile.TemporaryDirectory(prefix='factorial-delivery-',dir=args.work) as scratch:
    scratch=Path(scratch);source=scratch/'source.tar';bundle=scratch/'source-history.bundle'
    subprocess.run(['git','archive','--format=tar','-o',str(source),commit],check=True)
    subprocess.run(['git','bundle','create',str(bundle),'refs/heads/research/ucgnn-size-degree'],check=True,capture_output=True)
    def add_file(tar,path,name,kind):
        path=Path(path);digest=file_hash(path)
        manifest['inputs'].append(dict(path=name,sha256=digest,bytes=path.stat().st_size,kind=kind))
        tar.add(path,arcname=name,recursive=False)
    with tarfile.open(temporary,'w:gz',compresslevel=4) as tar:
        with tarfile.open(source) as src:
            for item in src:
                stream=src.extractfile(item) if item.isfile() else None
                item.name='ucgnn-size-degree-study/'+item.name
                tar.addfile(item,stream)
        add_file(tar,bundle,'source-history.bundle','git-history')
        for path in sorted(ROOT.rglob('*')):
            if path.is_file():add_file(tar,path,'ucgnn-size-degree-study/'+str(path),'new-runtime')
        for root,name,sub in [(BASE,'baseline-robust','ucgnn-robust-distribution'),(GLOBAL,'baseline-global','ucgnn-global-distribution')]:
            for path in sorted((root/'data'/sub/'banks').glob('*.npz')):
                add_file(tar,path,name+'/'+str(path.relative_to(root)),'historical-algorithm-bank')
        reference=[(BASE,'baseline-robust','results/ucgnn-robust-distribution/'+tail) for tail in
                   ('freeze.json','explanatory-witnesses.json','sampling-manifest.csv','confirmation-tests/path3__normalized-s41000__m500.json',
                    'confirmation-tests/regular-n512-d100__normalized-s41000__m500.json')]
        reference += [(GLOBAL,'baseline-global','results/ucgnn-global-distribution/graphs.json')]
        reference += [(GLOBAL,'baseline-global','results/ucgnn-global-distribution/sampling-manifest.json')]
        for root,name,path in reference:add_file(tar,root/path,name+'/'+path,'historical-reference')
        for spec in specs():
            assert file_hash(spec['checkpoint'])==spec['sha256']
            add_file(tar,spec['checkpoint'],'checkpoints/'+spec['label']+'.pt','checkpoint')
        add_file(tar,Path('/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic'),'tools/tectonic','document-compiler')
        logs=Path('/home/maf/Documents/Codex/2026-10-21/files-pasted-by-the-user-codex/work')
        for path in sorted(logs.glob('factorial-*.log')):add_file(tar,path,'run-logs/'+path.name,'log')
        mf=scratch/'DELIVERY_MANIFEST.json';atomic_json(mf,manifest);tar.add(mf,arcname='DELIVERY_MANIFEST.json')
    # Re-open the actual compressed delivery and verify every archived raw input,
    # checkpoint and compiler byte stream, not just source file names.
    expected={r['path']:r['sha256'] for r in manifest['inputs']}
    seen=set()
    with tarfile.open(temporary,'r:gz') as tar:
        for item in tar:
            if item.name in expected:
                handle=tar.extractfile(item);assert handle is not None
                digest=hashlib.file_digest(handle,'sha256').hexdigest()
                assert digest==expected[item.name],item.name;seen.add(item.name)
    assert seen==set(expected)
os.replace(temporary,archive)
manifest.update(archive=str(archive),archive_sha256=file_hash(archive),archive_bytes=archive.stat().st_size,
                archived_input_hashes_verified=len(seen))
atomic_json(args.outputs/'FACTORIAL_DELIVERY_MANIFEST.json',manifest)
light=args.outputs/'factorial-size-degree-study';light.mkdir(exist_ok=True)
for directory in ('docs','src/dcl/sizedegree','tests'):
    target=light/directory;target.mkdir(parents=True,exist_ok=True)
    paths=(base/directory).glob('FACTORIAL*.md') if directory=='docs' else (base/directory).glob('*')
    for path in paths:
        if path.is_file() and (directory!='tests' or path.name.startswith('test_sizedegree')):shutil.copy2(path,target/path.name)
shutil.copytree(OUT,light/'results',dirs_exist_ok=True)
for name in ('FACTORIAL_STUDY.md',):shutil.copy2(base/name,light/name)
shutil.copy2(OUT/'FINAL_REPORT.md',args.outputs/'INFORME_FACTORIAL_TAMANO_GRADO.md')
shutil.copy2(OUT/'RESUMEN_EJECUTIVO.md',args.outputs/'RESUMEN_FACTORIAL.md')
shutil.copy2(base/'paper/ucgnn/main.pdf',args.outputs/'manuscrito-factorial.pdf')
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,verified_raw_inputs=len(seen),source_commit=commit)),flush=True)
