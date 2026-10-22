"""Local delivery: full raw banks, immutable checkpoints, source and history."""
from pathlib import Path
import json,subprocess,hashlib,shutil,tarfile,gzip,os
base=Path(__file__).resolve().parents[1];os.chdir(base)
import sys
sys.path.insert(0,str(base/'src'))
from dcl.robustdist.common import OUT,DATA,OLD,read,specs,file_hash
root=Path('/home/maf/Documents/Codex/2026-10-21/files-pasted-by-the-user-codex')
out=root/'outputs';work=root/'work';delivery=out/'robust-coloring-study'
assert read(OUT/'audit.json')['passed'] and read(OUT/'manuscript-audit.json')['passed']
figure_review=read(OUT/'figure-visual-review.json');assert figure_review['passed']
for name,digest in figure_review['figures'].items():assert file_hash(name)==digest,name
review=read(OUT/'manuscript-audit.json');assert review['pdf_sha256']==file_hash(base/'paper/ucgnn/main.pdf')
for name,digest in review['source_hashes'].items():assert file_hash(base/name)==digest,name
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit reviewed results first'
out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
shutil.copytree(OUT,delivery,dirs_exist_ok=True);shutil.copytree(base/'docs',delivery/'docs',dirs_exist_ok=True)
shutil.copytree(base/'src/dcl/robustdist',delivery/'code/dcl/robustdist',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(base/'ROBUST_DISTRIBUTIONS.md',delivery/'REPRODUCIR.md')
shutil.copy2(OUT/'FINAL_REPORT.md',out/'ROBUST_DISTRIBUTION_REPORT.md')
shutil.copy2(OUT/'RESUMEN_EJECUTIVO.md',out/'RESUMEN_PRUEBAS_ROBUSTAS.md')
shutil.copy2(base/'paper/ucgnn/main.pdf',out/'manuscrito-pruebas-robustas.pdf')
bundle=work/'robust-source-history.bundle'
subprocess.run(['git','bundle','create',str(bundle),'research/ucgnn-robust-distribution-tests'],check=True)
source=work/'robust-delivery-source.tar'
with source.open('wb') as f:subprocess.run(['git','archive','--format=tar','HEAD'],stdout=f,check=True)
expected={};checkpoint_manifest=[]
with tarfile.open(source,'a') as tar:
    tar.add(DATA,arcname=str(DATA));tar.add(OLD/'data/ucgnn-global-distribution',arcname='data/ucgnn-global-distribution')
    tar.add(bundle,arcname='source-history.bundle')
    for s in specs():
        assert file_hash(s['checkpoint'])==s['sha256'];name='frozen-checkpoints/'+s['label']+'.pt'
        tar.add(s['checkpoint'],arcname=name);expected[name]=s['sha256'];checkpoint_manifest.append(dict(label=s['label'],file=name,sha256=s['sha256']))
    compiler=Path('/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic')
    if compiler.exists():tar.add(compiler,arcname='.tools/tectonic')
    ignored=subprocess.check_output(['git','ls-files','--others','--ignored','--exclude-standard',str(OUT)],text=True).splitlines()
    for name in ignored:tar.add(base/name,arcname=name)
archive=out/'ucgnn-robust-distribution-tests.tar.gz'
with source.open('rb') as f,gzip.open(archive,'wb',compresslevel=6) as g:shutil.copyfileobj(f,g)
for q in DATA.glob('queue-*.json'):
    for t in read(q)['tasks']:expected[t['file']]=t['sha256']
for t in read(OLD/'data/ucgnn-global-distribution/queue.json')['tasks']:expected[t['file']]=t['sha256']
for f in (OUT/'benchmark-trials').glob('*.json'):
    r=read(f);expected[r['raw_file']]=r['raw_sha256']
verified=set()
with tarfile.open(archive,'r:gz') as tar:
    for member in tar:
        if member.name in expected:
            with tar.extractfile(member) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            assert actual==expected[member.name],member.name;verified.add(member.name)
assert verified==set(expected)
manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_commit=read(OUT/'protocol-commit.json')['commit'],
      archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=file_hash(archive)),paper=dict(file='manuscrito-pruebas-robustas.pdf',sha256=file_hash(out/'manuscrito-pruebas-robustas.pdf')),
      verified_archived_inputs=len(verified),checkpoints=checkpoint_manifest,audit=read(OUT/'audit.json'),manuscript_audit=review,figure_review=figure_review,
      note='All source, Git history, original baseline banks, new raw banks, benchmark arrays and fifteen immutable checkpoints are included. No neural retraining or remote push.')
(out/'ROBUST_DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(output=str(out),archive_bytes=archive.stat().st_size,verified_inputs=len(verified),checkpoint_files=len(checkpoint_manifest),source_commit=manifest['source_commit']),indent=2))
