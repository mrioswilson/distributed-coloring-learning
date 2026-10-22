"""Create user-facing deliverables only after the complete audit and PDF check."""
from pathlib import Path
import json,subprocess,tarfile,gzip,shutil,hashlib
base=Path(__file__).resolve().parents[1]
results=base/'results/ucgnn-global-distribution'
data=base/'data/ucgnn-global-distribution'
out=Path('/home/maf/Documents/Codex/2026-10-21/files-pasted-by-the-user-codex/outputs')
work=Path('/home/maf/Documents/Codex/2026-10-21/files-pasted-by-the-user-codex/work')
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert read(results/'audit.json')['passed']
assert read(results/'manuscript-audit.json')['passed']
assert read(results/'manuscript-audit.json')['pdf_sha256']==sha(base/'paper/ucgnn/main.pdf')
assert all(t['status']=='complete' for t in read(data/'queue.json')['tasks'])
assert not subprocess.check_output(['git','status','--porcelain'],cwd=base,text=True).strip(),'Commit reviewed results before packaging'
out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
delivery=out/'global-coloring-study'
shutil.copytree(results,delivery,dirs_exist_ok=True)
shutil.copytree(base/'docs',delivery/'docs',dirs_exist_ok=True)
shutil.copy2(base/'GLOBAL_DISTRIBUTIONS.md',delivery/'REPRODUCIR.md')
shutil.copy2(results/'FINAL_REPORT.md',out/'INFORME_COLORACIONES_GLOBALES.md')
shutil.copy2(results/'RESUMEN_EJECUTIVO.md',out/'RESUMEN_EJECUTIVO.md')
shutil.copy2(base/'paper/ucgnn/main.pdf',out/'manuscrito-coloraciones-globales.pdf')
source_tar=work/'global-coloring-source.tar'
with source_tar.open('wb') as f:subprocess.run(['git','archive','--format=tar','HEAD'],cwd=base,stdout=f,check=True)
with tarfile.open(source_tar,'a') as tar:
    tar.add(data,arcname='data/ucgnn-global-distribution')
    ignored=subprocess.check_output(['git','ls-files','--others','--ignored','--exclude-standard','results/ucgnn-global-distribution'],cwd=base,text=True).splitlines()
    for name in ignored:tar.add(base/name,arcname=name)
archive=out/'ucgnn-global-distribution-study.tar.gz'
with source_tar.open('rb') as f,gzip.open(archive,'wb',compresslevel=6) as g:shutil.copyfileobj(f,g)
expected={t['file']:t['sha256'] for t in read(data/'queue.json')['tasks']}
verified=set()
with tarfile.open(archive,'r:gz') as tar:
    for member in tar:
        if member.name in expected:
            with tar.extractfile(member) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            assert actual==expected[member.name],member.name
            verified.add(member.name)
assert verified==set(expected)
manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip(),verified_raw_bank_archives=len(verified),
              archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive)),
              paper=dict(file='manuscrito-coloraciones-globales.pdf',sha256=sha(out/'manuscrito-coloraciones-globales.pdf')),
              audit=read(results/'audit.json'),checkpoint_policy='Frozen checkpoint weights remain at the original local paths; exact hashes and reproduction instructions are included. No new training or push.')
(out/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(output=str(out),archive_bytes=archive.stat().st_size,verified_banks=len(verified),source_commit=manifest['source_commit']),indent=2))
