#!/usr/bin/env python3
"""Build revised canonical TeX with native LaTeX and an actual compiler receipt."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/"paper/ucgnn"
OUT=ROOT/"results/paper-revision-20261030"


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    env=os.environ.copy();freeze=json.loads((OUT/"protocol-frozen.json").read_text())
    epoch=int(datetime.fromisoformat(freeze["created_utc"]).timestamp())
    env.update(SOURCE_DATE_EPOCH=str(epoch),FORCE_SOURCE_DATE="1",TZ="UTC")
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
              ['bibtex','main'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]
    logs=OUT/"build";logs.mkdir(exist_ok=True)
    for i,command in enumerate(commands):
        result=subprocess.run(command,cwd=PAPER,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (logs/f"{i+1:02d}-{command[0]}.txt").write_text(result.stdout)
        if result.returncode:raise RuntimeError(f"{command[0]} failed; see {logs}")
    log=(PAPER/"main.log").read_text();(OUT/"paper-build.log").write_text(log)
    unresolved=[line for line in log.splitlines() if 'undefined' in line.lower() or 'Citation `' in line]
    if unresolved:raise RuntimeError("Unresolved TeX citations/references: "+str(unresolved))
    import pypdfium2 as pdfium
    pdf=pdfium.PdfDocument(str(PAPER/"main.pdf"));texts=[]
    from PIL import Image
    renders=OUT/"pdf-review";renders.mkdir(exist_ok=True)
    thumbs=[]
    for i in range(len(pdf)):
        page=pdf[i];tp=page.get_textpage();texts.append(tp.get_text_range());tp.close()
        bitmap=page.render(scale=1.2);im=bitmap.to_pil();im.save(renders/f"page-{i+1:02d}.png")
        thumb=im.copy();thumb.thumbnail((255,350));thumbs.append(thumb.copy())
        bitmap.close();page.close()
    text="\n".join(texts);(OUT/"paper-text.txt").write_text(text)
    for offset in range(0,len(thumbs),12):
        chunk=thumbs[offset:offset+12];sheet=Image.new('RGB',(4*265,((len(chunk)+3)//4)*365),'#dedede')
        for j,thumb in enumerate(chunk):sheet.paste(thumb,((j%4)*265,(j//4)*365))
        sheet.save(renders/f"contact-{offset//12+1}.png")
    fonts=subprocess.run(['pdffonts',str(PAPER/'main.pdf')],text=True,capture_output=True,check=True).stdout
    (OUT/"pdf-fonts.txt").write_text(fonts)
    tools={}
    for name in ("pdflatex","bibtex"):
        executable=Path(shutil.which(name)).resolve()
        version=subprocess.run([name,'--version'],text=True,capture_output=True,check=True).stdout.splitlines()[0]
        tools[name]=dict(path=str(executable),sha256=sha(executable),version=version)
    receipt=dict(engine="native-pdflatex-bibtex",machine=platform.machine(),python=platform.python_version(),
        source_date_epoch=epoch,commands=commands,compiler=tools,pages=len(pdf),pdf_sha256=sha(PAPER/'main.pdf'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(PAPER.glob('*.tex'))},
        unresolved_references=unresolved,overfull_boxes=[l for l in log.splitlines() if 'Overfull' in l],
        visual_review="Renders produced; visual inspection recorded separately, not implied by compilation.",
        historical_engine="Old Linux x86-64 Tectonic receipt preserved; not executed on this ARM host.")
    (OUT/"paper-build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("engine","machine","pages","pdf_sha256","overfull_boxes")},indent=2))


if __name__=="__main__":main()
