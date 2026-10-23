#!/usr/bin/env python3
"""Render the final new contribution and verify preserved earlier assets."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import hashlib
import unicodedata

p=argparse.ArgumentParser();p.add_argument('--pdfium-root',type=Path);p.add_argument('--render-dir',type=Path,required=True);args=p.parse_args()
if args.pdfium_root:sys.path.insert(0,str(args.pdfium_root))
import pypdfium2 as pdfium
base=Path(__file__).resolve().parents[1];paper=base/'paper/ucgnn';out=base/'results/ucgnn-size-degree'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline='74134d865db4f142bfeea49b091200588d1f2803'
preserved=[]
for name in subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'paper/ucgnn'],cwd=base,text=True).splitlines():
    path=Path(name)
    if path.name in ('main.tex','main.pdf'):continue
    original=subprocess.check_output(['git','show',baseline+':'+name],cwd=base)
    assert (base/path).read_bytes()==original,'Earlier manuscript asset changed: '+name
    preserved.append(name)
original=subprocess.check_output(['git','show',baseline+':paper/ucgnn/main.tex'],cwd=base,text=True)
source=(paper/'main.tex').read_text()
restored=source.replace('\n\\input{factorial-size-degree}','').replace('\n\\IfFileExists{tables/factorial-abstract.tex}{\\input{tables/factorial-abstract.tex}}{}','')
assert restored==original,'Changes to main.tex exceed the two intended new inputs'
doc=pdfium.PdfDocument(str(paper/'main.pdf'));texts=[page.get_textpage().get_text_range() for page in doc]
# PDFium represents some discretionary line-break hyphens as U+FFFE.
# Normalize extracted text before matching the complete section heading.
compact=lambda text:re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).replace('-','').replace('\ufffe','').replace('\u00ad','')
starts=[i for i,t in enumerate(texts) if 'HowGraphSizeandDegreeAffectDistributionalGeneralization' in compact(t)]
assert starts
alltext=compact('\n'.join(texts))
for expected in ('720','120','240','WhatWeCanProve,WhatWeObserve,andWhatRemainsOpen'):
    assert expected in alltext,expected
log=(paper/'main.log').read_text(errors='replace')
undefined=[line for line in log.splitlines() if 'undefined' in line.lower() and ('reference' in line.lower() or 'citation' in line.lower())]
assert not undefined
args.render_dir.mkdir(parents=True,exist_ok=True);renders=[]
# New figures may float after the section; inspect the entire tail through
# the references, plus the changed abstract.
for i in sorted({0,*range(starts[0],len(doc))}):
    target=args.render_dir/f'factorial-manuscript-page-{i+1}.png'
    doc[i].render(scale=1.5).to_pil().save(target)
    renders.append(dict(page=i+1,file=str(target),sha256=sha(target)))
sources=sorted({*paper.rglob('*.tex'),*paper.rglob('*.bib'),*paper.glob('figures/*.pdf'),*paper.glob('figures/*.png')})
result=dict(passed=False,text_checks_passed=True,visual_review_passed=False,
            pdf_sha256=sha(paper/'main.pdf'),page_count=len(doc),factorial_section_pages=[i+1 for i in starts],
            earlier_assets_preserved=preserved,main_tex_restores_exactly=True,undefined_references=undefined,
            overfull_boxes=[line for line in log.splitlines() if 'Overfull' in line],renders=renders,
            source_hashes={str(p.relative_to(base)):sha(p) for p in sources},
            note='Mark passed only after reviewing all changed text/tables and all scientific figures; retain hashes.')
(out/'manuscript-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(text_checks_passed=True,page_count=len(doc),renders=len(renders)),indent=2))
