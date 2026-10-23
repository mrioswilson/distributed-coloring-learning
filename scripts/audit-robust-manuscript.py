"""Render changed manuscript pages; require explicit final visual review."""
from pathlib import Path
import argparse,json,hashlib,re,sys,subprocess
parser=argparse.ArgumentParser();parser.add_argument('--pdfium-root',type=Path);parser.add_argument('--render-dir',type=Path,required=True);args=parser.parse_args()
if args.pdfium_root:sys.path.insert(0,str(args.pdfium_root))
import pypdfium2 as pdfium
base=Path(__file__).resolve().parents[1];paper=base/'paper/ucgnn';out=base/'results/ucgnn-robust-distribution'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline='1ba4546c64d143c44480644ebb9d61f9917c8982'
historical=[]
for name in subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'paper/ucgnn'],cwd=base,text=True).splitlines():
    p=Path(name)
    if p.name in ('main.tex','main.pdf') or p.suffix=='.bib':continue
    original=subprocess.check_output(['git','show',baseline+':'+name],cwd=base)
    assert (base/p).read_bytes()==original,'Historical manuscript asset changed: '+name
    historical.append(name)
rows=json.loads((out/'primary.json').read_text());count=sum(r['reject'] for r in rows)
doc=pdfium.PdfDocument(str(paper/'main.pdf'));texts=[p.get_textpage().get_text_range() for p in doc]
compact=lambda t:re.sub(r'\s+','',t).replace('-','')
assert f'Therobustprimaryfamilyrejected{count}/795' in compact('\n'.join(texts))
starts=[i for i,t in enumerate(texts) if 'RobustTestingofCompleteColoringLaws' in compact(t)];assert starts
log=(paper/'main.log').read_text(errors='replace')
undefined=[line for line in log.splitlines() if 'undefined' in line.lower() and ('reference' in line.lower() or 'citation' in line.lower())];assert not undefined
args.render_dir.mkdir(parents=True,exist_ok=True);renders=[]
pages=sorted({0,*range(starts[0],min(len(doc),starts[0]+6))})
for i in pages:
    f=args.render_dir/f'robust-manuscript-page-{i+1}.png';doc[i].render(scale=1.5).to_pil().save(f)
    renders.append(dict(page=i+1,file=str(f),sha256=sha(f)))
sources=sorted({*paper.rglob('*.tex'),*paper.rglob('*.bib'),*paper.glob('figures/*.pdf'),*paper.glob('figures/*.png')})
result=dict(passed=False,text_checks_passed=True,visual_review_passed=False,page_count=len(doc),new_section_pages=[i+1 for i in starts],primary_rejections=count,
            historical_assets_preserved=historical,
            pdf_sha256=sha(paper/'main.pdf'),undefined_references=undefined,overfull_boxes=[line for line in log.splitlines() if 'Overfull' in line],
            rendered=renders,source_hashes={str(p.relative_to(base)):sha(p) for p in sources},note='Set passed only after inspecting final tables, equations, figure captions, abstract and all new scientific figures.')
(out/'manuscript-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(text_checks_passed=True,rendered=renders,page_count=len(doc)),indent=2))
