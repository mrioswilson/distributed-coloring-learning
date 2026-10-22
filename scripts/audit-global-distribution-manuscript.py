"""Text/source checks and page renders; visual review must be confirmed separately."""
from pathlib import Path
import argparse,json,re,sys,hashlib
parser=argparse.ArgumentParser()
parser.add_argument('--pdfium-root',type=Path)
parser.add_argument('--render-dir',type=Path,required=True)
args=parser.parse_args()
if args.pdfium_root:sys.path.insert(0,str(args.pdfium_root))
import pypdfium2 as pdfium
base=Path(__file__).resolve().parents[1];results=base/'results/ucgnn-global-distribution';paper=base/'paper/ucgnn'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
primary=json.loads((results/'primary.json').read_text());count=sum(r['reject_holm'] for r in primary)
doc=pdfium.PdfDocument(str(paper/'main.pdf'))
texts=[page.get_textpage().get_text_range() for page in doc]
compact=lambda t:re.sub(r'\s+','',t).replace('-','')
alltext=compact('\n'.join(texts))
assert f'Theprimaryfamilyrejected{count}of795' in alltext,'PDF lacks the final result count'
starts=[i for i,t in enumerate(texts) if 'AreLearnedDistributedColoringAlgorithms' in compact(t)]
assert starts,'PDF lacks the new section'
log=(paper/'main.log').read_text(errors='replace')
undefined=[line for line in log.splitlines() if 'undefined' in line.lower() and ('citation' in line.lower() or 'reference' in line.lower())]
assert not undefined,undefined
args.render_dir.mkdir(parents=True,exist_ok=True)
rendered=[]
for i in range(starts[0],min(len(doc),starts[0]+4)):
    file=args.render_dir/f'global-section-page-{i+1}.png'
    doc[i].render(scale=1.5).to_pil().save(file)
    rendered.append(dict(page=i+1,file=str(file),sha256=sha(file)))
inputs=sorted({*paper.rglob('*.tex'),*paper.rglob('*.bib'),*paper.glob('figures/*.pdf'),*paper.glob('figures/*.png')})
result=dict(passed=False,text_checks_passed=True,visual_review_passed=False,pdf_sha256=sha(paper/'main.pdf'),page_count=len(doc),
            global_section_start_pages=[i+1 for i in starts],rendered=rendered,primary_rejections=count,
            undefined_references=undefined,source_hashes={str(p.relative_to(base)):sha(p) for p in inputs},
            note='Set passed only after inspecting the rendered equations, table, figure and section layout.')
(results/'manuscript-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(text_checks_passed=True,rendered=rendered,page_count=len(doc)),indent=2))
