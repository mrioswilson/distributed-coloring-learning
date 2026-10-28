#!/usr/bin/env python3
"""Build the complete draft; preserve and hash the earlier manuscript."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pypandoc
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from dcl.integration.report import run as rebuild_report
from dcl.study.storage import file_hash

if (ROOT / "paper/ucgnn/main.tex").read_text().startswith("% Authoritative closure source"):
    raise SystemExit("Historical generator: use scripts/build-closure-paper.py here, or reproduce the historical source in a separate checkout.")

p = argparse.ArgumentParser()
p.add_argument("--compiler", type=Path, default=ROOT / ".tools/tectonic")
p.add_argument("--skip-report", action="store_true")
args = p.parse_args()
if not args.skip_report:
    rebuild_report()
paper = ROOT / "paper/ucgnn"
for extension in ("tex", "pdf"):
    old = paper / f"pre-integration-main.{extension}"
    if not old.exists():
        shutil.copyfile(paper / f"main.{extension}", old)
source = (ROOT / "paper/PAPER_DRAFT.md").read_text()
title, text = source.split("\n", 1)
text = re.sub(r"<!-- BEGIN_MARKDOWN_REFERENCES -->[\s\S]*?<!-- END_MARKDOWN_REFERENCES -->", "", text)
text = text.replace("ucgnn/figures/", "figures/")
text = re.sub(r"(figures/(?:integration|sample)-[a-z-]+)\.png", r"\1.pdf", text)
protected = re.compile(r"(`[^`]*`|\$\$[\s\S]*?\$\$|\$[^$\n]*\$|https?://[^\s)]+)")
parts = protected.split(text)
for i in range(0, len(parts), 2):
    for glyph, math in [("≤", r"$\le$ "), ("≥", r"$\ge$ "), ("Δ", r"$\Delta$ "),
                        ("ρ", r"$\rho$ "), ("∈", r"$\in$ "), ("²", r"$^2$ ")]:
        parts[i] = parts[i].replace(glyph, math)
text = "".join(parts)
text = text.replace(r"\setminus", r"\smallsetminus")
keys = set(re.findall(r"@([a-z][a-z0-9]+)", text))
bib = (paper / "references.bib").read_text() + (paper / "integration-references.bib").read_text()
available = set(re.findall(r"@\w+\{([^,]+),", bib))
if not keys <= available:
    raise RuntimeError(f"Unknown bibliography keys: {keys-available}")
command = [pypandoc.get_pandoc_path(), "--from=markdown", "--to=latex", "--standalone", "--natbib",
           "--number-sections", "--bibliography=references.bib", "--bibliography=integration-references.bib",
           "--metadata=title:" + title.lstrip("# "), "--metadata=author:Research collaboration (author order pending)",
           "--metadata=date:23 October 2026", "--variable=geometry:margin=1in", "--variable=fontsize:11pt",
           "--variable=colorlinks:true", "--variable=biblio-style:unsrtnat"]
converted = subprocess.run(command, input=text, text=True, capture_output=True, cwd=paper, check=True)
if "not found" in converted.stderr.lower():
    raise RuntimeError(converted.stderr)
tex = "% Generated from paper/PAPER_DRAFT.md by scripts/build-integrated-paper.py\n" + converted.stdout
tex = tex.replace(r"\begin{document}", "\\setlength{\\emergencystretch}{3em}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{document}")
(paper / "main.tex").write_text(tex)
compiler = args.compiler.resolve()
if not compiler.exists():
    raise SystemExit("Tectonic missing; follow the pinned instructions in paper/ucgnn/README.md")
result = subprocess.run([str(compiler), "-X", "compile", "--keep-logs", "main.tex"], cwd=paper, capture_output=True, text=True)
out = ROOT / "results/integration"
(out / "paper-build.txt").write_text(result.stdout + result.stderr)
if result.returncode:
    raise RuntimeError("Compilation failed; see results/integration/paper-build.txt")
doc = pdfium.PdfDocument(str(paper / "main.pdf"))
pages = [page.get_textpage().get_text_range() for page in doc]
log = (paper / "main.log").read_text(errors="replace")
undefined = [line for line in log.splitlines() if "undefined" in line.lower() and any(w in line.lower() for w in ("reference", "citation"))]
if undefined:
    raise RuntimeError(f"Unresolved references/citations: {undefined}")
renders = out / "paper-review"
renders.mkdir(exist_ok=True)
selected = {0, 1, len(doc)-1, len(doc)-2}
for i, page in enumerate(pages):
    if re.search(r"Figure\s+[1-4]:|Table\s+[1-7]\.", page): selected.add(i)
    if re.search(r"How Much Supervision|supervision curves|Extended.interface supervision",page,re.IGNORECASE): selected.add(i)
rendered = []
for i in sorted(selected):
    path = renders / f"page-{i+1:02d}.png"
    doc[i].render(scale=1.25).to_pil().save(path)
    rendered.append(dict(page=i+1, file=str(path.relative_to(ROOT)), sha256=file_hash(path)))
record = dict(compile_passed=True, visual_review_passed=False, pages=len(doc), citation_keys=sorted(keys),
              pdf_sha256=file_hash(paper/"main.pdf"), latex_sha256=file_hash(paper/"main.tex"),
              markdown_sha256=file_hash(ROOT/"paper/PAPER_DRAFT.md"), compiler_sha256=file_hash(compiler),
              pandoc_version=str(pypandoc.get_pandoc_version()), undefined_references=undefined,
              overfull_boxes=[line for line in log.splitlines() if "Overfull" in line],
              missing_characters=[line for line in log.splitlines() if "Missing character" in line],
              prior_manuscript={ext:file_hash(paper/f"pre-integration-main.{ext}") for ext in ("tex","pdf")},
              renders=rendered, note="Record visual review after inspecting rendered pages and figures.")
(out / "paper-build.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({k:record[k] for k in ("compile_passed","pages","undefined_references","overfull_boxes","missing_characters")}),flush=True)
