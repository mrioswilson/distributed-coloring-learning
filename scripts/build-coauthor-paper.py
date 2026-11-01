#!/usr/bin/env python3
"""Compile the integrated article without rewriting either historical receipt."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/ucgnn"
OUT = ROOT / "results/integrated-paper-20261031"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    freeze = json.loads((OUT / "functional-freeze.json").read_text())
    epoch = int(datetime.fromisoformat(freeze["created_utc"]).timestamp())
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch), FORCE_SOURCE_DATE="1", TZ="UTC")
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-recorder", "main.tex"]
    commands = [latex, ["bibtex", "main"], latex, latex]
    logs = OUT / "build"
    logs.mkdir(exist_ok=True)
    for i, command in enumerate(commands):
        result = subprocess.run(command, cwd=PAPER, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (logs / f"{i+1:02d}-{command[0]}.txt").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{command[0]} failed; see {logs}")
    log = (PAPER / "main.log").read_text()
    (OUT / "paper-build.log").write_text(log)
    unresolved = [l for l in log.splitlines() if "undefined" in l.lower() or "Citation `" in l]
    if unresolved:
        raise RuntimeError(f"Unresolved citations/references: {unresolved}")
    import pypdfium2 as pdfium
    from PIL import Image
    pdf = pdfium.PdfDocument(str(PAPER / "main.pdf"))
    text, thumbs = [], []
    renders = OUT / "pdf-review"
    renders.mkdir(exist_ok=True)
    for i in range(len(pdf)):
        page = pdf[i]
        tp = page.get_textpage()
        text.append(tp.get_text_range())
        tp.close()
        bitmap = page.render(scale=1.2)
        im = bitmap.to_pil()
        im.save(renders / f"page-{i+1:02d}.png")
        thumb = im.copy()
        thumb.thumbnail((255,350))
        thumbs.append(thumb.copy())
        bitmap.close()
        page.close()
    pages = len(pdf)
    pdf.close()
    (OUT / "paper-text.txt").write_text("\n".join(text))
    for offset in range(0, pages, 12):
        chunk = thumbs[offset:offset+12]
        sheet = Image.new("RGB", (1060, ((len(chunk)+3)//4)*365), "#dedede")
        for j, thumb in enumerate(chunk):
            sheet.paste(thumb, ((j%4)*265, (j//4)*365))
        sheet.save(renders / f"contact-{offset//12+1}.png")
    fonts = subprocess.run(["pdffonts", str(PAPER/"main.pdf")], text=True, capture_output=True, check=True).stdout
    (OUT / "pdf-fonts.txt").write_text(fonts)
    compilers = {}
    for name in ("pdflatex", "bibtex"):
        binary = Path(shutil.which(name)).resolve()
        version = subprocess.run([name,"--version"],text=True,capture_output=True,check=True).stdout.splitlines()[0]
        compilers[name] = dict(path=str(binary),sha256=sha(binary),version=version)
    actual_inputs = {}
    for line in (PAPER/"main.fls").read_text().splitlines():
        if line.startswith("INPUT "):
            path = Path(line[6:])
            path = path.resolve() if path.is_absolute() else (PAPER/path).resolve()
            if path.is_relative_to(PAPER) and path.suffix in (".tex", ".pdf", ".png", ".sty", ".cls") and path != PAPER/"main.pdf":
                actual_inputs[str(path.relative_to(ROOT))] = sha(path)
    for path in PAPER.glob("*.bib"):
        actual_inputs[str(path.relative_to(ROOT))] = sha(path)
    receipt = dict(engine="pdflatex-bibtex",machine=platform.machine(),python=platform.python_version(),
        source_date_epoch=epoch,commands=commands,compiler=compilers,pages=pages,pdf_sha256=sha(PAPER/"main.pdf"),
        actual_inputs=actual_inputs,builder_sha256=sha(__file__),unresolved_references=unresolved,
        overfull_boxes=[l for l in log.splitlines() if "Overfull" in l],
        visual_review="Renders generated; inspection recorded separately in INTEGRATED_MANUSCRIPT_REVIEW.md")
    (OUT/"paper-build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("engine","machine","pages","pdf_sha256","overfull_boxes")},indent=2))


if __name__ == "__main__":
    main()
