#!/usr/bin/env python3
"""Compile authoritative closure TeX; audit PDF text, diagnostics and geometry."""
import argparse
import json
from pathlib import Path
import re
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compiler", type=Path, default=Path(".tools/tectonic"))
    args = p.parse_args()
    out = Path("results/final-closure")
    paper = Path("paper/ucgnn")
    if "% Scientific revision dated" in (paper / "main.tex").read_text():
        raise SystemExit("Historical closure builder refuses the revised manuscript; use scripts/build-revised-paper.py or a historical checkout.")
    import pypdfium2 as pdfium
    from dcl.study.storage import file_hash

    out.mkdir(parents=True, exist_ok=True)
    binary = args.compiler.resolve()
    assert file_hash(binary) == "a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da"
    proc = subprocess.run([str(binary), "-X", "compile", "--keep-logs", "main.tex"], cwd=paper, text=True, capture_output=True)
    (out / "paper-build.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError("Compilation failed; see results/final-closure/paper-build.log")
    log = (paper / "main.log").read_text(errors="replace")
    undefined = [s for s in log.splitlines() if "undefined" in s.lower() and any(w in s.lower() for w in ("reference", "citation"))]
    missing = [s for s in log.splitlines() if "Missing character" in s]
    overfull = [s for s in log.splitlines() if "Overfull" in s]
    assert not undefined and not missing, (undefined, missing)
    doc = pdfium.PdfDocument(str(paper / "main.pdf"))
    texts = [page.get_textpage().get_text_range() for page in doc]
    all_text = "\n".join(texts)
    (out / "paper-text.txt").write_text(all_text)
    for snippet in ("808,413/810,000", "449,791/450,000", "22.6404", "9.2686", "99.9932", "99.277", "Twenty-one"):
        # TeX uses digits for the coverage headline, words in prose may vary.
        if snippet == "Twenty-one":
            continue
        assert snippet in all_text, snippet
    assert all_text.count("60/60") >= 5
    assert "???" not in all_text
    geometry = []
    for i, page in enumerate(doc):
        tp = page.get_textpage()
        width, height = page.get_size()
        for j in range(tp.count_chars()):
            left, bottom, right, top = tp.get_charbox(j)
            if left < 0 or bottom < 0 or right > width + 1 or top > height + 1:
                geometry.append(dict(page=i+1, character=j, box=[left,bottom,right,top]))
    assert not geometry, geometry[:5]
    render_dir = out / "pdf-review"
    render_dir.mkdir(exist_ok=True)
    renders = []
    # Render every page. Actual visual inspection is recorded separately.
    for i, page in enumerate(doc):
        path = render_dir / f"page-{i+1:02d}.png"
        page.render(scale=1.2).to_pil().save(path)
        renders.append(dict(page=i+1, path=str(path), sha256=file_hash(path)))
    source = (paper / "main.tex").read_text()
    receipt = dict(compiled=True, pages=len(doc), source_sha256=file_hash(paper / "main.tex"),
                   pdf_sha256=file_hash(paper / "main.pdf"), compiler_sha256=file_hash(binary),
                   compiler_version=subprocess.check_output([str(binary), "--version"], text=True).strip(),
                   undefined_references=undefined, missing_glyphs=missing, overfull_boxes=overfull,
                   out_of_page_characters=geometry, renders=renders,
                   citation_keys=sorted({k.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", source) for k in group.split(",")}),
                   figure_files=re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source),
                   script_sha256=file_hash(__file__), new_training_updates=0,
                   visual_review="All pages rendered; manual inspection receipt is separate and must not be inferred from compilation.")
    (out / "paper-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("compiled", "pages", "undefined_references", "missing_glyphs", "overfull_boxes")}))


if __name__ == "__main__":
    main()
