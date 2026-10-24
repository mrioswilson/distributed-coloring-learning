#!/usr/bin/env python3
"""Self-contained source-history/runtime delivery, with complete file digests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from dcl.study.storage import atomic_json,file_hash

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/"results/multiteacher-counterfactual"


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--work-dir",type=Path,required=True);a=p.parse_args()
    output=a.output_dir.resolve();work=a.work_dir.resolve();output.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    verification=json.loads((RUNTIME/"verification.json").read_text());paper=json.loads((RUNTIME/"paper-build.json").read_text())
    assert verification["passed"] and paper["visual_review_passed"] and paper["pdf_sha256"]==file_hash(ROOT/"paper/ucgnn/main.pdf")
    assert json.loads((RUNTIME/"tests-verification.json").read_text())["passed"]==668
    subprocess.run(["git","diff","--exit-code","HEAD","--"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    files=[p for p in sorted(RUNTIME.rglob("*")) if p.is_file() and p.name not in
           ("RUNTIME_MANIFEST.json","RUNTIME_FILES.sha256",".generation.lock") and p.suffix!=".pyc"]
    rows=[dict(path=str(p.relative_to(ROOT)),sha256=file_hash(p),bytes=p.stat().st_size) for p in files]
    tracked=subprocess.check_output(["git","ls-files","-z","--","src","scripts","tests","configs","docs","paper","requirements","README.md","pyproject.toml"],cwd=ROOT).decode().split("\0")
    atomic_json(RUNTIME/"RUNTIME_MANIFEST.json",dict(source_commit=head,branch=branch,base="61f39760",
        source_hashes={p:file_hash(ROOT/p) for p in tracked if p},files=rows,
        scope="All new raw data, model/optimizer/RNG checkpoints, failed candidates and audit artifacts. Manifest/checksum self-files excluded to avoid recursion; outer delivery hashes authenticate them."))
    (RUNTIME/"RUNTIME_FILES.sha256").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in rows))
    print(f"Hashed {len(rows)} runtime files",flush=True)
    bundle=work/"source.bundle"
    if bundle.exists():bundle.unlink()
    subprocess.run(["git","-c","pack.threads=2","bundle","create",str(bundle),branch],cwd=ROOT,check=True)
    subprocess.run(["git","bundle","verify",str(bundle)],cwd=ROOT,check=True,capture_output=True)
    print("Complete source-history bundle verified",flush=True)
    report=output/"Informe_P1-P14.md";pdf=output/"Articulo_actualizado.pdf"
    shutil.copyfile(ROOT/"docs/MULTITEACHER_COUNTERFACTUAL_RESULTS.md",report)
    shutil.copyfile(ROOT/"paper/ucgnn/main.pdf",pdf)
    readme=work/"LEEME.md"
    readme.write_text(f'''# Investigación multiteacher: entrega reproducible

Cobertura21/21: doce certificaciones históricas preservadas y nueve nuevas,
cada una con cinco checkpoints y1000/1000 ejecuciones por semilla. El informe
P1–P14 conserva todos los resultados negativos y distingue los protocolos.

Código e historia Git completos: source.bundle, rama {branch}, commit {head}.
Artículo actualizado:36 páginas. Suite completa:668 tests aprobados.
No se hizo push.

Para restaurar, extraiga este archivo en un directorio nuevo y ejecute:

```bash
git clone --branch {branch} source.bundle research
cp -a results/multiteacher-counterfactual research/results/
cd research
sha256sum -c results/multiteacher-counterfactual/RUNTIME_FILES.sha256
```

El README del directorio de resultados documenta entorno, validación,
reanudación, reproducción, fases, presupuestos y la errata de metadatos CT.
source.bundle incluye la historia del baseline, por lo que la auditoría de
fuentes no depende de que un commit nuevo haya sido publicado en GitHub.
Los pesos originales de Pedro siguen ausentes; los21 archivos originales
M6 autenticados sí están copiados en historical-recovery/m6-originals/.
''')
    delivery=dict(source_commit=head,branch=branch,source_bundle_sha256=file_hash(bundle),
        runtime_manifest_sha256=file_hash(RUNTIME/"RUNTIME_MANIFEST.json"),runtime_files=len(rows),
        runtime_bytes=sum(r["bytes"] for r in rows),paper_sha256=file_hash(pdf),report_sha256=file_hash(report),
        no_push=True)
    atomic_json(work/"DELIVERY_MANIFEST.json",delivery)
    archive=output/"Evidencia_reproducible.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=4) as tar:
        for path,name in ((bundle,"source.bundle"),(readme,"LEEME.md"),(work/"DELIVERY_MANIFEST.json","DELIVERY_MANIFEST.json"),
                          (pdf,pdf.name),(report,report.name)):
            tar.add(path,arcname=name,recursive=False)
        for path in sorted(RUNTIME.rglob("*")):
            if path.is_file() and path.name!=".generation.lock" and path.suffix!=".pyc":
                tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    artifacts=[report,pdf,archive]
    (output/"SHA256SUMS.txt").write_text("".join(f"{file_hash(p)}  {p.name}\n" for p in artifacts))
    atomic_json(output/"Manifiesto_entrega.json",dict(**delivery,artifacts=[dict(name=p.name,bytes=p.stat().st_size,sha256=file_hash(p)) for p in artifacts]))
    # Validate a full archive stream and every member's size; SHA manifests
    # provide per-file content authentication after extraction.
    with tarfile.open(archive,"r:gz") as tar:
        members=tar.getmembers()
        assert next(m for m in members if m.name=="source.bundle").size==bundle.stat().st_size
        assert len([m for m in members if m.name.startswith("results/")])==len(files)+2
    print(json.dumps(dict(source_commit=head,runtime_files=len(rows),archive_bytes=archive.stat().st_size,
                          outputs=[str(p) for p in artifacts])),flush=True)
