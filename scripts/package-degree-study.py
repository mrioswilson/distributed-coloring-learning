#!/usr/bin/env python3
"""Deterministic local runtime archive, including all 225 original weights."""
import gzip
import json
from pathlib import Path
import tarfile
import time
from dcl.degree.inventory import ROOT
from dcl.study.storage import atomic_json,file_hash


if __name__=='__main__':
    assert json.loads((ROOT/'verification.json').read_text())['complete']
    archive=ROOT/'frozen-degree-runtime.tar.gz'
    excluded={'runtime-manifest.json','artifact-index.json','clean-recovery.json','package-command.txt'}
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in excluded and p.suffix not in ('.gz','.tmp'))
    entries=[dict(path=str(p),sha256=file_hash(p),bytes=p.stat().st_size) for p in paths]
    manifest=ROOT/'runtime-manifest.json'
    atomic_json(manifest,dict(files=entries,files_count=len(entries),bytes=sum(x['bytes'] for x in entries),
        scientific_training_updates=0,archive_scope='All local degree-study raw rows, traces, plans, audits, figures and 225 frozen checkpoint files. Repository source and manuscript are versioned separately.'))
    paths.append(manifest)
    tick=time.perf_counter()
    temporary=archive.with_suffix('.tmp')
    with temporary.open('wb') as output:
        with gzip.GzipFile(filename='',fileobj=output,mode='wb',compresslevel=4,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as tf:
                for path in sorted(paths):
                    info=tf.gettarinfo(str(path),arcname=str(path))
                    info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                    with path.open('rb') as f:tf.addfile(info,f)
    temporary.replace(archive)
    result=dict(path=str(archive),sha256=file_hash(archive),bytes=archive.stat().st_size,
        uncompressed_bytes=sum(x['bytes'] for x in entries),files=len(paths),checkpoints=sum(p.suffix=='.pt' for p in paths),
        manifest_sha256=file_hash(manifest),wall_seconds=time.perf_counter()-tick,
        availability='Local workspace artifact only; no upload, push or public release performed.')
    atomic_json(ROOT/'artifact-index.json',result)
    print(json.dumps(result))
