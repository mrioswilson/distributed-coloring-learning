from pathlib import Path
import hashlib
import json
import numpy as np
from dcl.study.storage import atomic_json, atomic_write, file_hash, now
from dcl.artifacts import resolve_artifact

ROOT = Path('data/ucgnn-size-degree')
OUT = Path('results/ucgnn-size-degree')
BASE = Path('/workspaces/AlgoLearning/ucgnn-robust-distribution-tests')
GLOBAL = Path('/workspaces/AlgoLearning/ucgnn-global-distribution-study')
NS = (128, 256, 512, 1024)
DS = (3, 10, 50, 100)

def read(path):
    return json.loads(Path(path).read_text())

def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(['size-degree-2026-v1', *parts]).encode()).digest()[:8], 'big') % (2**63)

def rng(*parts):
    return np.random.default_rng(seed(*parts))

def specs():
    records = read(OUT / 'checkpoints.json')
    return [dict(s, historical_checkpoint=s['checkpoint'],
                 checkpoint=str(resolve_artifact(s['checkpoint'], s['sha256']))) for s in records]

def graphs():
    return read(OUT / 'graphs.json')

def save_bank(path, **a):
    atomic_write(path, lambda f: np.savez_compressed(f, **a))

def bank(label, gid, phase='confirmation'):
    with np.load(ROOT / 'banks' / f'{phase}__{label}__{gid}.npz') as a:
        return {k: a[k] for k in a.files}

def verify_frozen():
    f = read(OUT / 'freeze.json')
    for path, sha in f['source_hashes'].items():
        if file_hash(path) != sha:
            raise RuntimeError(f'Frozen source changed: {path}')
    for name in ('graphs', 'checkpoints'):
        if file_hash(OUT / f'{name}.json') != f[f'{name}_sha256']:
            raise RuntimeError(f'Frozen {name} changed')
    if file_hash('docs/FACTORIAL_SIZE_DEGREE_PROTOCOL.md') != f['protocol_sha256']:
        raise RuntimeError('Frozen protocol changed')
    return f
