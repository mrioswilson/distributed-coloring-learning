from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path('data/ucgnn-local-palettes')
RESULTS = Path('results/ucgnn-local-palettes')
PREVIOUS = Path('/workspaces/AlgoLearning/ucgnn-paper-study')
PROTOCOL = Path('docs/LOCAL_PALETTES_STATISTICAL_PROTOCOL.md')
TREATMENTS = ('normalized', 'raw_endpoints', 'sum', 'mean', 'deepsets', 'transformer')
SEEDS = range(41000, 41005)
REGIMES = ('GLOBAL', 'LOCAL', 'OVERSIZED-1.5', 'OVERSIZED-2')
BUDGET_SECONDS = 12 * 3600


def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, ('local-palette-v1', *parts))).encode()).digest()[:8], 'big') % (2**63)


def model_specs():
    from dcl.study.storage import file_hash
    return [dict(label=f'{t}-s{s}', treatment=t, seed=s, epoch=120,
                 checkpoint=str(p), sha256=file_hash(p))
            for t in TREATMENTS for s in SEEDS
            for p in [PREVIOUS / f'data/ucgnn-paper-study/runs/{t}-s{s}/epoch-120.pt']]
