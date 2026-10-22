"""Prospective independent-power banks after the original queue."""
from .core import *
from .analyze import read
from .run import main as run
from dcl.localpalettes.runner import persistent_gpu_lock

def main():
    with persistent_gpu_lock():
        q=read(ROOT/'queue.json')
        assert all(t['status']=='complete' for t in q['tasks'])
        ids={t['id'] for t in q['tasks']}
        for s in [None]+[s for s in specs() if s['seed']==41000]:
            label='Luby' if s is None else s['label'];ident=f'independent-power__{label}__path3'
            if ident not in ids:q['tasks'].append(dict(id=ident,stage='independent-power',m=77000,model=s,graph='path3',status='pending',supplement_sha256=file_hash('docs/GLOBAL_COLORING_POWER_SUPPLEMENT.md')))
        atomic_json(ROOT/'queue.json',q)
    run()
if __name__=='__main__':main()
