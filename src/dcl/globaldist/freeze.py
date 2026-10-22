import json
from .core import *
from dcl.study.storage import provenance,now

def main():
    path=OUT/'freeze.json'
    if path.exists():raise RuntimeError('Already frozen')
    pilot=json.loads((OUT/'pilot.json').read_text())
    assert pilot['positive_rejections']>=95
    gs=graphs();ss=specs()
    assert len(gs)==53 and len(ss)==15
    assert len({s['sha256'] for s in ss})==15
    atomic_json(OUT/'graphs.json',gs);atomic_json(OUT/'checkpoints.json',ss)
    atomic_json(path,dict(protocol_sha256=file_hash('docs/GLOBAL_COLORING_DISTRIBUTION_PROTOCOL.md'),
        graphs_sha256=file_hash(OUT/'graphs.json'),checkpoints_sha256=file_hash(OUT/'checkpoints.json'),
        environment=provenance(),created=now(),pilot=pilot,m=250,B=19999,alpha=.05,family_size=795))
    print('Frozen',path)
if __name__=='__main__':main()
