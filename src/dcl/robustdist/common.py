from pathlib import Path
import hashlib,json
import numpy as np
from dcl.study.storage import atomic_json,atomic_write,file_hash

OLD=Path('/workspaces/AlgoLearning/ucgnn-global-distribution-study')
OUT=Path('results/ucgnn-robust-distribution')
DATA=Path('data/ucgnn-robust-distribution')
def read(path):return json.loads(Path(path).read_text())
def seed(*parts):return int.from_bytes(hashlib.sha256(json.dumps(['robust-coloring-v1',*parts]).encode()).digest()[:8],'big')%(2**63)
def rng(*parts):return np.random.default_rng(seed(*parts))
def graphs():return read(OLD/'results/ucgnn-global-distribution/graphs.json')
def specs():return read(OLD/'results/ucgnn-global-distribution/checkpoints.json')
def valid(x,g):
    e=np.array(g['edges'],dtype=int).reshape(2,-1)
    return (x>=0).all(1)&(x<=np.array(g['degrees'])).all(1)&(x[:,e[0]]!=x[:,e[1]]).all(1)
def save_bank(path,**arrays):atomic_write(path,lambda f:np.savez_compressed(f,**arrays))
