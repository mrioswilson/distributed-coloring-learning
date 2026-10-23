"""Precompute fixed tests as banks become available; never adapt the design."""
import time
from .core import *
from .analyze import read,bank,successes

def main():
    directory=OUT/'tests';directory.mkdir(exist_ok=True)
    gs={g['id']:g for g in read(OUT/'graphs.json')}
    while True:
        queue=read(ROOT/'queue.json')
        for task in queue['tasks']:
            if task['status']!='complete' or task['stage']!='confirmatory' or task['model'] is None:continue
            s=task['model'];g=gs[task['graph']];path=directory/f"{g['id']}__{s['label']}.json"
            if path.exists():continue
            teacher_path=ROOT/'banks'/f"confirmatory__Luby__{g['id']}.npz"
            if not teacher_path.exists():continue
            a=bank('confirmatory',s['label'],g['id']);teacher=bank('confirmatory','Luby',g['id']);x=successes(teacher);y=successes(a)
            ps=seed('primary-permutation',g['id'],s['label']);r=mmd_test(x,y,np.random.default_rng(ps),B=19999)
            if 'error' in r:r.update(p=1,m=len(x),n=len(y))
            r.update(graph=g['id'],model=s['label'],architecture=s['treatment'],training_seed=s['seed'],n_vertices=g['n'],
                delta=g['delta'],family=g['family'],permutation_seed=ps,teacher_status=np.bincount(teacher['status'],minlength=4).tolist(),
                model_status=np.bincount(a['status'],minlength=4).tolist(),planned_m=250)
            atomic_json(path,r)
        count=len(list(directory.glob('*.json')))
        print('Fixed primary statistics archived',count,'/795',flush=True)
        if count==795:break
        if queue['gpu_seconds']>=43140:break
        time.sleep(10)
if __name__=='__main__':main()
