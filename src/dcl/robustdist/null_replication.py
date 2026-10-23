"""An immutable-method diagnostic for the independent noisy-star null."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from .common import *
from .benchmark import trial,summarize

def main():
    frozen=read(OUT/'freeze.json')
    for name,digest in frozen['source_hashes'].items():assert file_hash(name)==digest,name
    stage='null-replication2'
    jobs=[(stage,'star_null',250,i,200,None) for i in range(4000)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(trial,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%100==0:print('Independent null replication',i,len(jobs),flush=True)
    _,rows=summarize(stage)
    target=next(r for r in rows if r['method']=='omnibus')
    result=dict(stage=stage,independent_trials=4000,nominal_alpha=.05,
                method_unchanged=True,flag_probability=.01,
                implementation_flag=target['null_exceedance_p']<.01,
                target=target,all_methods=rows,
                note='Post-freeze independent diagnostic; does not select a method or change any neural confirmatory inference.')
    atomic_json(OUT/'null-replication-diagnostic.json',result)
    print(result,flush=True)
if __name__=='__main__':main()
