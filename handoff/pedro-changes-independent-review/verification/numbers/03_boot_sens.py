import json,sys,numpy as np
T=json.load(open(sys.argv[1])); seeds=range(105001,105011)
be=np.array([T[f'be_base_d3|{s}|cf-test']['ba']-T[f'be_base_d3|{s}|native-test']['ba'] for s in seeds])
lo=[];hi=[]
for k in range(1000):
    r=np.random.default_rng(k); s=be[r.integers(10,size=(10000,10))].mean(1); a,b=np.quantile(s,[.025,.975]); lo.append(a);hi.append(b)
lo=np.array(lo)*100;hi=np.array(hi)*100
print('lower: median %.2f  5-95%% [%.2f, %.2f]  min %.2f max %.2f'%(np.median(lo),*np.percentile(lo,[5,95]),lo.min(),lo.max()))
print('upper: median %.2f  5-95%% [%.2f, %.2f]'%(np.median(hi),*np.percentile(hi,[5,95])))
print('fraction of rng seeds with lower <= -15:',(lo<=-15).mean())
# leave-one-seed-out mean
for i,s in enumerate(seeds):
    print(s,'LOO mean diff %.2f'%(np.delete(be,i).mean()*100))
# excluding the 3 native-collapse seeds
print('mean diff excluding 105004/5/7: %.3f'%(np.delete(be,[3,4,6]).mean()*100))
