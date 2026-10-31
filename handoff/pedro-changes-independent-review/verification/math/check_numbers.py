import numpy as np, math
nat=[99.435,99.334,99.539,50.193,50.320,99.152,50.348,99.195,99.462,99.451]
cf=[52.756,99.872,99.987,99.923,99.885,99.897,99.923,52.094,99.859,99.949]
d=np.array(cf)-np.array(nat)
print("mean nat",np.mean(nat),"mean cf",np.mean(cf),"mean diff",d.mean())
rng=np.random.default_rng(0)
bs=[d[rng.integers(0,10,10)].mean() for _ in range(10000)]
print("pct CI",np.percentile(bs,[2.5,97.5]))
# KW chain
B=49;seq=[B];
while B>4:
    q=math.ceil(4*B/5); assert 4*(B-q)<=q; B=q; seq.append(B)
print(seq,len(seq)-1)
# Linial
print(13**5>=65536, 13>3*4, 7**3>=169, 7>3*2)
# Luby/Johansson bound sanity: worst case brute force small
# Clopper-Pearson
print(0.025**(1/1000),0.05**(1/1000))
# BE pooled confusion
def ba(tp,tn,fp,fn): return 0.5*(tp/(tp+fn)+tn/(tn+fp))
print("BE native pooled",ba(995,0+91+26956,228+42+11688,0))
print("BE cf pooled",ba(814,228+52+38644,81,181))
print("BE gate pooled",ba(995,133+38644,228,0))
print("GM gate pooled",ba(15554,1483,22963,0))
print("sign test",2/2**5)
print("unseen BA matched?")
pos=[52,108,90,56,60]; neg=[497,351,627,466,662]
cfc=[549,437,717,510,722]; mc=[158,342,229,182,246]
def mba(correct): return np.mean([0.5*(1+(c-p)/n) for c,p,n in zip(correct,pos,neg)])
print("unseen BA cf",mba(cfc),"matched",mba(mc))
posall=[3070,2895,2891,2773,3115]; print("pos frac",[p/99840 for p in posall], "always-reject acc", 1-np.mean(posall)/99840)
print("CF historical acc", 1-34/499200)
