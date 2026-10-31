import json, sys
d = json.load(open(sys.argv[1])); n = d["n"]
adj = {v: set() for v in range(n)}
for a, b in d["edges"]: adj[a].add(b); adj[b].add(a)
def poly(v, f):
    out = [0]*f; pw = [1]*f
    while v:
        out = [(o + (v % f)*p) % f for o, p in zip(out, pw)]; v //= f; pw = [p*x % f for p, x in zip(pw, range(f))]
    return out
aux1 = [(c//13)*13 + c%13 for c in d["prefix"][0]["choices"]["auxiliary"]]
ch = d["prefix"][1]["choices"]["auxiliary"]
for v in range(n):
    P = poly(aux1[v], 7)
    blocked = [any(poly(aux1[u], 7)[x] == P[x] for u in adj[v]) for x in range(7)]
    x = blocked.index(False)
    assert (ch[v]//13, ch[v] % 13) == (x, P[x]), (v, ch[v], x, P[x])
print("stage-2 Linial choices are first unblocked points with correct values: OK")
