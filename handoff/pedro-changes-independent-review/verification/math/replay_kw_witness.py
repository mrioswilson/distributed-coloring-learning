"""Independent arithmetic replay of the KW worked example (no repo imports)."""
import json, math, sys
d = json.load(open(sys.argv[1]))
n = d["n"]; adj = {v: set() for v in range(n)}
for a, b in d["edges"]:
    adj[a].add(b); adj[b].add(a)
assert all(len(adj[v]) == 3 for v in adj), "not cubic"
def proper(aux): return all(aux[a] != aux[b] for a, b in d["edges"])
# LINIAL: choice encodes point*13+value; new aux = point*field+value
aux = None
for step, field in ((0, 13), (1, 7)):
    ch = d["prefix"][step]["choices"]["auxiliary"]
    aux = [(c // 13) * field + (c % 13) for c in ch]
    assert proper(aux), step
print("after LINIAL", aux, "max", max(aux))
B = 49; step = 2
while step < d["native_step"]:
    q = math.ceil(4 * B / 5)
    col = d["prefix"][step]["choices"]["color"]
    new = aux[:]
    for v in range(n):
        if aux[v] >= q:
            opts = [(B - 1 - aux[v]) * 4 + c for c in range(4)]
            free = [c for c in range(4) if all(aux[u] != opts[c] for u in adj[v])]
            assert col[v] == free[0] + 1, (step, v, col[v], free)
            new[v] = opts[col[v] - 1]
        else:
            assert col[v] == 0, (step, v)
    aux = new; assert proper(aux), step
    B = q; step += 1
q = math.ceil(4 * B / 5); r = d["root"]
print("state at native step", step, "B", B, "q", q, "aux", aux)
opts = [(B - 1 - aux[r]) * 4 + c for c in range(4)]
nbr = sorted(aux[u] for u in adj[r])
print("root a_v", aux[r], "eligible", aux[r] >= q, "options", opts, "neighbor aux", nbr)
for c, p in zip([x[0] for x in d["cf_c"]], [x[0] for x in d["cf_p"]]):
    F = all(aux[u] != opts[c] for u in adj[r])
    O = int((aux[r] >= q) and p and F)
    print(f"query c={c} p={p}: F={F} O={O}")
print("stored cf_y", d["cf_y"], "native", d["native"])
