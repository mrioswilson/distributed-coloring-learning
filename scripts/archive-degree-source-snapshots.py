#!/usr/bin/env python3
"""Recover all pre-amendment source versions and authenticate their old hashes.

The pilot preceded the final action-trace fields and scheduling optimizations.
Reverse only the documented edits; a snapshot is accepted only at the exact
SHA-256 committed in the corresponding pre-outcome plan.
"""
import hashlib
import json
from pathlib import Path
import re
from dcl.degree.inventory import ROOT
from dcl.study.storage import atomic_json


def pilot_version(path,source,recovery):
    if path.endswith('/adapter.py'):
        source=source.replace('''            if overrides is not None and "query" in overrides:
                # Declared evaluation-only intervention on COMMIT's query;
                # never used in an autonomous rollout.
                query = torch.from_numpy(np.asarray(overrides["query"], np.int64)[roots])
                if np.any((query.numpy() < 0) | (query.numpy() >= program.palette[roots])):
                    raise ValueError("Counterfactual query outside receiver palette")
''','')
    if path.endswith('/graphs.py'):
        source=source.replace('from functools import lru_cache\n','').replace('@lru_cache(maxsize=6000)\n','')
    if path.endswith('/evaluation.py'):
        source=source.replace('first_divergence=None, first_action_divergence=None,','first_divergence=None,')
        source=source.replace('''                        if not bool(equal.all()) and st["first_action_divergence"] is None:
                            st["first_action_divergence"] = dict(step=p.steps, phase=state.phase,
                                roots=r[~equal].tolist(), component=first)
                            if len(traces) < trace_limit:
                                first_traces[i] = dict(first_component_trace=first_traces[i],
                                    first_action_trace=trace_state(p, state, decision, exact, first))
''','')
        source=source.replace('''                first_action_divergence=st["first_action_divergence"],
''','')
        if not recovery:
            source=source.replace(', wall_budget_seconds=None):','):')
            source=source.replace('''        batch_started = time.perf_counter()
''','')
            source=source.replace('''            if wall_budget_seconds is not None and time.perf_counter() - batch_started >= wall_budget_seconds:
                for p in programs:
                    if not p.finished:
                        p.finished, p.failure = True, "resource_cutoff"
                break
''','')
            source=source.replace('cutoff=p.failure in ("cutoff", "resource_cutoff")','cutoff=p.failure == "cutoff"')
            source=source.replace('''    return rows, traces, dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                             wall_budget_seconds=wall_budget_seconds)
''','''    return rows, traces, dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
''')
    if path.endswith('/run.py'):
        start=source.index('def suite_for(');end=source.index('def worker(',start)
        source=source[:start]+'''def suite_for(plan, condition, replica):
    return [make_case(plan["namespace"], condition["n"], condition["delta"],
                      condition["families"][i % len(condition["families"])],
                      i, replica) for i in range(condition["runs"])]


'''+source[end:]
        start=source.index('    reference_jobs = ');end=source.index('    atomic_json(ROOT / stage / "exact-validation.json"',start)
        source=source[:start]+'''    for teacher in active_teachers:
        for c in plan["conditions"]:
            for replica in range(5):
                path = ROOT / stage / "exact" / teacher / c["id"] / f"r{replica}.json"
                if path.exists():
                    rows = json.loads(path.read_text())
                else:
                    cases = suite_for(plan, c, replica)
                    rows, _, cost = evaluate(None, teacher, cases, c["palette"], diagnostics=False)
                    atomic_json(path, rows)
                if not all(r["success"] for r in rows):
                    raise RuntimeError(f"Exact teacher failed; student evaluation forbidden: {teacher}/{c['id']}/{replica}")
                validated.append(dict(path=str(path), sha256=file_hash(path), executions=len(rows)))
'''+source[end:]
        source=source.replace('''        conds = [c for c in conds if not c.get("primary_only") or cp["primary"]]
''','')
        source=source.replace('''        inventory += json.loads((ROOT / "supplemental-checkpoint-inventory.json").read_text())
''','')
        if not recovery:
            source=source.replace('''                                      batch=plan["batch"], diagnostics=True, trace_limit=1,
                                      wall_budget_seconds=10. if stage == "pilot" and cp["teacher"] == "COLOR_TRIALS" and condition["delta"] >= 50 else None)
''','''                                      batch=plan["batch"], diagnostics=True, trace_limit=1)
''')
            source=source.replace('''        amendment = ROOT / "pilot" / "amendment-v2.json"
        if amendment.exists():
            expected = json.loads(amendment.read_text())["sources_after"]
            if any(file_hash(p) != sha for p, sha in expected.items()):
                raise RuntimeError("Pilot recovery sources differ from amendment")
''','')
    return source


if __name__=='__main__':
    plans=[('pilot-initial',json.loads((ROOT/'pilot/design.json').read_text())['sources'],False),
           ('pilot-recovery',json.loads((ROOT/'pilot/amendment-v2.json').read_text())['sources_after'],True),
           ('confirmatory',json.loads((ROOT/'confirmatory/design.json').read_text())['sources'],None)]
    index=[];bad=[]
    for stage,expected,recovery in plans:
        for path,sha in expected.items():
            data=Path(path).read_bytes()
            if hashlib.sha256(data).hexdigest()!=sha:
                data=pilot_version(path,data.decode(),recovery).encode()
            if hashlib.sha256(data).hexdigest()!=sha:
                bad.append((stage,path,hashlib.sha256(data).hexdigest(),sha))
                continue
            target=ROOT/'source-snapshots'/stage/path
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            index.append(dict(stage=stage,path=str(target),sha256=sha))
    atomic_json(ROOT/'source-snapshots/index.json',dict(verified=index,unresolved=bad))
    if bad:raise RuntimeError(bad)
    print(json.dumps(dict(authenticated_source_files=len(index),unresolved=0)))
