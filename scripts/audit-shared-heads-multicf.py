#!/usr/bin/env python3
"""Verify the shared confirmation pathway across paired palette architectures."""
import hashlib
import json
from pathlib import Path

import torch

from dcl.study.storage import atomic_json,file_hash

OUT=Path("results/multiteacher-counterfactual")
PREFIXES=("core.message2.","core.state2.","core.commit_head.","context_message2.","context_state2.")


if __name__=="__main__":
    rows=[]
    for teacher,budget in (("BE_BASE_D3",3000),("SPECULATIVE_GM",3000),("COLOR_TRIALS",30000)):
        for seed in range(94101,94106):
            for update in (1500,4000):
                heads={};hashes={}
                for arch in ("normalized","deepsets","transformer"):
                    p=OUT/"runs"/teacher.lower()/arch/f"s{seed}"/f"b{budget}-coverage1-alternate0/counterfactual"/f"u{update:06d}.pt"
                    state=torch.load(p,map_location="cpu",weights_only=False)["model"]
                    heads[arch]={k:v for k,v in state.items() if k.startswith(PREFIXES)}
                    h=hashlib.sha256()
                    for k,v in sorted(heads[arch].items()):h.update(k.encode());h.update(v.numpy().tobytes())
                    hashes[arch]=h.hexdigest()
                reference=heads["normalized"]
                equal=all(all(torch.equal(v,heads[a][k]) for k,v in reference.items()) for a in ("deepsets","transformer"))
                rows.append(dict(teacher=teacher,seed=seed,update=update,all_confirmation_tensors_identical=equal,
                    hashes=hashes,tensor_names=sorted(reference)))
    atomic_json(OUT/"audit/shared-confirmation-heads.json",dict(rows=rows,script_sha256=file_hash(__file__),
        explanation="For BE/GM, fixed teacher/counterfactual proposal losses use the same raw state and second-stage modules; candidate and participation losses do not depend on that stage. With paired initialization/batches, confirmation weights can therefore be identical across palette scorers. ColorTrials reads its candidate after stage two, coupling those gradients. These are correlated architecture comparisons, not independent confirmation-head replicas."))
    print({t:sum(r["all_confirmation_tensors_identical"] for r in rows if r["teacher"]==t) for t in ("BE_BASE_D3","SPECULATIVE_GM","COLOR_TRIALS")})
