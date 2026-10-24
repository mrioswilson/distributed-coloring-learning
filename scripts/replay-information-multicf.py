#!/usr/bin/env python3
"""Replay JP ablations with intermediate Adam/RNG checkpoints and verify weights."""
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import torch

import dcl.multicf.training as training
from dcl.multicf.protocol import freeze
from dcl.multiteacher.programs import named_seed as native_seed
from dcl.samplecomplexity.training import initialize as base_initialize
from dcl.study.storage import atomic_json,file_hash

spec=importlib.util.spec_from_file_location("multicf_info",Path(__file__).with_name("information-multicf.py"))
info=importlib.util.module_from_spec(spec);sys.modules[spec.name]=info;spec.loader.exec_module(info)
ROOT=Path("results/multiteacher-counterfactual")


def job(args):
    config,seed,mode=args
    def initialize(teacher,architecture,seed):
        base=base_initialize(teacher,architecture,seed)
        model=info.InformationPolicy(mode);model.load_state_dict(base.state_dict())
        return model
    def seeds(*parts):
        if parts==(config["namespace"],"batch-order",seed):
            return native_seed(config["namespace"],"information-batches",seed)
        return native_seed(*parts)
    training.initialize=initialize;training.named_seed=seeds
    directory=ROOT/"information-resumable"/mode/f"s{seed}"
    training.train(dict(config,information_mode=mode),"JP","normalized",seed,
        ROOT/"information/datasets"/f"s{seed}",directory,"matched",500,[250,500])
    actual=json.loads((directory/"u000500-record.json").read_text())
    expected=json.loads((ROOT/"information"/mode/f"s{seed}"/"record.json").read_text())
    matched=actual["weights_hash"]==expected["weights_hash"]
    atomic_json(directory/"numerical-replay.json",dict(mode=mode,seed=seed,tensor_weights_identical=matched,
        original_checkpoint_sha256=expected["checkpoint_sha256"],replay_checkpoint_sha256=actual["checkpoint_sha256"],
        source_sha256=file_hash(__file__),information_adapter_sha256=file_hash("scripts/information-multicf.py"),
        scope="Reexecution with identical initialization, batches and labels; intermediate update250 and final update500 plus authoritative last.pt contain Adam/RNG. Additional computation, no new labels or training replica."))
    if not matched:raise RuntimeError("JP numerical training replay differs")
    return dict(mode=mode,seed=seed,matched=True)


if __name__=="__main__":
    config=json.loads(Path("configs/multiteacher-counterfactual.json").read_text())
    config.update(namespace=config["namespace"]+"-information",updates=500,checkpoints=[250,500])
    freeze(ROOT/"information-resumable/protocol.json",dict(config=config,source_sha256=file_hash(__file__),
        adapter_sha256=file_hash("scripts/information-multicf.py"),
        scope="Numerical reproducibility replay of already-fixed 20 JP ablations; save intermediate and final optimizer/RNG state without replacing the original results"))
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(job,[(config,s,m) for s in config["seeds"] for m in info.MODES]))
    atomic_json(ROOT/"information-resumable/replay-results.json",rows)
