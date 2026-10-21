"""Explicit CPU held-out E1 candidate study, overlapping fixed GPU training."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from dcl.study.storage import atomic_json,file_hash,now
from dcl.universal.train import load_model
from .config import ROOT,HISTORICAL
from .mechanisms import DIRECTORY,freeze_calibration,selection_probe,export_mechanisms
from .predictors import freeze_analytic


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    amendment="docs/UCGNN_PAPER_PROTOCOL_AMENDMENTS.md"
    commit=subprocess.check_output(["git","log","-1","--format=%H","--",amendment],text=True).strip()
    if not commit:raise RuntimeError("The prospective addendum must be committed first")
    frozen=subprocess.check_output(["git","show",f"{commit}:{amendment}"])
    if hashlib.sha256(frozen).hexdigest()!=file_hash(amendment):raise RuntimeError("Addendum has uncommitted changes")
    checkpoint=next((HISTORICAL/"data/ucgnn/runs").glob("e1-*-s00-*/epoch-120.pt"))
    model,_=load_model(checkpoint,"cpu")
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    calibration=freeze_calibration(model,checkpoint)
    analytic=freeze_analytic(model,checkpoint,DIRECTORY/"frozen-analytic-quantiles.json")
    registration={"addendum_commit":commit,"addendum_sha256":file_hash(amendment),"utc_before_heldout":now(),
                  "calibration_sha256":file_hash(DIRECTORY/"frozen-quantile-calibration.json"),
                  "analytic_sha256":file_hash(DIRECTORY/"frozen-analytic-quantiles.json"),
                  "device":"cpu","dtype":"torch.float32","torch_version":torch.__version__,
                  "checkpoint_sha256":file_hash(checkpoint),"source_hashes":{str(p):file_hash(p) for p in
                    [Path(__file__),Path("src/dcl/paperstudy/mechanisms.py"),Path("src/dcl/paperstudy/predictors.py")]}}
    path=DIRECTORY/"cpu-prediction-registration.json"
    if not path.exists():atomic_json(path,registration)
    started=time.monotonic()
    spec={"label":"historical-e1-s0-epoch120","treatment":"E1","seed":0}
    selection_probe(model,spec,True,calibration,analytic=analytic)
    export_mechanisms()
    atomic_json(DIRECTORY/"cpu-prediction-complete.json",{"complete":True,"seconds":time.monotonic()-started,
                "device":"cpu","completed_utc":now(),"addendum_commit":commit})


if __name__=="__main__":
    main()
