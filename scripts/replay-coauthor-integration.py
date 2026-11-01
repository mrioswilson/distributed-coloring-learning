#!/usr/bin/env python3
"""Run fresh executions in a new directory using the original sealed drivers.

Unlike the original drivers' resumable mode, the destination must not exist.
No historical record or integrated reference outcome is overwritten.
"""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/integrated-paper-20261031"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("functional", "transfer"))
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--limit",type=int,default=0,choices=(0,32),help="32 is a technical first-cohort check, not a full scientific replay")
    a = p.parse_args()
    destination = a.destination.resolve()
    if destination.exists():
        raise FileExistsError("Choose a new destination; existing results are never deleted")
    script = ROOT / "scripts" / ("integrated-functional-diagnostic.py" if a.kind == "functional" else "integrated-replay-transfer.py")
    spec = importlib.util.spec_from_file_location("sealed_integration_driver", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    destination.mkdir(parents=True)
    module.OUT = destination
    if a.kind == "functional":
        shutil.copyfile(REFERENCE / "functional-freeze.json", destination / "functional-freeze.json")
        freeze, digest = module.verify_freeze()
        # Serial execution avoids spawning workers with a different output root.
        for seed in module.SEEDS:
            for arm in module.ARMS:
                for mode in module.MODES:
                    print(json.dumps(module.execute((seed, arm, mode, a.limit, digest))), flush=True)
                    name=f"s{seed}-{arm}-{mode}.json.gz"
                    with gzip.open(destination/("functional-smoke" if a.limit else "functional")/name,"rt") as f:actual=json.load(f)
                    with gzip.open(REFERENCE/"functional"/name,"rt") as f:expected=json.load(f)
                    assert actual["rows"] == expected["rows"][:a.limit or None]
        if not a.limit:
            module.summarize()
            observed = json.loads((destination / "functional-summary.json").read_text())["rows"]
            original = json.loads((REFERENCE / "functional-summary.json").read_text())["rows"]
            # Elapsed time is measured again; all scientific summaries must match.
            fields = ("seed","arm","mode","runs","successes","original_successes","rescued","newly_failed",
                      "failures","same_terminal_rows_as_original","technical_errors","mean_steps","diagnostics")
            assert all(all(x[k] == y[k] for k in fields) for x,y in zip(observed, original))
    else:
        assert not a.limit, "--limit only applies to the functional technical check"
        shutil.copyfile(REFERENCE / "transfer-replay-freeze.json", destination / "transfer-replay-freeze.json")
        sys.argv = [str(script)]
        module.main()
    (destination / "fresh-replay-verification.json").write_text(json.dumps(dict(kind=a.kind,
        scientific_summaries_match=True, original_outputs_preserved=True,limit=a.limit,
        scope="Fresh fixed-battery reproduction, not a new test distribution"), indent=2)+"\n")


if __name__ == "__main__":
    main()
