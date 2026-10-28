#!/usr/bin/env python3
"""Independent closure accounting. Reads frozen outcomes; never runs training.

Run from the repository root with PYTHONPATH=src. Outputs are deliberately
separate from the historical study and its frozen manifests.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import networkx as nx
import numpy as np
import scipy
from scipy.stats import beta

from dcl.degree.graphs import graph_for
from dcl.study.graphs import fingerprint

BASE = Path("results/multiteacher-degree-generalization")
OUT = Path("results/final-closure")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    if rows:
        with path.open("w") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None:
            assert digest == expected, (str(path), digest, expected)
        sources[str(path)] = dict(sha256=digest, bytes=path.stat().st_size)
        return json.loads(path.read_text())

    inv = read(BASE / "checkpoint-inventory.json") + read(BASE / "supplemental-checkpoint-inventory.json")
    weights = {cp["id"]: cp for cp in inv}
    for cp in inv:
        assert sha(cp["path"]) == cp["sha256"], cp["id"]
        sources[cp["path"]] = dict(sha256=cp["sha256"], bytes=Path(cp["path"]).stat().st_size)
    plan = read(BASE / "confirmatory/design.json")
    for path, digest in plan["sources"].items():
        assert sha(path) == digest, path
        sources[path] = dict(sha256=digest, bytes=Path(path).stat().st_size)
    assert set(plan["checkpoints"]) == set(plan["checkpoint_hashes"])
    grouped = defaultdict(Counter)
    cells, failures, stages = [], [], {}
    graph_cache = {}
    for stage in ("pilot", "confirmatory", "larger-size-pilot"):
        index = read(BASE / stage / "index.json")
        stage_total = Counter()
        seen = set()
        unique_graphs, unique_tapes, unique_cases = set(), set(), set()
        for cell in index:
            cp = weights[cell["checkpoint_id"]]
            key = cp["id"], cell["condition"]["id"]
            assert key not in seen, key
            seen.add(key)
            assert cell["sha256"] == cp["sha256"]
            if stage == "confirmatory":
                assert plan["checkpoint_hashes"][cp["id"]] == cp["sha256"]
            rows = read(cell["rows_path"], cell["rows_sha256"])
            assert len(rows) == cell["executions"]
            assert len({r["case_id"] for r in rows}) == len(rows)
            n_ok = 0
            by_family = Counter()
            for row in rows:
                assert row["n"] == cell["condition"]["n"]
                assert row["delta"] == row["actual_delta"] == cell["condition"]["delta"]
                assert row["palette"] == cell["condition"]["palette"]
                assert row["teacher"] == cp["teacher"]
                assert row["family"] in cell["condition"]["families"]
                gkey = row["family"], row["n"], row["delta"], row["graph_seed"]
                if gkey not in graph_cache:
                    g = graph_for(*gkey)
                    degree = np.array([g.degree(v) for v in range(len(g))])
                    edges = np.array(list(g.edges()), dtype=int).reshape(-1, 2).T
                    assert len(g) == row["n"] and degree.max() == row["delta"]
                    assert not nx.number_of_selfloops(g)
                    graph_cache[gkey] = degree, edges, fingerprint(g)
                    # The generator has its own cache; retain arrays only here.
                    graph_for.cache_clear()
                degree, (a, b), fp = graph_cache[gkey]
                assert fp == row["fingerprint"]
                colors = np.array(row["colors"])
                assert len(colors) == row["n"]
                palette = degree + 1 if row["palette"] == "local" else row["delta"] + 1
                conflicts = int(np.sum((colors[a] >= 0) & (colors[a] == colors[b])))
                violation = int(np.sum((colors < -1) | (colors >= palette)))
                incomplete = int(np.sum(colors < 0))
                assert (conflicts, violation, incomplete) == (row["conflicts"], row["palette_violations"], row["incomplete"])
                ok = not conflicts and not violation and not incomplete and row["invariants"] and row["failure"] is None
                assert bool(ok) == row["success"]
                n_ok += ok
                by_family[row["family"]] += 1
                unique_graphs.add((row["n"], row["delta"], row["fingerprint"]))
                unique_tapes.add(row["tape_seed"])
                unique_cases.add((row["case_id"], row["palette"]))
                for name, value in (("teacher", cp["teacher"]), ("architecture", cp["architecture"]),
                                    ("checkpoint", cp["id"]), ("seed", cp["seed"]), ("n", row["n"]),
                                    ("delta", row["delta"]), ("palette", row["palette"]),
                                    ("family", row["family"]), ("cohort", cp["cohort"]),
                                    ("condition", cell["condition"]["id"]), ("primary", cp["primary"])):
                    d = grouped[stage, name, str(value)]
                    d["executions"] += 1
                    d["successes"] += ok
                    d["failures"] += not ok
                stage_total["executions"] += 1
                stage_total["successes"] += ok
                stage_total["failures"] += not ok
                if not ok:
                    stage_total["failure_" + str(row["failure"])] += 1
            assert n_ok == cell["successes"]
            c = dict(stage=stage, checkpoint=cp["id"], sha256=cp["sha256"], teacher=cp["teacher"],
                     architecture=cp["architecture"], cohort=cp["cohort"], primary=cp["primary"],
                     method=cp["method"], seed=cp["seed"], condition=cell["condition"]["id"],
                     n=cell["condition"]["n"], delta=cell["condition"]["delta"], palette=cell["condition"]["palette"],
                     executions=len(rows), successes=n_ok, failures=len(rows)-n_ok,
                     cp95_lower=(float(beta.ppf(.025, n_ok, len(rows)-n_ok+1)) if n_ok else 0.) if stage == "confirmatory" else None,
                     cp95_upper=(float(beta.ppf(.975, n_ok+1, len(rows)-n_ok)) if n_ok < len(rows) else 1.) if stage == "confirmatory" else None,
                     interval_scope="pointwise_95_under_declared_iid_mixture" if stage == "confirmatory" else "exploratory_no_confidence_interval",
                     rows_path=cell["rows_path"], rows_sha256=cell["rows_sha256"])
            cells.append(c)
            if n_ok != len(rows):
                first = next(r for r in rows if not r["success"])
                failures.append(dict(**c, first_case_id=first["case_id"], graph_seed=first["graph_seed"],
                                     tape_seed=first["tape_seed"], family=first["family"], failure=first["failure"],
                                     first_divergence=first.get("first_divergence"),
                                     first_action_divergence=first.get("first_action_divergence")))
            if len(cells) % 100 == 0:
                print(stage, "cells", len(cells), "rows", stage_total["executions"], flush=True)
        exact = read(BASE / stage / "exact-validation.json")
        exact_n = 0
        for meta in exact["files"]:
            refs = read(meta["path"], meta["sha256"])
            assert len(refs) == meta["executions"] and all(r["success"] for r in refs)
            exact_n += len(refs)
        stages[stage] = dict(stage_total, checkpoint_conditions=len(index),
                             failed_cells=sum(c["failures"] > 0 for c in cells if c["stage"] == stage),
                             exact_reference_executions=exact_n, unique_labeled_graphs=len(unique_graphs),
                             unique_tape_seeds=len(unique_tapes), unique_case_palette_pairs=len(unique_cases))
    assert stages["confirmatory"]["executions"] == 810000
    assert stages["confirmatory"]["successes"] == 808413
    assert stages["confirmatory"]["failed_cells"] == 15
    # Verify previous derived tables, not just their overall headline.
    previous = list(csv.DictReader((BASE / "checkpoint-condition.csv").open()))
    lookup = {(c["stage"], c["checkpoint"], c["condition"]): c for c in cells}
    for r in previous:
        c = lookup[r["stage"], r["checkpoint"], r["condition"]]
        assert int(r["executions"]) == c["executions"] and int(r["successes"]) == c["successes"]
    replays = read(BASE / "failure-replays/index.json")
    assert len(replays) == 15
    for path in replays:
        r = read(path)
        assert r["replay_matches_original"]
        assert r["sha256"] == weights[r["checkpoint"]]["sha256"]
    write_csv(OUT / "generalization-by-factor.csv", [dict(stage=k[0], factor=k[1], level=k[2], **v) for k, v in sorted(grouped.items())])
    write_csv(OUT / "generalization-cells.csv", cells)
    write_json(OUT / "failed-cells.json", failures)
    write_json(OUT / "verified-inputs.json", sources)
    report = dict(stages=stages, checkpoint_hashes_verified=len(inv), preserved_failure_replays_verified=len(replays),
                  reconstructed_graphs=len(graph_cache), verified_input_files=len(sources),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, networkx=nx.__version__,
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  script_sha256=sha(__file__), seconds=time.perf_counter()-started, new_training_updates=0,
                  independence="Graphs/tapes are shared across teachers, cores, and treatments. Each checkpoint/condition has its own 1000 outcomes; pooled accounting is not a policy estimand. Training seed is the unit for treatment replication. Stored invariant flags are checked against terminal outcomes but intermediate invariants require preserved traces/replay.")
    write_json(OUT / "generalization-audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
