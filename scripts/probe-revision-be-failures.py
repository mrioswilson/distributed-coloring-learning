#!/usr/bin/env python3
"""Post-test selected-model diagnostics; no training or primary-result writes.

Re-evaluate acquired and final banks, then replay the original first 32-case
cohort of BE/105010/native. The participation intervention is a local COMMIT
query only: the autonomous replay always uses the unchanged policy actions.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from dcl.multicf.oracles import confirmation
from dcl.multicf.protocol import cases
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, DETECT, KW, LINIAL, teacher_choices
from dcl.multiteacher_e1.model import native_prediction
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.paperrevision.keys import commit_key, key_text
from dcl.samplecomplexity.evaluation import evaluate, invariant_after
from dcl.samplecomplexity.training import initialize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper-revision-20261030"
SELECTED = (("BE_BASE_D3", (105001, 105002, 105004, 105008, 105010)),
            ("SPECULATIVE_GM", (105001,)))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def confusion(pred, truth):
    tp, tn = int((pred & truth).sum()), int((~pred & ~truth).sum())
    fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
    return dict(n=len(truth), tp=tp, tn=tn, fp=fp, fn=fn,
                ba=.5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else None)


@torch.inference_mode()
def inspect(model, teacher, bank, prefix, truth, grouped):
    logits = np.zeros(len(truth), np.float32)
    participation = np.zeros(len(truth), bool)
    candidates = np.zeros(len(truth), np.int64)
    for phase in sorted(np.unique(bank["native"][:, 3])):
        positions = np.flatnonzero(bank["native"][:, 3] == phase)
        for start in range(0, len(positions), 256):
            use = positions[start:start + 256]
            arrays = tuple(bank[f"a{j}"][use] for j in range(4))
            p, c = bank[prefix + "p"][use], bank[prefix + "c"][use]
            assert np.array_equal(confirmation(arrays, teacher, p, c) > .5, truth[use])
            state = model.first(tensor_arrays(arrays), int(phase))
            logits[use] = model.commit_for(state, torch.from_numpy(p), torch.from_numpy(c)).numpy()
            participation[use] = ((state.p_logits[:, 0] > 0) & (state.centers[:, 0, 7] > .5)).numpy()
            candidates[use] = state.scores[:, 0].argmax(-1).numpy()
            for j, i in enumerate(use):
                grouped[key_text(commit_key(arrays, j, p, c, teacher))].append(float(logits[i]))
    result = confusion(logits > 0, truth)
    eligible = bank["a0"][:, 0, 7] > .5
    if teacher == "BE_BASE_D3":
        eligible &= bank["a0"][:, 0, 18] >= 0
        assert np.array_equal(eligible, bank["p"][:, 0].astype(bool))
    occupancy = (bank["a2"][:, 0, :, -4:] > .5 if teacher == "BE_BASE_D3"
                 else bank["a1"][:, 0, :, 1:5] > .5)
    blocked = (occupancy & bank["a3"][:, 0, :, None]).any(1)[np.arange(len(truth)), candidates]
    result["first_stage"] = dict(
        participation=confusion(participation, bank["p"][:, 0].astype(bool)),
        eligible_queries=int(eligible.sum()),
        occupied_proposals_on_eligible=int((blocked & eligible).sum()),
        differs_from_native_proposal=int((candidates != bank["c"][:, 0]).sum()))
    return result


def replay_first_failure(model, config, stored):
    teacher, seed = "BE_BASE_D3", 105010
    # Preserve both membership and order of the original inference cohort.
    suite = cases(config, "final", seed)[:32]
    ids = {id(graph): meta["case_id"] for graph, meta in suite}
    selected = suite[0][1]["case_id"]
    witness = []
    original_apply = ColoringProgram.apply

    @torch.inference_mode()
    def observed(program, state, choices):
        before, old_aux = program.colors.copy(), program.aux.copy()
        original_apply(program, state, choices)
        if invariant_after(program, state, before, old_aux) or ids[id(program.graph)] != selected:
            return
        assert state.phase == KW and not witness
        assert ((old_aux >= 0) & (old_aux < state.bound)).all()
        assert all(old_aux[u] != old_aux[v] for u, v in program.graph.edges())
        roots = np.flatnonzero(state.permission)
        arrays = encode_roots(state, program.neighbors, roots)
        out = model(*tensor_arrays(arrays), phase=state.phase)
        assert np.array_equal(native_prediction(out)[0].numpy(), choices.color[roots])
        exact = teacher_choices(state, program.neighbors, teacher)
        offending = []
        for vertex in np.flatnonzero((program.aux < 0) | (program.aux >= state.target)):
            j = int(np.flatnonzero(roots == vertex)[0])
            forced_p = out.p.clone()
            forced_p[j, 0] = True
            assert torch.count_nonzero(forced_p != out.p).item() == 1
            h1, candidates = out.state.h1.clone(), out.candidates.clone()
            forced = model.commit_for(out.state, forced_p, out.candidates)[j].item()
            assert torch.equal(h1, out.state.h1) and torch.equal(candidates, out.candidates)
            actual_truth = confirmation(arrays, teacher, out.p.numpy(), out.candidates.numpy())
            forced_truth = confirmation(arrays, teacher, forced_p.numpy(), out.candidates.numpy())
            candidate = int(out.candidates[j, 0])
            decoded = int((state.bound - 1 - state.aux[vertex]) * 4 + candidate)
            assert actual_truth[j] == 0 and forced_truth[j] == 1
            assert out.commit[j] < 0 < forced and exact.color[vertex] == candidate + 1
            assert 0 <= decoded < state.target
            offending.append(dict(
                node=int(vertex), aux_before=int(old_aux[vertex]), aux_after=int(program.aux[vertex]),
                exact_color_action=int(exact.color[vertex]), actual_color_action=int(choices.color[vertex]),
                participation=bool(out.p[j, 0]), candidate=candidate,
                participation_logit=float(out.state.p_logits[j, 0]),
                actual_commit_logit=float(out.commit[j]), forced_active_commit_logit=float(forced),
                own_proposal_decodes_to=decoded, actual_oracle=int(actual_truth[j]),
                forced_active_oracle=int(forced_truth[j]), only_one_participation_bit_changed=True,
                first_stage_and_candidates_unchanged=True))
        assert offending
        witness.append(dict(
            case_id=selected, step=program.steps, phase=state.phase,
            bound_before=state.bound, bound_after=state.target,
            offending_vertices=offending, observer_actions_equal_original=True))

    ColoringProgram.apply = observed
    try:
        rows, _ = evaluate(model, teacher, suite, batch_size=1024, cohort=32)
    finally:
        ColoringProgram.apply = original_apply
    lookup = {row["case_id"]: row for row in stored["rows"]}
    fields = ("colors", "native_steps", "failure", "invariants", "terminal_correct")
    for row in rows:
        assert all(row[field] == lookup[row["case_id"]][field] for field in fields)
        assert row["technical_error"] is None
    assert len(rows) == 32 and len(witness) == 1
    return dict(
        scope="Selected first original cohort; local intervention only, no continuation or population rescue estimate",
        teacher=teacher, seed=seed, arm="native", replayed_rows=len(rows),
        batch_size=1024, cohort=32, selected_case=suite[0][1],
        original_terminal_rows_reproduced=True, matched_fields=list(fields), witness=witness)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "data/paper-revision-20261030")
    args = parser.parse_args()
    torch.set_num_threads(1)
    inputs = {}

    def track(path):
        path = path.resolve()
        # Runtime paths remain portable even when the restored root is elsewhere.
        if path.is_relative_to(args.runtime.resolve()):
            label = "runtime/" + str(path.relative_to(args.runtime.resolve()))
        else:
            label = str(path.relative_to(ROOT))
        digest = sha(path)
        if label in inputs:
            assert inputs[label]["sha256"] == digest
        inputs[label] = dict(path=str(path), sha256=digest)
        return path

    def read_json(path):
        return json.loads(track(path).read_text())

    def load_bank(path):
        with np.load(track(path), allow_pickle=False) as bank:
            return {key: bank[key] for key in bank.files}

    config = read_json(ROOT / "configs/paper-revision-20261030.json")
    freeze = read_json(OUT / "weights-frozen.json")
    protocol = read_json(OUT / "protocol-frozen.json")
    assert config == protocol["config"] and freeze["sources"] == protocol["sources"]
    for path, digest in freeze["sources"].items():
        assert sha(track(ROOT / path)) == digest
    track(Path(__file__))
    rows = []
    for teacher, seeds in SELECTED:
        for seed in seeds:
            data = args.runtime / "datasets" / teacher.lower() / f"s{seed}"
            testdir = args.runtime / "test" / teacher.lower() / f"s{seed}"
            for directory in (data, testdir):
                manifest = read_json(directory / "manifest.json")
                for path, digest in manifest["files"].items():
                    assert sha(track(directory / path)) == digest
            base, aux, test = load_bank(data / "base.npz"), load_bank(data / "aux.npz"), load_bank(testdir / "aux.npz")
            mask = ~np.isin(base["native"][:, 3], [LINIAL, DETECT])
            color = {key: value[mask] for key, value in base.items()}
            for arm in ("native", "cf"):
                directory = args.runtime / "models" / teacher.lower() / f"s{seed}" / arm
                checkpoint = track(directory / "final.pt")
                record = read_json(directory / "record.json")
                original = next(row for row in freeze["weights"]
                                if (row["teacher"], row["seed"], row["arm"]) == (teacher, seed, arm))
                assert sha(checkpoint) == original["sha256"] == record["checkpoint_sha256"]
                assert record["updates"] == 1500
                assert record["base_visited"] == len(base["native"]) and record["auxiliary_visited"] == len(aux["native"])
                model = initialize(teacher, "normalized", seed).eval()
                model.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])
                grouped = defaultdict(list)
                native_train = inspect(model, teacher, color, "", color["native"][:, 0] > 0, grouped)
                auxiliary_train = inspect(model, teacher, aux, "cf_" if arm == "cf" else "",
                                          aux["cf_y" if arm == "cf" else "native_y"] > .5, grouped)
                final = inspect(model, teacher, test, "cf_", test["cf_y"] > .5, grouped)
                stored = read_json(OUT / "prospective" / teacher.lower() / f"s{seed}" / (arm + "-test.json"))
                assert all(final[key] == stored["diagnostic"][key] for key in ("tp", "tn", "fp", "fn", "n"))
                row = dict(
                    teacher=teacher, seed=seed, arm=arm, checkpoint_sha256=sha(checkpoint),
                    native_train=native_train, auxiliary_train=auxiliary_train, final=final,
                    stored_final_confusion_reproduced=True, fine_types_checked=len(grouped),
                    max_within_type_logit_span=max(max(v) - min(v) for v in grouped.values()),
                    within_type_sign_conflicts=sum(min(v) <= 0 < max(v) for v in grouped.values()),
                    base_visited=record["base_visited"], auxiliary_visited=record["auxiliary_visited"])
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("teacher", "seed", "arm", "stored_final_confusion_reproduced")}), flush=True)
                if (teacher, seed, arm) == ("BE_BASE_D3", 105010, "native"):
                    replay = replay_first_failure(model, config, stored)
    assert all(sha(Path(entry["path"])) == entry["sha256"] for entry in inputs.values())
    result = dict(
        scope="Post-test selected frozen models; descriptive diagnosis, no retuning, independent replicas or new primary endpoint",
        selection="Both BE arms in five selected good/bad seeds and both GM arms in seed105001; not a prevalence sample",
        participation_scope="Auxiliary P/C are conditioning values, not participation-loss targets; first-stage scores there are evaluations",
        rows=rows, first_failure=replay,
        commit_predictions=sum(row[bank]["n"] for row in rows for bank in ("native_train", "auxiliary_train", "final")),
        final_commit_predictions=sum(row["final"]["n"] for row in rows),
        inputs={label: entry["sha256"] for label, entry in sorted(inputs.items())},
        inputs_unchanged_after_probe=True,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, threads=1),
        independent_scientific_review="Pending; this selected-model probe is not independent review")
    output = OUT / "be-failure-diagnostics.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), commit_predictions=result["commit_predictions"],
                          replayed_rows=replay["replayed_rows"])), flush=True)


if __name__ == "__main__":
    main()
