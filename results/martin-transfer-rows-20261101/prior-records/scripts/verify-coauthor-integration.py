#!/usr/bin/env python3
"""Verify delivered hashes, sealed sources, archived runtimes and replay rows."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/integrated-paper-20261031"


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def archive_members(archive, records):
    expected={r["path"]:r for r in records}
    with tarfile.open(archive) as tf:
        assert set(tf.getnames()) == set(expected)
        for member in tf:
            assert member.isfile() and not member.name.startswith("/") and ".." not in Path(member.name).parts
            row=expected[member.name]
            assert member.size == row["bytes"]
            assert hashlib.sha256(tf.extractfile(member).read()).hexdigest() == row["sha256"]
    return len(expected)


def passed(row):
    return all(row.get(k) is True for k in ("proper","palette","invariants","terminated","no_technical_error","no_cutoff","terminal_correct"))


def release_audit(restored):
    directory=ROOT/"results/martin-artifacts-20261031"
    pins={"pedrorev-cf.tar.gz":"8353b7135cefad7159832ad0b4df901a6e0cd6168e57b0872a49d041d84fe2cc",
          "pedrorev-cf-followup.tar.gz":"6a009a4abd5853b9e7259e82d3ee458bdbbff7a6720c99b1eada5559a67773bb",
          "confirmatory-rows.tar.gz":"ed1039ac91f572eeb4c0e5e5b06e630d08c6d407a24aceb28b14fd57233ac489"}
    downloads=read(directory/"download-receipt.json")["files"]
    assert len(downloads)==9
    for record in downloads:
        if Path(record["path"]).name in pins:
            assert record["sha256"]==pins[Path(record["path"]).name]
        if restored or not record["path"].endswith(".tar.gz"):
            p=ROOT/record["path"]
            assert p.stat().st_size==record["bytes"] and sha(p)==record["sha256"],str(p)
    members=read(directory/"member-index.json")
    assert members["verified_all_archive_members"] and len(members["files"])==1988
    assert sum(r["scientific_payload"] for r in members["files"])==1976
    assert len({(r["package"],r["member"]) for r in members["files"]})==1988
    if restored:
        for r in members["files"]:
            p=ROOT/r["path"]
            assert p.stat().st_size==r["bytes"] and sha(p)==r["sha256"],r["path"]
    for name in ("model-replay-freeze.json","raw-audit-freeze.json"):
        frozen=read(directory/name)
        for path,digest in {**frozen["sources"],**frozen["inputs"]}.items():assert sha(ROOT/path)==digest,path
    frozen=read(directory/"model-replay-freeze.json");digest=sha(directory/"model-replay-freeze.json")
    inventory=read(directory/"checkpoint-inventory-audit.json")
    assert inventory["freeze_sha256"]==digest
    assert (inventory["references"],inventory["unique_frozen_paths"],inventory["reused_control_paths"],inventory["training_directories"],inventory["checkpoint_files"])==(220,205,15,105,315)
    assert inventory["snapshots_by_filename"]=={"last.pt":105,"u001500.pt":105,"u004000.pt":105}
    assert all(inventory[k] for k in ("all_frozen_weight_hashes_match","all_saved_response_hashes_match","all_acquired_bank_hashes_match","original_source_freezes_match","shared_initial_hash_metadata_agrees"))
    labels=read(directory/"raw-query-label-audit.json")
    metrics=read(directory/"recomputed-metrics-audit.json")
    assert labels["freeze_sha256"]==metrics["freeze_sha256"]==digest
    assert sum(r["queries"] for r in labels["records"])==985992
    assert all(r["raw_oracle_labels_and_annotations_match"] for r in labels["records"])
    assert sum(r["rows"] for r in metrics["records"])==466 and all(r["all_metrics_equal"] for r in metrics["records"])
    for study in ("cf-experiment","cf-followup"):
        assert read(directory/(study+"-recomputed-analysis.json"))==read(ROOT/"results/pedro-revision"/study/"analysis.json")
    inference=read(directory/"model-reinference-audit.json")
    assert inference["freeze_sha256"]==digest and inference["models_and_batteries"]==len(inference["records"])==220
    assert inference["decisions"]==sum(r["queries"] for r in inference["records"])==50173920
    assert inference["all_decisions_match"] and all(r["sign_mismatches"]==0 for r in inference["records"])
    assert inference["bitwise_equal_models"]==sum(r["bitwise_equal"] for r in inference["records"])==0
    assert inference["max_abs_error"]==max(r["max_abs_error"] for r in inference["records"])==3.0517578125e-05
    for reference,record in zip(frozen["references"],inference["records"]):
        assert all(record[k]==reference[k] for k in ("study","teacher","seed","arm","update"))
        assert record["checkpoint_sha256"]==reference["sha256"] and record["freeze_sha256"]==digest
        assert sha(ROOT/record["raw_recomputed_logits"])==record["raw_recomputed_sha256"]
        if restored:
            import numpy as np
            assert sha(ROOT/reference["checkpoint"])==reference["sha256"]
            if record["study"]=="cf-experiment":
                original=ROOT/f"data/pedrorev-cf/responses/{record['teacher'].lower()}/s{record['seed']}-{record['arm']}-u{record['update']:06d}.npy"
            else:original=ROOT/f"data/pedrorev-cf-followup/responses/s{record['seed']}-{record['arm']}.npy"
            assert sha(original)==record["original_response_sha256"]
            old=np.load(original,allow_pickle=False)
            with gzip.open(ROOT/record["raw_recomputed_logits"],"rb") as f:new=np.load(f,allow_pickle=False)
            assert len(new)==record["queries"] and np.array_equal(new>0,old>0)
            assert float(np.max(np.abs(new-old)))==record["max_abs_error"]
    functional=read(directory/"functional-replay-audit.json")
    assert functional["freeze_sha256"]==digest and functional["evaluations"]==len(functional["records"])==120
    assert functional["executions"]==120000 and functional["all_original_summary_fields_match"] and functional["technical_errors"]==0
    executions=0
    for record in functional["records"]:
        assert all(record["matches"].values()) and record["technical_errors"]==0 and sha(ROOT/record["path"])==record["sha256"]
        with gzip.open(ROOT/record["path"],"rt") as f:d=json.load(f)
        assert d["summary"]==record["summary"] and d["freeze_sha256"]==digest
        source=ROOT/"results/pedro-revision"/record["study"]/"functional"
        if record["study"]=="cf-experiment":source=source/record["teacher"].lower()
        source=source/f"s{record['seed']}-{record['arm']}.json"
        assert sha(source)==d["source_summary_sha256"]
        original=read(source)
        assert all(value==original[k] for k,value in record["summary"].items())
        assert len(d["rows"])==len({r["case_id"] for r in d["rows"]})==record["summary"]["executions"]==1000
        assert sum(passed(r) for r in d["rows"])==record["summary"]["successes"]
        assert all(r["technical_error"] is None for r in d["rows"])
        executions+=len(d["rows"])
    assert executions==120000
    terminal=read(directory/"raw-terminal-audit.json")
    assert terminal["frozen_input_sha256"]==sha(directory/"raw-audit-freeze.json")
    assert terminal["totals"]==dict(executions=810000,successes=808413,failures=1587)
    assert terminal["primary"]==dict(executions=450000,successes=449791,failures=209)
    assert terminal["failure_kinds"]==dict(collision=1559,cutoff=28)
    assert terminal["cells"]==len(terminal["summaries"])==810 and terminal["failed_cells"]==15
    assert terminal["reconstructed_graphs"]==terminal["unique_labeled_graphs"]==4000
    assert terminal["all_stored_terminal_fields_agree"] and terminal["all_compact_cell_counts_agree"]
    assert not terminal["full_trajectories_reconstructed"] and terminal["new_inference_executions"]==0
    separate=read(directory/"separate-transfer-availability.json")
    assert separate["missing_original_raw_files"]==len(separate["rows"])==240
    assert separate["all_240_design_summary_inventory_identities_match"] and separate["driver_matches_design_sha256"]
    other=ROOT/"results/pedro-revision/transfer-experiment"
    design=read(other/"design.json");outcomes=read(other/"summary.json")["rows"]
    cross={(t,s,a,c["id"]) for t,s,a,_ in design["checkpoints"] for c in design["conditions"]}
    assert cross=={(r["teacher"],r["seed"],r["arm"],r["condition"]) for r in outcomes} and len(cross)==240
    assert sum(r["executions"] for r in outcomes)==sum(r["successes"] for r in outcomes)==240000
    original_rows={str(other.relative_to(ROOT)/p):h for h,p in (line.split(maxsplit=1) for line in (other/"ROWS.sha256").read_text().splitlines())}
    assert original_rows=={r["path"]:r["sha256"] for r in separate["rows"]}
    assert not set(original_rows)&{r["path"] for r in members["files"]}
    recovered=read(directory/"source-recovery/receipt.json")
    assert sha(ROOT/recovered["path"])==recovered["sha256"]==design["transfer_source_sha256"]
    assert sha(ROOT/recovered["git_path"])==recovered["current_source_sha256"]
    assert recovered["ast_equal_after_removing_optional_observer_and_nullable_output_field"]
    assert sha(ROOT/"scripts/pedrorev-transfer-experiment.py")==design["script_sha256"]
    fresh=read(directory/"fresh-replay-checks.json")
    assert sha(ROOT/"scripts/replay-release-artifacts.py")==fresh["wrapper_final_source_sha256"]
    assert fresh["original_raw_source_guard_passed"] and fresh["raw_terminal_reconstruction"]["all_executed_checks_passed"]
    assert fresh["one_checkpoint_fresh_inference"]["decisions"]==449280
    assert fresh["one_checkpoint_fresh_inference"]["sign_mismatches"]==0
    assert fresh["one_checkpoint_fresh_inference"]["verification"]["all_executed_checks_passed"]
    editorial=ROOT/"results/paper-editorial-cleanup-20261101"
    visual=read(editorial/"pdf-visual-review.json" if (editorial/"pdf-visual-review.json").exists() else directory/"pdf-visual-review.json")
    assert visual["pdf_sha256"]==sha(ROOT/"paper/ucgnn/main.pdf")
    assert visual["pages"]==read(OUT/"paper-build.json")["pages"] and visual["all_pages_inspected_via_contact_sheets"]
    for path,h in visual["images"].items():assert sha(ROOT/path)==h,path
    current_missing=sum(not (ROOT/p).is_file() for p in original_rows)
    for p,expected_raw in original_rows.items():
        if (ROOT/p).exists():assert sha(ROOT/p)==expected_raw,p
    prior=read(directory/"prior-integration.json")
    assert sha(ROOT/prior["path"])==prior["sha256"]
    with tarfile.open(ROOT/prior["path"]) as tf:
        name="results/integrated-paper-20261031/manifest.json"
        raw=tf.extractfile(name).read();assert hashlib.sha256(raw).hexdigest()==prior["prior_manifest_sha256"]
        old=json.loads(raw)
        assert len(tf.getmembers())==prior["files"]==844
        for row in old["files"]:
            assert hashlib.sha256(tf.extractfile(row["path"]).read()).hexdigest()==row["sha256"]
    if (editorial/"prior-integration.json").exists():
        previous=read(editorial/"prior-integration.json")
        assert sha(ROOT/previous["path"])==previous["sha256"]
        with tarfile.open(ROOT/previous["path"]) as tf:
            name="results/integrated-paper-20261031/manifest.json"
            raw=tf.extractfile(name).read()
            assert hashlib.sha256(raw).hexdigest()==previous["prior_manifest_sha256"]
            assert len(tf.getmembers())==previous["files"]==1462
            for row in json.loads(raw)["files"]:
                assert hashlib.sha256(tf.extractfile(row["path"]).read()).hexdigest()==row["sha256"]
    return dict(archive_hashes_verified=3,member_inventory=1988,restored_members_verified=1988 if restored else 0,
        checkpoint_references=220,distinct_frozen_paths=205,trainings=105,query_labels=985992,
        model_replay_sealed_sources=len(frozen["sources"]),
        recomputed_metric_rows=466,reinferred_binary_decisions=50173920,binary_comparison_rechecked=restored,
        functional_evaluations=120,functional_executions=120000,raw_terminal_colorings=810000,
        prior_delivery_preserved=True,separate_transfer_original_raw_files_missing=current_missing,
        separate_transfer_historical_source_recovered=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--restored-runtime",action="store_true")
    args=p.parse_args()
    manifest=read(OUT/"manifest.json")
    for row in manifest["files"]:
        path=ROOT/row["path"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],row["path"]
    source=read(OUT/"sources.json")
    deliberate=set(manifest["deliberate_import_edits"])
    for path,digest in source["imported_sha256"].items():
        if path not in deliberate:
            assert sha(ROOT/path) == digest,path
    for extension in ("tex","pdf"):
        assert sha(ROOT/f"paper/ucgnn/archive/pre-integration-5993a5e4.{extension}") == source["initial_sha256"][f"paper/ucgnn/main.{extension}"]
    freeze=read(OUT/"functional-freeze.json")
    for path,digest in freeze["sources"].items():
        assert sha(ROOT/path) == digest,path
    build=read(OUT/"paper-build.json")
    assert not build["unresolved_references"] and not build["overfull_boxes"]
    assert sha(ROOT/"paper/ucgnn/main.pdf") == build["pdf_sha256"]
    for path,digest in build["actual_inputs"].items():assert sha(ROOT/path) == digest,path
    assets=read(OUT/"assets.json")
    for path,digest in assets["outputs"].items():assert sha(ROOT/path) == digest,path
    for row in (OUT/"pdf-fonts.txt").read_text().splitlines()[2:]:
        assert row.split()[-5] == "yes",row
    historical=read(ROOT/"results/paper-revision-20261030/runtime-index.json")
    members=[dict(row,path=historical["runtime_directory"]+"/"+row["path"]) for row in historical["files"]]
    old_members=archive_members(ROOT/"results/paper-revision-20261030/paper-revision-runtime.tar.gz",members)
    new_members=archive_members(OUT/"integration-replay-runtime.tar.gz",read(OUT/"integration-runtime-index.json")["files"])
    if args.restored_runtime:
        for row in members+read(OUT/"integration-runtime-index.json")["files"]:
            assert sha(ROOT/row["path"]) == row["sha256"],row["path"]
        for w in freeze["weights"]:assert sha(ROOT/w["path"]) == w["sha256"]
    total=rescued=0
    baseline_fields=("case_id","colors","native_steps","failure","invariants","terminal_correct","technical_error")
    summary=read(OUT/"functional-summary.json")
    for record in summary["rows"]:
        seed,arm,mode=(record[k] for k in ("seed","arm","mode"))
        name=f"s{seed}-{arm}-{mode}.json.gz"
        with gzip.open(OUT/"functional"/name,"rt") as f:d=json.load(f)
        assert sha(OUT/"functional-freeze.json") == d["freeze_sha256"]
        assert len(d["rows"]) == record["runs"] == 1000
        assert len({r["case_id"] for r in d["rows"]}) == 1000
        assert sum(passed(r) for r in d["rows"]) == record["successes"]
        assert all(r["technical_error"] is None for r in d["rows"])
        original=read(ROOT/f"results/paper-revision-20261030/prospective/be_base_d3/s{seed}/{arm}-test.json")["rows"]
        assert [r["case_id"] for r in d["rows"]] == [r["case_id"] for r in original]
        rescue=[r["case_id"] for r,b in zip(d["rows"],original) if passed(r) and not passed(b)]
        harm=[r["case_id"] for r,b in zip(d["rows"],original) if not passed(r) and passed(b)]
        assert rescue == d["rescued_case_ids"] and len(rescue) == record["rescued"]
        assert harm == d["newly_failed_case_ids"] == []
        if mode == "baseline":
            assert all(all(r[k] == b[k] for k in baseline_fields) for r,b in zip(d["rows"],original))
        if mode == "commit" and arm == "cf":rescued+=len(rescue)
        total+=len(d["rows"])
    assert total == 30000 and rescued == 1799
    transfer=read(OUT/"transfer-replay-summary.json")
    assert transfer["runs"] == 15000 and transfer["failed_observations"] == 1587 and transfer["all_match"]
    for r in transfer["summaries"]:
        assert all(r["matches"].values()) and sha(ROOT/r["artifact"]) == r["sha256"]
        with gzip.open(ROOT/r["artifact"],"rt") as f:d=json.load(f)
        assert len(d["rows"]) == r["runs"] == 1000
        assert sum(x["success"] for x in d["rows"]) == r["successes"]
    inference=read(OUT/"continuation-reinference.json")
    assert inference["models"] == 40 and inference["decisions"] == 160000 and inference["all_original_metrics_equal"]
    evidence=read(OUT/"evidence-reconciliation.json")
    assert evidence["original_20_pair_audit_exactly_reproduced"] and evidence["additional_summary_rows_validated"] == 466
    native=read(ROOT/"results/final-closure/native-certifications.json")
    assert native["packages"] == 21 and native["selected_policies"] == len(native["checkpoints"]) == 105
    assert all(r["successes"] == r["executions"] == 1000 for r in native["checkpoints"])
    assert set(Counter((r["teacher"],r["architecture"]) for r in native["checkpoints"]).values()) == {5}
    cells=list(csv.DictReader((ROOT/"results/final-closure/generalization-cells.csv").open()))
    confirm=[r for r in cells if r["stage"] == "confirmatory"]
    primary=[r for r in confirm if r["primary"] == "True"]
    assert len(confirm) == 810 and len(primary) == 450
    assert sum(int(r["successes"]) for r in confirm) == 808413
    assert sum(int(r["successes"]) for r in primary) == 449791
    assert sum(int(r["executions"]) for r in primary) == 450000
    assert sum(int(r["failures"]) > 0 for r in primary) == 2
    by_combination=defaultdict(list)
    for r in primary:by_combination[r["teacher"],r["architecture"]].append(r)
    assert len(by_combination) == 18
    assert sum(all(int(r["failures"]) == 0 for r in group) for group in by_combination.values()) == 16
    for p,digest in inference["inputs"].items():
        if args.restored_runtime or not p.startswith("data/"):assert sha(ROOT/p) == digest,p
    release=release_audit(args.restored_runtime)
    result=dict(manifest_sha256=sha(OUT/"manifest.json"),files_verified=len(manifest["files"]),
        immutable_imports_verified=len(source["imported_sha256"])-len(deliberate),sealed_sources_verified=len(freeze["sources"]),
        runtime_archive_members_verified=old_members+new_members,restored_runtime_verified=args.restored_runtime,
        functional_executions=total,cf_commit_rescues=rescued,historical_replay_executions=15000,
        continuation_decisions=160000,pages=build["pages"],overfull_boxes=0,unresolved_references=0,
        compact_record_accounting=dict(native_packages=21,native_successes=105000,
            primary_transfer_successes=449791,primary_transfer_runs=450000,all_transfer_successes=808413,
            all_transfer_runs=810000,primary_failed_cells=2,primary_all_pass_combinations=16),
        release_audit=release,
        result="passed",limitation="Does not rerun all810000 neural trajectories or reconstruct every intermediate invariant. Separate240000-run contrast still lacks240 original raw files, retained as summary-level evidence. Fixed-battery replication is not a new independent test or independent scientific review")
    (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
