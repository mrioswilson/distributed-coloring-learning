#!/usr/bin/env python3
"""Create machine-readable claim provenance and complete failure/cost tables."""
import csv
import json
from pathlib import Path
import re
import subprocess

from dcl.study.storage import atomic_json, file_hash

OUT = Path("results/final-closure")
DEG = Path("results/multiteacher-degree-generalization")
CF = Path("results/multiteacher-counterfactual")


def reference(path):
    p = Path(path)
    assert p.is_file(), str(p)
    return dict(path=str(p), sha256=file_hash(p), bytes=p.stat().st_size)


def main():
    inventory = json.loads((DEG / "checkpoint-inventory.json").read_text()) + json.loads((DEG / "supplemental-checkpoint-inventory.json").read_text())
    cp_by_id = {r["id"]: r for r in inventory}
    native = json.loads((OUT / "native-certifications.json").read_text())
    be = json.loads((OUT / "counterfactual-audit.json").read_text())["rows"]
    failures = json.loads((OUT / "failed-cells.json").read_text())
    replay = json.loads((OUT / "failure-replay-audit.json").read_text())["rows"]
    ids = lambda predicate: [r["id"] for r in inventory if predicate(r)]
    claims = []

    def claim(identifier, text, kind, files, protocol, cpids, seeds, result, limits, confidence):
        claims.append(dict(id=identifier, claim=text, status=kind, sources=[reference(p) for p in files],
                           protocol=protocol, checkpoint_ids=cpids, seeds=seeds, result=result,
                           limitations=limits, confidence=confidence))

    claim("C01", "21/21 native packages have five all-pass checkpoints under their respective protocols", "finite experimental result",
          [OUT / "native-certifications.json", DEG / "native-21-consolidation.csv", "results/sample-complexity/certification.json", CF / "plans/successful-policies.json"],
          "Twelve passive historical packages; nine new CF packages; 1000 native runs per checkpoint; different selection/battery strata and budgets.",
          ids(lambda r: r["primary"]), [83101,83102,83103,83104,83105,94101,94102,94103,94104,94105],
          "105000/105000 selected native rows; all hashes and terminal properness/list validity checked.",
          "Not a homogeneous cohort, theoretical minimum, new-training success probability, universal correctness or learned termination proof.", "high within finite authenticated scope")
    claim("C02", "BE native action fidelity and typed continuation fidelity separate", "paired development diagnostic",
          [OUT / "counterfactual-audit.json", OUT / "review-ml-be-strata.json", CF / "diagnostics/paired-u004000-BE_BASE_D3.json"],
          "Same existing 60 teacher trajectories per seed; 4000 updates; E1; original RNG17001 queries reconstructed.",
          [r["checkpoint"] for r in be], list(range(94101,94106)),
          "Each arm 28800/28800 native actions per seed; means 22.640425/9.268630/99.993189%; BA60.141255/53.253641/99.996494%.",
          "3000/3000/6776 labels; repeated development; only five paired training seeds; labels heavily imbalanced; oracle is explicitly chosen continuation.", "high descriptive; insufficient intervention-only causal identification")
    claim("C03", "CF advantage persists on some unseen sufficient local COMMIT types", "posthoc diagnostic subset",
          [OUT / "counterfactual-audit.json", OUT / "review-ml-be-strata.json"],
          "Sufficient queried-slot BE input keys; compare against combined native+CF training bank; original diagnostic queries only.",
          [r["checkpoint"] for r in be], list(range(94101,94106)),
          "Unseen occurrence denominators549,459,717,522,722; positives52,108,90,56,60; CF549,437,717,510,722 vs matched158,342,229,182,246 correct.",
          "Most occurrence types seen; full-input novelty differs; same query template; all methods accept all positives; not prospective OOD family holdout.", "moderate descriptive posthoc; no confirmatory generalization claim")
    claim("C04", "BE/GM shared confirmation heads do not yield independent architecture replication", "tensor identity audit",
          [CF / "audit/shared-confirmation-heads.json", "scripts/audit-shared-heads-multicf.py", "src/dcl/multicf/training.py"],
          "Hash/equality of fixed-proposal stage-two parameters at1500/4000; paired initialization and minibatches.",
          ids(lambda r: r["teacher"] in ("BE_BASE_D3","SPECULATIVE_GM")), list(range(94101,94106)),
          "BE/GM confirmation tensors identical across three palette architectures at audited points; CT differs.",
          "Not independent cores for this diagnostic; different candidate scorers and full trajectories still can differ.", "high for audited tensor identities")
    claim("C05", "Fresh fixed-label optimizer contrast isolates additional updates for BE more closely than CF contrast", "paired independent final comparison",
          ["results/sample-complexity/optimization-confirmation/results.json", "results/sample-complexity/optimization-confirmation/frozen-plan.json"],
          "E1 nominal3000 labels, identical seed/bank lineage; 500 vs1500 updates; new paired final tapes.", [], list(range(83101,83106)),
          "BE500 successes72,90,84,92,42/1000; BE1500 all1000; Luby/JP all1000 at both points.",
          "Only these banks/teachers/lineages; not CF causality, optimal updates, or monotonicity theorem.", "high within paired protocol; limited external scope")
    claim("C06", "JP representation matters under the tested budget", "controlled exploratory development ablation",
          [CF / "information/protocol.json", CF / "information/results.json", "scripts/information-multicf.py"],
          "Same E1 dimensions/init,3000labels,500updates; without IDs/numeric IDs/differences/relative signs.", [], list(range(94101,94106)),
          "Relative comparisons60/60 all five; raw representations finite failures retained.",
          "Not final1000 certification; not impossibility of learning numeric IDs; coherent signs required.", "high for controlled descriptive result")
    claim("C07", "Confirmatory transfer totals reconciled from raw data", "frozen confirmatory accounting",
          [OUT / "generalization-audit.json", OUT / "generalization-cells.csv", OUT / "generalization-by-factor.csv", OUT / "reviewer3-statistics-check.json", DEG / "confirmatory/design.json"],
          "180 extensible checkpoints;90primary five conditions,90controls four conditions;1000runs/cell; exact references first.",
          ids(lambda r: r["teacher"] != "BE_BASE_D3"), [83101,83102,83103,83104,83105,94101,94102,94103,94104,94105],
          "810000 rows;808413correct;1587failures;15/810 failedcells;primary449791/450000,2/450failedcells.",
          "4000graphs/20000tapes shared across policies; aggregate accounting not mean-policy probability; intervals pointwise and conditional on independence/batch neutrality.", "high exact accounting; conditional per-policy inference")
    claim("C08", "Selected-condition primary transfer is broad but not universal", "frozen confirmatory result",
          [OUT / "generalization-cells.csv", DEG / "confirmatory/design.json", "paper/ucgnn/tables/closure-generalization.tex"],
          "Six teachers,threecores,fiveseeds;globaln64delta3/4/10,localn64delta4,globaln256delta4.",
          ids(lambda r: r["primary"] and r["teacher"] != "BE_BASE_D3"), [83101,83102,83103,83104,83105,94101,94102,94103,94104,94105],
          "16/18 teacher/core combinations pass all conditions in all five seeds;JohanssonT83101global10 and GMT94102local4 fail.",
          "No BE extension; delta/K/private scale confounding; not full cross-product of maxima.", "high within selected conditions")
    claim("C09", "Every failed confirmatory cell has a reproduced first failure", "deterministic diagnostic replay",
          [OUT / "failure-replay-audit.json", OUT / "failed-cells.json", "scripts/final-failure-replays.py"],
          "First incorrect raw row per cell, original full inference batch; frozen weights, exact same graph/tape.",
          sorted({r["checkpoint"] for r in failures if r["stage"] == "confirmatory"}), sorted({r["seed"] for r in replay}),
          "15/15 terminal status/colors/steps reproduced;Johanssonproposal/progress;GMCFmissedrepair;GMcontrolsforbiddenwrite;CTconfirmationcollisions.",
          "One witness per cell not every mechanism per failure; first discrepancy not sufficient cause; latentalpha not semantic availability mask.", "high replay facts; causal scope mechanism-specific")
    claim("C10", "High degree200 and size4096 are exploratory evidence", "separate pilot results",
          [OUT / "generalization-audit.json", DEG / "pilot/design.json", DEG / "pilot/amendment-v2.json", DEG / "larger-size-pilot/design.json"],
          "One run per regular checkpoint/cell in scans;42denseCT resourcecutoffs with heterogeneous amendment;later parallel-teacher size pilot.",
          ids(lambda r: r["primary"] and r["teacher"] != "BE_BASE_D3"), [83101,83102,83103,83104,83105,94101,94102,94103,94104,94105],
          "Pilot3526/3570; later size240/240 on4labeledgraphs/20tapes.",
          "No statistical certification of maxima;different teacher coverage, regularity and resource limits.", "high observed counts; low generalization strength")
    claim("C11", "CT scale and GM palette contexts expose specific frozen-rule limitations", "posthoc typed mechanism probes",
          [DEG / "ct-normalization-sensitivity.json", DEG / "palette-degree-mechanism.json", "scripts/probe-ct-normalization.py", "scripts/isolate-gm-palette-failure.py"],
          "Typed private CT gap1 with rootdegree1/list2 and separate degree star;GM exact repair prefix with fixed degree3/IDs and global4/local2 lists.",
          ids(lambda r: r["primary"] and r["teacher"] in ("COLOR_TRIALS","SPECULATIVE_GM")), list(range(94101,94106)),
          "CT15CF reject queried strictwinner fromdelta10 in mainembedding;GM selectedTransformer changes repairproposal with palette.",
          "Not random functional failure rates;embeddings both preserve information;not proof of numerical cause or architecture impossibility.", "high finite witnesses; not universal causality")
    claim("C12", "Exact teachers have conditional safety/progress guarantees", "reviewed conditional propositions",
          ["paper/ucgnn/main.tex", "docs/MATHEMATICAL_CLAIMS_AUDIT.md", "src/dcl/degree/teachers.py", "src/dcl/multiteacher/programs.py", "docs/reviews/REVIEW_1_DISTRIBUTED.md"],
          "Simplegraphs;declaredlists;ideal independentrandomness;coherentIDs;schedules/algebra explicitly supplied;D=3nativeCT,D=delta transfer.", [], [],
          "Availability;Luby/Johansson>=1/4;CTpositive;JP<=n;Greedyn;GM<=2ceil(n/min(4,n))+n;BEonlydegree3.",
          "Not arbitrary neural networks, PRNG or hardware certificates;no learned worst-case complexity.", "high conditional proof confidence; independently reviewed internally")
    claim("C13", "MAX certificate, information obstruction and law arguments have limited premises", "reviewed interface propositions",
          ["paper/ucgnn/main.tex", "docs/MATHEMATICAL_CLAIMS_AUDIT.md", DEG / "commit-certificates/index.json", "src/dcl/paperstudy/certification.py"],
          "Canonical6types,nonnegativeMAX,one-slotgather;observationindistinguishability;cliqueK=n equivariance;validcoupling.",
          ids(lambda r: r["teacher"] in ("LUBY","JOHANSSON","GREEDY")), list(range(83101,83106)),
          "45selectedcanonicalcores fail sufficient certificate;5differenthistoricalE1cores certified;no universal learned safety/progress claim.",
          "Real vs finiteprecision assumptions;no residualcontext extension;failurecertificate not automatic reachable counterexample;standard results not new theorems.", "high scoped derivations; numerical assumptions explicit")
    claim("C14", "Novelty is specific evidence/mechanisms, not general neural coloring or intervention", "focused primary-source review",
          ["docs/NOVELTY_AND_RELATED_WORK.md", "docs/reviews/REVIEW_4_NOVELTY_EDITORIAL.md", "paper/ucgnn/closure-references.bib"],
          "Focused search through2026-10-26 including Hint-ReLIC,no-hint,2026coloring/generalization.", [], [],
          "Primary precedents delimit integrated empirical contribution;no priority/first or SOTA claim.",
          "Not exhaustive proof of absence;requires external expert judgment;current venue format still pending.", "moderate novelty judgment; primary sources directly inspected")
    registry = dict(date="2026-10-26", base_scientific_commit="71153993e50ec5833cee68e19636e96d802e087a",
                    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
                    claims=claims, checkpoints={r["id"]:{k:r[k] for k in ("path","sha256","source_absolute","source_relative","teacher","architecture","method","cohort","seed","primary","update","budget","parameters")} for r in inventory},
                    native_selected_raw_receipt=reference(OUT / "native-certifications.json"),
                    new_training_updates=0, script_sha256=file_hash(__file__))
    atomic_json(OUT / "evidence-register.json", registry)
    lines = ["# Registro de afirmaciones centrales", "", "La versión estructurada contiene todos los hashes, pesos, semillas y protocolos.", "", "| ID | Afirmación | Estatus | Confianza |", "|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['claim']} | {r['status']} | {r['confidence']} |" for r in claims]
    (Path("docs") / "EVIDENCE_REGISTER.md").write_text("\n".join(lines)+"\n")

    lines = ["| Checkpoint | SHA-256 abreviado | Condición | Éxitos/1000 | Primera familia | Primera acción (paso/fase) | Mecanismo observado |", "|---|---|---|---:|---|---|---|"]
    for r in replay:
        cell = next(c for c in failures if c["stage"] == "confirmatory" and c["checkpoint"] == r["checkpoint"] and c["condition"] == r["condition"])
        mechanism = "propuesta/diversidad; cutoff" if r["teacher"] == "JOHANSSON" else ("propuesta ocupada; rechazo correcto; reparación omitida" if r["teacher"] == "SPECULATIVE_GM" and r["method"] == "counterfactual" else "escritura prohibida durante reparación" if r["teacher"] == "SPECULATIVE_GM" else "confirmación no ganadora/colisión simultánea")
        lines.append(f"| `{r['checkpoint']}` | `{r['sha256'][:16]}` | `{r['condition']}` | {cell['successes']} | {r['family']} | {r['first_action_step']}/{r['first_action_phase']} | {mechanism} |")
    p = Path("docs/GENERALIZATION_FAILURE_REVIEW.md")
    txt = p.read_text().split("<!-- GENERATED_FAILURE_TABLE -->")[0]
    p.write_text(txt+"<!-- GENERATED_FAILURE_TABLE -->\n\n"+"\n".join(lines)+"\n")

    # Preserve bibliographic verification provenance, distinguishing today's
    # direct inspection from inherited verification receipts.
    today = {"velickovic2021nar","velickovic2022clrs","velickovic2020execution","lemos2019coloring","xu2021extrapolate","loukas2020depth","sato2019approximation","mathys2023flood","solodova2025hogwild","ross2011dagger","dehaan2019causal","gretton2012kernel","linial1992locality","kuhn2006complexity","barenboim2008linear","gebremedhin2000scalable","le2026contrastive","vanderbush2026warm","wittig2026which","langedal2025ordering","bevilacqua2023causal","rodionov2023without"}
    bibliography = []
    for p in Path("paper/ucgnn").glob("*references.bib"):
        for entry in re.split(r"(?=@(?:article|misc|inproceedings)\{)", p.read_text()):
            m = re.match(r"@(\w+)\{([^,]+),", entry)
            if not m: continue
            url = re.search(r"url\s*=\s*\{([^}]+)\}", entry)
            year = re.search(r"year\s*=\s*\{([^}]+)\}", entry)
            bibliography.append(dict(key=m[2], year=year[1] if year else None, primary_url=url[1] if url else None,
                                     bibtex_file=str(p), bibtex_sha256=file_hash(p), directly_inspected_closure=m[2] in today,
                                     verification_scope="Primary source inspected2026-10-26" if m[2] in today else "Historical verification receipts; no claim of new full-text review today"))
    atomic_json(OUT / "literature-sources.json", dict(date="2026-10-26", records=bibliography,
                inherited_verification=[reference("paper/ucgnn/"+f) for f in ("bibliography-verification.json","integration-bibliography-verification.json","sample-complexity-bibliography-verification.json")],
                caution="Focused search, no priority claim. Johansson publisher403 today; self-contained proof of implemented variant used."))
    print(f"Registered {len(claims)} central claims and {len(inventory)} checkpoint hashes; fifteen-cell table generated.")


if __name__ == "__main__":
    main()
