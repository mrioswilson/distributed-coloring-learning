# Literature review and novelty delimitation (Pedro revision)

Search and verification date: **30 October 2026**. Scope: `paper/ucgnn/main.tex`
(Introduction, "Relation to previous work"; Appendix "Novelty Comparison"),
`docs/NOVELTY_AND_RELATED_WORK.md` and the four existing `.bib` files. No
existing file was edited. New BibTeX entries are in
`paper/ucgnn/revision-references.bib`.

**Status labels.** VERIFIED means that during this review I opened a primary
record for the work: the arXiv abstract page, arXiv API record or PDF; a
publisher or proceedings page (PMLR, JMLR, ACL Anthology, NeurIPS, icml.cc);
or the Crossref record of its DOI. "Venue per arXiv comment" means the
venue comes only from the authors' arXiv comment field. UNVERIFIED means I
did not open a primary record, or I did not inspect the specific content
detail named. OpenReview returned bot checks throughout, so I used arXiv and
mlanthology for ICLR papers.

**Abbreviations for our contribution claims.**
- **C1**: functional learning of local decision modules across seven
  heterogeneous teachers (21/21).
- **C2**: continuation contracts, the typed teacher oracle with admissibility rules.
- **C3**: per-type continuation diagnostics that separate native agreement
  from continuation fidelity, with label-budget accounting.
- **C4**: frozen transfer, with reproducible safety and progress failure witnesses.
- **C5**: information obstructions (Proposition 1 and the JP representation
  control), plus the finite-type MAX certificate.

---

## 0. Bottom line

1. **Existing citations.** No citation names a wrong paper or wrong authors.
   I found six metadata or description problems:
   - `lemos2019coloring` is not just an arXiv preprint. It was published at ICTAI 2019.
   - `velickovic2021nar` was published in *Patterns*. The intro also cites it
     as a "trace-based reasoning benchmark", but it is a perspective paper.
   - `wittig2026which` is now an ICML 2026 oral paper, not only an arXiv preprint.
   - `langedal2025ordering` lacks its ALENEX pages and DOI.
   - Several teacher entries lack DOIs or pages.
   - Five works the manuscript uses are not cited: DeepSets and Transformer
     (their keys already exist in the bib files), Clopper–Pearson, Adam, and
     the original algorithmic-alignment paper (Xu et al. 2020).

   See Section 1.
2. **Main novelty threats.** None of these does what the paper does as a
   whole, but each one removes a claim that was previously implicit.
   - **Qharabagh et al. 2026** proves exact learnability of *any LOCAL-model
     algorithm* by training node MLPs on local-instruction tables. Those
     tables query the local rule on chosen inputs, including message bits.
   - **Interchange-intervention training and causal distillation** (Geiger et
     al. 2022; Wu et al. 2022) train networks to match a high-level model's
     counterfactual outputs under interventions on aligned intermediate
     variables. They also separate behavioural accuracy from interventional
     accuracy.
   - **Expert queries at synthetic off-trajectory states** come from Galashov
     et al. 2022, DART, and dynamic oracles (Goldberg and Nivre 2012).
   - **SALSA-CLRS** (2023) already trains GNNs on a *randomized distributed
     symmetry-breaking algorithm* (Fast MIS, a Luby-type rule), with the
     randomness supplied as input.
   - **Underspecification and "effective but unfaithful" results**: D'Amour
     et al. 2022, Saldyt and Kambhampati 2025, and the concurrent
     Najafgholizadeh and Afsharchi 2026 ("Fitting every hint is not enough").
   - **Discrete NAR** (Rodionov and Prokhorenkova 2025) proves the correctness
     of learned reasoners by exhaustively unit-testing finite state transitions.
3. **What survives.** I found no prior work that does all of the following:
   imitates *distributed coloring* rules; queries the teacher's
   *confirmation rule under typed interventions on received messages* at
   admissible teacher states; and shows, in paired runs, that perfect native
   agreement coexists with near-chance continuation fidelity. I also found
   none that splits *frozen-transfer failures into safety and progress
   witnesses*. These findings are defensible as **an audited empirical
   instance and an evaluation protocol**. They are not a new learning
   paradigm, a new supervision method, or a new conceptual distinction.
4. The recommended formulation is in Section 5.

---

## 1. Verification of every citation used in `main.tex`

The table covers all 29 keys cited in `main.tex`.

| Key | Where used | Status | Finding | Action |
|---|---|---|---|---|
| `lemos2019coloring` | Intro, App. novelty | VERIFIED (arXiv; Crossref DOI 10.1109/ICTAI.2019.00125) | Title and authors correct. Bib entry is `@misc` arXiv ("Under submission"). The paper is published: *ICTAI 2019*, pp. 879–885. | **Fix entry** (see corrected BibTeX below) |
| `velickovic2020execution` | Intro, App. | VERIFIED (arXiv 1910.10593, "To appear at ICLR 2020") | Correct. The abstract does advocate maximisation-based MPNNs, so "MAX aggregation" is accurate. | Keep |
| `velickovic2022clrs` | Intro, App. | VERIFIED (PMLR 162:22084–22102) | Correct. | Keep |
| `velickovic2021nar` | Intro | VERIFIED (arXiv; Crossref) | Published as an opinion paper: *Patterns* 2(7):100273, 2021, DOI 10.1016/j.patter.2021.100273. **Description error**: the intro cites it among "trace-based reasoning benchmarks", but it is a perspective paper, not a benchmark. | Fix entry; reword the sentence ("…benchmarks \citep{velickovic2022clrs} within the NAR programme \citep{velickovic2021nar}") |
| `ross2011dagger` | Intro, App. | VERIFIED (PMLR 15:627–635) | Correct. | Keep |
| `dehaan2019causal` | Intro, App. | VERIFIED (NeurIPS 32 page) | Correct. The paper proposes targeted interventions (environment interaction or expert queries), so the manuscript's description is accurate. | Keep |
| `loukas2020depth` | Intro, App. | VERIFIED (arXiv 1907.03199, "ICLR 2020"; mlanthology) | Title, author and venue correct. The bib uses OpenReview id `B1l2bp4YwS`, while mlanthology lists `I-0XcXzV_p`. I could not open either because of the OpenReview bot check. | Prefer the arXiv URL |
| `sato2019approximation` | Intro, App. | VERIFIED (arXiv 1905.10261, "Accepted to NeurIPS 2019") | Correct. | Keep |
| `xu2021extrapolate` | Intro, App. | VERIFIED (arXiv 2009.11848; mlanthology ICLR 2021, forum UH-cmocLJC) | Correct. However, "Alignment … affect[s] extrapolation" credits alignment to this paper. Algorithmic alignment originates in Xu et al., ICLR 2020. | Add `xu2020reason` |
| `mathys2023flood` | Intro, App. | VERIFIED (arXiv 2310.06970 v3, 3 Jun 2024; PDF says "Preprint. Under review") | Correct as a preprint. I found no venue. | Keep |
| `solodova2025hogwild` | Intro | VERIFIED (arXiv JR: ICLR 2025) | Correct. The description ("asynchronous robustness requires its own assumptions") is accurate: it concerns implicitly-defined GNNs. | Keep |
| `langedal2025ordering` | Intro, App. | VERIFIED (Crossref DOI 10.1137/1.9781611978339.5) | Correct venue (ALENEX 2025). Missing pp. 56–67, the DOI and the SIAM publisher. The URL points to arXiv. | Add pages and DOI |
| `vanderbush2026warm` | Intro, App. | VERIFIED (arXiv 2601.05137 v1, 8 Jan 2026) | Correct. The description (recursive warm starts, approximate k-coloring) is accurate. | Keep |
| `le2026contrastive` | Intro, App. | VERIFIED (arXiv 2606.03923 v1, 2 Jun 2026; PDF) | Correct. Note that it is *supervised by optimal colorings*, not by teacher algorithms. | Keep |
| `wittig2026which` | Intro, App. | VERIFIED (arXiv v2, 23 Aug 2026; icml.cc/virtual/2026/oral/71194) | Now **ICML 2026 (oral)**. The bib still says arXiv `@misc`. Content: sufficient conditions for MPNNs to learn *deterministic* algorithms (SSSP, MST, DP/knapsack) with size-extrapolation guarantees, plus impossibility results. It does not cover randomized symmetry breaking. | **Update entry**. Reword "go beyond our finite evidence" to "cover deterministic algorithms; randomized symmetry-breaking rules such as ours are outside their hypotheses" |
| `bevilacqua2023causal` | Intro, App. | VERIFIED (PMLR 202:2272–2288) | Correct. The description is accurate: augmentations under which the next step is unchanged, plus an invariance objective. | Keep |
| `rodionov2023without` | Intro, App. | VERIFIED (NeurIPS 36 page; DOI 10.52202/075280-2249) | Correct. The pages 51663–51674 do not appear on the proceedings page (UNVERIFIED). | Add DOI; check pages |
| `luby1993randomness` | App. A | VERIFIED metadata (Crossref: JCSS 47(2):250–286). Content detail UNVERIFIED (paywalled). | Metadata correct, and the paper's main result is a parallel (Δ+1)-coloring algorithm. I did not re-inspect whether its randomized rule uses participation probability 1/2. | Keep; note "Luby-style", which the text already does |
| `johansson1999coloring` | App. A | VERIFIED (Crossref: IPL 70(5):229–232) | Correct. | Keep |
| `schneider2010symmetry` | App. A | VERIFIED (Crossref pp. 257–266, DOI 10.1145/1835698.1835760; PDF) | Correct. The PDF contains "Algorithm ColorTrials", so the teacher name is grounded. Missing pages and DOI. | Add pages and DOI |
| `jones1993parallel` | App. A | VERIFIED (Crossref) | Correct. | Keep |
| `gebremedhin2000scalable` | App. A | VERIFIED (Crossref) | Correct. Missing issue 12 and DOI 10.1002/1096-9128(200010)12:12<1131::AID-CPE528>3.0.CO;2-2. | Add |
| `linial1992locality` | App. A | VERIFIED (Crossref) | Correct. Missing DOI 10.1137/0221015. | Add DOI |
| `kuhn2006complexity` | App. A | VERIFIED (Crossref) | Correct. Missing pp. 7–15 and DOI 10.1145/1146381.1146387. | Add |
| `barenboim2008linear` | App. A | VERIFIED (arXiv v2 PDF; Crossref STOC 2009) | The arXiv citation is correct. The refereed versions are STOC 2009 (pp. 111–120), and SICOMP 43(1):72–95, 2014 with Kuhn added. | Also cite `barenboim2009distributed` (new) |
| `gretton2012kernel` | App. laws | VERIFIED (JMLR 13(25):723–773) | Correct. | Keep |
| `hemerik2018exact` | App. laws | VERIFIED (Crossref) | Correct. Issue 4 is missing. | Optional |
| `phipson2010permutation` | App. laws | VERIFIED (Crossref) | Correct. | Keep |
| `holm1979simple` | App. laws | UNVERIFIED this pass (no DOI; JSTOR not opened) | Unchanged from the prior audit. | Keep |

**Uses without a citation.**
- The **DeepSets** and **LocalPaletteTransformer** architectures (§3, "Shared
  communication") should cite `zaheer2017sets` and `vaswani2017attention`.
  Both keys already exist in `references.bib`.
- The **Clopper–Pearson** limits (§4) need a citation: `clopper1934use` (new).
- **Adam** (§5) needs a citation: `kingma2015adam` (new).

**Corrected BibTeX for existing keys.** These are not added to the new
file, to avoid duplicate keys. Replace the entries in place when an editor
is authorized.

```bibtex
@inproceedings{lemos2019coloring,
  title={Graph Colouring Meets Deep Learning: Effective Graph Neural Network Models for Combinatorial Problems},
  author={Lemos, Henrique and Prates, Marcelo and Avelar, Pedro and Lamb, Luis},
  booktitle={2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)},
  pages={879--885}, year={2019}, doi={10.1109/ICTAI.2019.00125}}
@article{velickovic2021nar,
  title={Neural Algorithmic Reasoning}, author={Veli{\v{c}}kovi{\'c}, Petar and Blundell, Charles},
  journal={Patterns}, volume={2}, number={7}, pages={100273}, year={2021},
  doi={10.1016/j.patter.2021.100273}}
@inproceedings{wittig2026which,
  title={Which Algorithms Can Graph Neural Networks Learn?},
  author={Wittig, Solveig and Vasileiou, Antonis and Nerem, Robert R. and Stoll, Timo and Geerts, Floris and Wang, Yusu and Morris, Christopher},
  booktitle={International Conference on Machine Learning (ICML)}, year={2026},
  note={Oral; arXiv:2602.13106v2}, url={https://arxiv.org/abs/2602.13106v2}}
@inproceedings{langedal2025ordering,
  title={Graph Neural Networks as Ordering Heuristics for Parallel Graph Coloring},
  author={Langedal, Kenneth and Manne, Fredrik},
  booktitle={2025 Proceedings of the Symposium on Algorithm Engineering and Experiments (ALENEX)},
  pages={56--67}, publisher={SIAM}, year={2025}, doi={10.1137/1.9781611978339.5}}
@inproceedings{zhao2020scheduling,  % only if cited; the key is kept for stability
  title={Distributed Scheduling Using Graph Neural Networks},
  author={Zhao, Zhongyuan and Verma, Gunjan and Rao, Chirag and Swami, Ananthram and Segarra, Santiago},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4720--4724}, year={2021}, doi={10.1109/ICASSP39728.2021.9414098}}
```

Add DOIs or pages to `kuhn2006complexity`, `schneider2010symmetry`,
`linial1992locality`, `gebremedhin2000scalable` and `rodionov2023without`
as listed in the table.

**Existing keys to keep.**
- Keep all 29 cited keys (with the corrections above).
- Among uncited existing keys:
  - **Start citing** `zaheer2017sets`, `vaswani2017attention` and
    `zhao2020scheduling` (learned module inside a distributed greedy MWIS
    solver; same pattern as Langedal–Manne).
  - Keep `suomela2013local` if the paper defines the LOCAL model explicitly.
  - Keep `ross2010efficient` and `rajaraman2020limits` only if the
    imitation-learning paragraph discusses compounding error.
  - `zhou2023algorithms`, `kazemnejad2023position`, `schulman2017ppo`,
    `schrab2023mmdagg` and `biggs2023mmdfuse` are not needed by the current
    narrative. They are harmless if uncited, since natbib ignores them.

---

## 2. Relevant literature, by theme

For each work: citation and status; one-sentence content; what it already
establishes relative to C1–C5.

### 2.1 Neural algorithmic reasoning (NAR): execution, benchmarks, alignment

- **Veličković et al., ICLR 2020, *Neural Execution of Graph Algorithms*** (existing; VERIFIED).
  - *Content.* Trains MPNNs on step-wise traces of graph algorithms and
    argues for max aggregation.
  - *Establishes.* Imitating algorithm steps with MAX-MPNNs (background to
    C1). Our use of MAX is not novel.
- **Veličković et al., ICML 2022, CLRS** (existing; VERIFIED); **Ibarz et al., LoG 2022, *A Generalist Neural Algorithmic Learner*, PMLR 198:2:1–2:23** (VERIFIED).
  - *Content.* 30 algorithms with hint traces; one shared processor learns
    all 30.
  - *Establishes.* "Many heterogeneous teachers, one shared core" is
    standard (C1). Ibarz et al. also analyse **teacher forcing of hints and
    remove it** (PDF, paragraph "Removing teacher forcing"). Our *matched* arm, which evaluates COMMIT on teacher
    messages, is teacher forcing applied to messages. The *passive* arm is
    free-running.
- **Xu et al., ICLR 2020, *What Can Neural Networks Reason About?*** (VERIFIED: arXiv 1905.13211; ICLR 2020 per mlanthology and the iclr.cc listing).
  - *Content.* Introduces algorithmic alignment and sample-complexity
    arguments for GNNs on dynamic-programming tasks.
  - *Establishes.* That architecture–algorithm alignment matters (C1, C4). It
    is the correct citation for "alignment".
- **Minder, Grötschla, Mathys, Wattenhofer 2023, *SALSA-CLRS*** (VERIFIED: arXiv 2309.12253; LoG 2023 extended abstract per arXiv comment; PDF).
  - *Content.* A sparse, scalable CLRS extension that adds **distributed
    algorithms**: *Fast MIS 2*, where each node draws a random value and
    joins the MIS if it is a local minimum, and distributed eccentricity. The
    randomness is precomputed and given as input; runs are tested up to 100×
    the training size, with and without hints.
  - *Establishes.* **Direct precedent for C1.** It learns a randomized
    distributed symmetry-breaking rule (Luby-type MIS) by GNN imitation, with
    supplied randomness, and tests size transfer. It covers no coloring rule,
    no confirmation or veto stage, no continuation oracle and no failure
    mechanisms. **Must be cited.**
- **Engelmayer, Georgiev, Veličković, LoG 2023, PMLR 231:31:1–31:13** (VERIFIED).
  - *Content.* Parallel versions of algorithms align better with neural
    execution than sequential ones.
  - *Establishes.* That parallel and distributed formulations suit MPNNs
    (C1). Useful framing, not a threat.
- **Mahdavi et al., TMLR 2023** (VERIFIED; venue per arXiv comment).
  - *Content.* Out-of-distribution generalization in NAR, and the role of hints.
  - *Establishes.* Size and OOD issues in NAR (C4).
- **Mirjanić, Pascanu, Veličković, LoG 2023, PMLR 231:10:1–10:24** (VERIFIED).
  - *Content.* Identifies latent-space failure modes: loss of resolution
    between similar values, and inability to handle values outside the
    training range.
  - *Establishes.* **Mechanistic precedent for our CT "scale sensitivity of
    preference comparison" witness and the JP numeric-ID control** (C4, C5).
    Our witnesses are concrete instances of these modes, not a new
    phenomenon class.
- **Bevilacqua et al., ICML 2023, Hint-ReLIC** (existing; VERIFIED).
  - *Content.* Builds augmentations whose next execution step is unchanged,
    and enforces invariance of the next-step prediction.
  - *Establishes.* Interventions on algorithmic inputs in NAR (C2). The
    difference stated in `main.tex` is correct and should stay: our targets
    *change* under intervention, and we acquire explicit oracle labels.
- **Rodionov and Prokhorenkova, NeurIPS 2023, NAR without intermediate supervision** (existing; VERIFIED).
  - *Content.* Input–output training is competitive with hint supervision.
  - *Establishes.* Trajectory imitation is neither necessary nor equivalent
    to output success (C3 background).
- **Rodionov and Prokhorenkova, ICML 2025, *Discrete Neural Algorithmic Reasoning*, PMLR 267:51851–51865** (VERIFIED; PDF).
  - *Content.* Forces execution through finite predefined states. Because
    states are finite, "we can unit-test all possible state transitions" and
    prove that the learned model is correct for any test input.
  - *Establishes.* **Precedent for our finite-type MAX certificate (C5).**
    Exhaustive verification of a learned finite-type local computation is
    known. Our certificate must be presented as an instance and a diagnostic.
    Note that 45 of our selected transfer cores *fail* it.
- **Kujawa et al. 2024, *NAR with Multiple Correct Solutions*** (VERIFIED; arXiv preprint v4, May 2025).
  - *Content.* NAR methods that sample several valid outputs, demonstrated
    on Bellman-Ford and DFS.
  - *Establishes.* That "one valid output" is too weak a target when several
    solutions exist. This is relevant to the output-law appendix, since
    coloring has many valid outputs. It is not a threat.
- **Najafgholizadeh and Afsharchi 2026, *Fitting Every Hint Is Not Enough: Premature Floods in Recurrent Algorithmic Reasoners*** (VERIFIED: arXiv 2610.14982 v1, 10 Oct 2026; PDF).
  - *Content.* On SALSA-CLRS BFS, a reasoner that fits every training hint
    still fails on deep graphs. The failure is driven by a degree input that
    falls in the tail of the training range. They locate the mechanism, test
    interventions (clamping the input, adding dense graphs, detaching the
    state), and recommend auditing input ranges.
  - *Establishes.* **Closest recent analogue for C3 and C4, and concurrent.**
    Hint or trace agreement does not identify the rule, and transfer
    failures have reproducible, range-driven mechanisms. They do not
    intervene on messages or query a teacher off-trajectory. **Cite as
    concurrent work.**
- **Saldyt and Kambhampati 2025, *Mind the Gap*** (VERIFIED: arXiv 2505.18623).
  - *Content.* Uses neural compilation to compare learned GNN parameters and
    intermediate vectors with an exact compiled solution, asking "how
    faithful are learned algorithms when they are effective?" for BFS, DFS
    and Bellman-Ford.
  - *Establishes.* **The separation of effectiveness (functional success)
    from faithfulness to the intended algorithm (threat (b)).** Their test is
    mechanistic (parameters and internals). Ours is behavioural, at a
    declared communication interface.
- **He et al. 2026, MINAR** (VERIFIED: arXiv 2602.21442).
  - *Content.* Circuit discovery by attribution patching in NAR GNNs; it
    finds shortcut learning in a GNN trained on Bellman-Ford together with BFS.
  - *Establishes.* Intervention-based checks that a NAR model executes the
    intended algorithm are an active line of work (threat (b)). These are
    internal patches, not teacher-labelled input interventions.
- *Considered, not recommended for citation:*
  - **CLRS-Text** (Markeeva et al. 2024, arXiv 2406.04229; VERIFIED). LLM
    text benchmark; only relevant if the paper discusses LLM algorithm learning.
  - **Huang et al., AAAI 2026, auxiliary reconstruction** (arXiv 2606.00559; VERIFIED). Encoder improvements; not relevant.
  - **He and Vitercik, ICML 2025, primal–dual NAR** (arXiv 2505.24067; VERIFIED). Approximation algorithms; not relevant.

### 2.2 Learning local and distributed algorithms; locality and anonymity theory

- **Qharabagh, Back de Luca, Giapitzakis, Fountoulakis 2026, *Learning to Execute Graph Algorithms Exactly with Graph Neural Networks*** (VERIFIED: arXiv 2601.23207 v2, 5 Jun 2026; PDF).
  - *Content.*
    - Trains an ensemble of node MLPs on binary *local-instruction* tables,
      with separate computation and message blocks, then uses them as the
      GNN update.
    - Uses NTK theory to prove exact, high-probability execution with
      bounded degree and finite precision.
    - Gives a learnability theorem for **any LOCAL-model algorithm**, plus
      flooding, BFS, DFS and Bellman-Ford.
    - Global IDs bound the admissible graph size.
  - *Establishes.* **The strongest threat to the "learning distributed
    algorithms" framing (C1) and to the supervision idea in C2.**
    - Exact learnability of LOCAL algorithms is now a theorem, under strong
      assumptions: infinite-width NTK, an instruction encoding, and ensembles.
    - Its supervision is a table of the local rule queried on *chosen inputs*
      (including message bits), not traces. That is the theoretical
      counterpart of our counterfactual bank.
    - It does not treat randomized symmetry breaking with private draws,
      learning from native traces, finite-width SGD, or empirical
      continuation gaps.
  - *Use it positively.* Our BE result is empirical evidence that
    *trace-only* supervision underdetermines the local rule, whereas
    supervision covering message configurations (as instruction-style
    theory requires) closes the gap. Say "consistent with", not "confirms".
- **Loukas, ICLR 2020** and **Sato, Yamada, Kashima, NeurIPS 2019** (existing; VERIFIED).
  - *Establishes.* GNN–LOCAL correspondence and lower bounds (C5 background).
- **Sato, Yamada, Kashima, SDM 2021, *Random Features Strengthen GNNs*, pp. 333–341** (VERIFIED) and **Abboud et al., IJCAI 2021, pp. 2112–2118** (VERIFIED).
  - *Content.* Random node features break symmetry and increase expressivity.
  - *Establishes.* That private randomness acts as a symmetry breaker in GNNs
    (C5). This is context for the anonymous Luby and Johansson teachers,
    which receive private draws.
- **Hella et al., *Distributed Computing* 28(1):31–53, 2015, *Weak models of distributed computing*** (VERIFIED) and **Angluin, STOC 1980, *Local and global properties in networks of processors*, pp. 82–93** (VERIFIED).
  - *Establishes.* Classical indistinguishability and anonymity obstructions.
    **Proposition 1 is an elementary instance of these** (C5) and should be
    labelled as such, with these citations.
- **Tolstaya et al., CoRL 2019 (PMLR 100:671–682, 2020)** (VERIFIED) and **Gama et al., IEEE TSP 70:1932–1946, 2022** (VERIFIED).
  - *Content.* GNN decentralized controllers trained by imitation of
    centralized experts, which then transfer to larger teams.
  - *Establishes.* Learned communicating local policies trained by imitation
    and size transfer (C1, C4). These are the closest "learned distributed policy"
    precedent outside NAR.
- **Zhao et al., ICASSP 2021, *Distributed Scheduling Using GNNs*** (existing key `zhao2020scheduling`; VERIFIED via Crossref).
  - *Content.* A GCN supplies weights to a distributed greedy MWIS solver.
  - *Establishes.* "Learned module inside a fixed distributed algorithmic
    skeleton" (C1, interpreter design). The same pattern appears in
    `langedal2025ordering`.
- **Parkar, Chaturvedi, Daymude 2025, *Unsupervised Learning of Local Updates for MIS in Dynamic Graphs*** (VERIFIED: arXiv 2505.13754 v3; IJCNN 2026 per arXiv comment).
  - *Content.* Learned single-step distributed local update rules for MIS,
    tested on graphs up to 100× larger.
  - *Establishes.* Learned local symmetry-breaking rules with size transfer
    are being studied, though without a teacher (C1, C4).
- **Solodova et al., ICLR 2025** (existing; VERIFIED). Asynchronous robustness of implicit GNNs. It confirms that asynchrony is out of scope.

### 2.3 Neural graph coloring

- **Lemos et al., ICTAI 2019** (existing; fix entry).
  - *Content.* GNN decision and coloring model.
  - *Establishes.* Neural coloring predates us (C1).
- **Schuetz, Brubaker, Zhu, Katzgraber, *Phys. Rev. Research* 4:043131, 2022** (VERIFIED).
  - *Content.* Unsupervised Potts-model GNN coloring that scales to millions
    of nodes.
  - *Establishes.* Large-scale GNN coloring without a teacher. Not a threat.
- **Li et al. 2022, *Rethinking GNNs for the Graph Coloring Problem*** (VERIFIED: arXiv 2208.06975).
  - *Content.* Shows that aggregation-combine GNNs are *local coloring
    methods* and identifies node pairs they cannot discriminate.
  - *Establishes.* The locality and expressivity limits of GNN coloring (C5).
- **Langedal and Manne, ALENEX 2025** (existing).
  - *Content.* A GNN learns vertex orderings for a parallel greedy (JP-like)
    coloring.
  - *Establishes.* Learned priorities inside a parallel coloring skeleton (C1).
- **Vanderbush and Weber 2026** and **Le, Zhao, Weber 2026** (existing; VERIFIED).
  - *Content.* Approximate k-coloring with warm starts; supervised
    contrastive coloring geometry trained on optimal colorings.
  - *Establishes.* Neither imitates distributed teachers. Not threats.
- *Optional context, not in the new bib:*
  - **Watkins, Montana, Branke, LION 2023, LNCS pp. 491–505** (VERIFIED via Crossref). Deep-Q construction heuristic.
  - **Colantonio et al. 2024–2026** (VERIFIED: arXiv 2408.01503 v3, updated 20 Oct 2026). Planted supervision with noise-annealed dynamics.
  - **Tan, Sim, Chua, "A survey of modern neural network based methods for the graph coloring problem", *Neural Networks* 205:109320** (metadata VERIFIED via Crossref; **content UNVERIFIED**). Read it before citing, because it may contain a coloring-specific related-work section that reviewers will expect.

### 2.4 Imitation learning, oracles, and off-trajectory or interventional supervision

- **Ross et al., AISTATS 2011 (DAgger)** (existing). Learner-induced state distributions.
- **Daumé, Langford, Marcu, *Machine Learning* 75(3):297–325, 2009 (SEARN)** (VERIFIED) and **Chang et al., ICML 2015 (LOLS), PMLR 37:2058–2066** (VERIFIED).
  - *Content.* Learning to search: costs of one-step deviations are evaluated
    with reference-policy roll-outs.
  - *Establishes.* That querying a reference policy on *continuations after
    deviations* is a classical supervision device (C2).
- **Goldberg and Nivre, COLING 2012, *A Dynamic Oracle for Arc-Eager Dependency Parsing*, pp. 959–976** (VERIFIED).
  - *Content.* An oracle that returns the optimal transitions for *every*
    parser configuration, including ones off the gold path. A static oracle
    defines labels only on gold trajectories.
  - *Establishes.* **The static/dynamic oracle distinction is our
    native/continuation distinction** (C2). Our "continuation contract" is a
    dynamic oracle restricted to typed message interventions at admissible
    teacher states.
- **Laskey et al., CoRL 2017 (DART), PMLR 78:143–156** (VERIFIED).
  - *Content.* Injects noise into expert demonstrations so that the expert
    labels recovery behaviour, as an off-policy alternative to DAgger.
  - *Establishes.* Expert labels on perturbed states, collected without
    learner roll-outs (C2).
- **Galashov, Merel, Heess, NeurIPS 2022, *Data Augmentation for Efficient Learning from Parametric Experts*** (VERIFIED: NeurIPS page; arXiv 2205.11448).
  - *Content.* Policy cloning with *queries of the expert at synthetic states
    around sampled trajectories* (APC).
  - *Establishes.* **The closest general-IL precedent for our fixed-bank
    counterfactual supervision (threat (a), general form)** (C2). They query
    a parametric expert at synthetic perturbations of expert states, offline,
    without DAgger. Our domain-specific content is that the perturbations are
    typed interventions on received messages in a two-exchange distributed
    protocol, with admissibility constraints, and that the expert is an exact
    algorithm.
- **de Haan et al., NeurIPS 2019** (existing) and **Spencer et al. 2021, *Feedback in Imitation Learning: The Three Regimes of Covariate Shift*** (VERIFIED: arXiv 2102.02872).
  - *Content.* Spencer et al. argue that "causal confusion" is covariate
    shift driven by feedback between decisions and input features.
  - *Establishes.* **Our passive arm is exactly such a feedback loop.** Its
    COMMIT loss is evaluated on the model's own messages against native
    labels (C3 interpretation). Cite it when explaining why passive and
    matched training differ.
- **Williams and Zipser, *Neural Computation* 1(2):270–280, 1989** (VERIFIED) and **Bengio et al., NeurIPS 2015, *Scheduled Sampling*** (VERIFIED).
  - *Establishes.* Teacher forcing versus free running. Our matched and
    passive arms are message-level instances of these (C3). Name them as
    such to avoid inventing terminology.
- **Angluin, *Machine Learning* 2(4):319–342, 1988, *Queries and Concept Learning*** (VERIFIED).
  - *Establishes.* Learning from membership queries at chosen inputs. This
    is the generic ancestor of a "typed oracle" (C2).
- **Kaushik, Hovy, Lipton, ICLR 2020** (VERIFIED; venue per arXiv comment) and **Huang, Liu, Bowman, Insights workshop 2020, pp. 82–87** (VERIFIED).
  - *Content.* Counterfactually augmented data relabels minimally edited
    inputs. When dataset size is controlled, the gains largely disappear
    (Huang et al.; Khashabi et al. 2020 as summarized there).
  - *Establishes.* **Why our missing label-matched control matters** (C3).
    Counterfactual relabelling looked beneficial in NLP until training size
    was equalized. The manuscript already declines efficiency claims; cite
    this to show the concession is principled.
- *Considered, not recommended:*
  - Pitis et al., NeurIPS 2020, counterfactual data augmentation with locally factored dynamics (VERIFIED). It generates transitions from model structure, not from teacher queries.
  - Beyer et al., CVPR 2022, "function matching" distillation (VERIFIED). A teacher queried on augmented inputs; an analogy only.
  - D'urso et al., RSS 2026 workshop, counterfactual nuisance behaviour cloning (VERIFIED: arXiv 2607.27261). Workshop paper on visual nuisances.

### 2.5 Fidelity under interventions versus functional correctness

- **Geiger et al., ICML 2022, *Inducing Causal Structure for Interpretable Neural Networks* (IIT), PMLR 162:7324–7338** (VERIFIED).
  - *Content.* Aligns high-level causal-model variables with network
    representations, and trains the network to match the causal model's
    **counterfactual outputs under interchange interventions**. A zero loss
    makes the high-level model a causal abstraction of the network.
  - *Establishes.* **Training a network to reproduce an algorithm's outputs
    under interventions on intermediate variables (threat (a), conceptual
    form)** (C2, C3). Our CF arm can be described as IIT with a *fixed,
    prescribed alignment*. The intervened variables are the communicated
    messages P and C, which are interface variables of both teacher and
    student by construction, and the interventions are typed families
    rather than values from a second input. This framing is honest and
    strengthens the paper. It also means "counterfactual supervision" must
    not be claimed as new.
- **Wu et al., NAACL 2022, *Causal Distillation for Language Models*, pp. 4288–4295** (VERIFIED).
  - *Content.* A student matches the teacher under interchange interventions (DIITO).
  - *Establishes.* Teacher–student training under interventions. The teacher
    here is a network, not an algorithm (C2).
- **Geiger et al., JMLR 26(83):1–64, 2025, *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*** (VERIFIED) and **Sutter et al., NeurIPS 2025, *The Non-Linear Representation Dilemma*** (VERIFIED; venue per arXiv comment).
  - *Content.* Geiger et al. give the general theory and interchange-
    intervention accuracy (IIA) as a graded faithfulness measure. Sutter et
    al. prove that with unrestricted alignment maps any network can be
    mapped to any algorithm, so causal abstraction can become trivial.
  - *Establishes.* (b) is a recognized theme: behavioural accuracy versus
    interventional faithfulness. Our evaluation sidesteps Sutter et al.'s
    objection *because* the alignment is fixed at the communication
    interface. That is a real, statable advantage of interface-level
    continuation tests. It is also a limitation: we say nothing about
    internal mechanism.
- **Lowe et al., AAMAS 2019, *On the Pitfalls of Measuring Emergent Communication*, pp. 693–701** (VERIFIED).
  - *Content.* Agents whose messages predict their actions may not actually
    *use* incoming messages. The proposed causal influence of communication
    intervenes on messages and measures the change in the listener's action
    distribution.
  - *Establishes.* **Intervening on messages to evaluate learned
    communicating agents (threat (b), multi-agent form)** (C3). Our
    diagnostic adds a teacher-defined *correct* response, so it measures
    fidelity, not mere influence.
- **D'Amour et al., JMLR 23(226):1–61, 2022 (underspecification)** (VERIFIED), **McCoy, Min, Linzen, BlackboxNLP 2020, pp. 217–227** (VERIFIED) and **Zhong et al., NeurIPS 2023, *The Clock and the Pizza*** (VERIFIED; venue per arXiv comment).
  - *Content.* Predictors that are equivalent on held-out data diverge under
    stress tests and across seeds. Networks with the same function can
    implement different algorithms.
  - *Establishes.* **The general phenomenon behind our BE result.** Native
    agreement leaving COMMIT underdetermined is an instance of
    underspecification. The BE passive-arm seed range (9.16–42.33% continuation
    accuracy, with identical native agreement) is itself a McCoy-style
    seed-variability observation (C3).
- **Gardner et al., Findings EMNLP 2020 (contrast sets), pp. 1307–1323** (VERIFIED) and **Ribeiro et al., ACL 2020 (CheckList), pp. 4902–4912** (VERIFIED).
  - *Content.* Evaluation by typed, label-changing perturbations, beyond
    aggregate accuracy.
  - *Establishes.* That evaluation with typed perturbations whose labels
    change is established practice (C3). Our per-type continuation
    diagnostics are the distributed-algorithm analogue.

### 2.6 Size generalization and provable extrapolation

- **Xu et al., ICLR 2021** (existing) and **Yehudai et al., ICML 2021, PMLR 139:11975–11986** (VERIFIED).
  - *Content.* Yehudai et al. show that size generalization fails when local
    structures (neighbourhood patterns) differ between small and large graphs.
  - *Establishes.* **Degree and palette transfer is a local-structure shift**
    (C4). Our confounding of degree with K and preference scale fits this
    account and should cite it.
- **Nerem, Chen, Dasgupta, Wang, COLT 2026, PMLR 336:5273–5331** (VERIFIED).
  - *Content.* Sparsity-regularized GNNs provably extrapolate for shortest
    paths by implementing Bellman-Ford steps.
  - *Establishes.* Provable extrapolation for one deterministic algorithm (C4).
- **Wittig et al., ICML 2026 (oral)** (existing; update). General sufficient
  conditions and impossibility results for deterministic algorithms (C4).
- **Qharabagh et al. 2026** (Section 2.2). Exact execution on unbounded sizes
  under bounded degree and finite precision (C4).
- *Takeaway for C4.* Theory now gives **guarantees for deterministic
  algorithms and for LOCAL algorithms under instruction training**. Our
  frozen-transfer evidence is finite and empirical, for randomized,
  scheduled and repair-based coloring rules learned from traces. Do not
  frame it as competing with these results. Frame it as documenting *how*
  transfer fails when such hypotheses are absent.

### 2.7 Program induction with interpreters; verifying learned modules

- **Reed and de Freitas, ICLR 2016, *Neural Programmer-Interpreters*** (VERIFIED arXiv; ICLR 2016 per mlanthology listing and Google blog).
  - *Content.* Learns subroutines from execution traces inside an
    environment or interpreter.
  - *Establishes.* **"Learned modules plus supplied interpreter, trained from
    teacher traces" is an established design** (C1).
- **Cai, Shin, Song, ICLR 2017, *Making Neural Programming Architectures Generalize via Recursion*** (VERIFIED; venue per arXiv comment).
  - *Content.* Recursion shrinks each module's domain enough to *prove*
    generalization by checking a finite verification set.
  - *Establishes.* Finite verification of learned modules within an
    interpreter. This is a precedent for the finite-type certificate (C5).

---

## 3. Threat analysis

### (a) Prior work that queries a teacher on alternative inputs or messages to supervise or evaluate a learned rule

| Work | What it does | Overlap with our CF arm | Remaining difference |
|---|---|---|---|
| Qharabagh et al. 2026 | Trains local MLPs on instruction tables of a LOCAL rule, including message bits | High in principle: supervision of the local rule on chosen message inputs | Theory (NTK, ensembles, binary encodings). Deterministic LOCAL rules. No traces, no randomized coloring, no empirical passive/matched/CF contrast |
| Geiger et al. 2022 (IIT); Wu et al. 2022 | Train to match a high-level model's counterfactual outputs under interventions on aligned variables | High conceptually: counterfactual targets that change under intervention | Ours fixes the alignment at the message interface. The teacher is an exact distributed algorithm. Typed families with admissibility. Per-type fidelity reporting |
| Galashov et al. 2022 (APC); Laskey et al. 2017 (DART) | Query the expert at synthetic or perturbed states near demonstrations, offline | High procedurally: a fixed bank of expert-labelled off-trajectory queries | Interventions are on received messages in a multi-node protocol. Silence and alternative-proposal types. Legality constraints |
| Goldberg and Nivre 2012; SEARN; LOLS | Oracles defined off the gold path; reference roll-outs after deviations | Conceptual: dynamic-oracle labels | Ours is not learner-state collection (as `main.tex` says). One-step confirmation queries |
| Hint-ReLIC 2023 | Augmentations under which the next step is unchanged; invariance objective | Interventions on algorithmic inputs in NAR | Their targets are invariant by construction. Ours change and are labelled by the oracle |
| Lowe et al. 2019 | Intervenes on messages to measure their influence on the listener | Evaluation by message intervention | No teacher-correct response. Influence, not fidelity |

**Verdict on (a).** The *idea* of supervising or evaluating a learned local
rule with teacher labels at off-trajectory or intervened inputs is
established in four separate literatures: dynamic oracles, expert
augmentation in imitation learning, causal abstraction, and instruction-based
exact learning. I found **no work** that instantiates it for *distributed
coloring teachers*, with *typed interventions on received messages* at
*legal teacher states*, and compares it in pairs against passive and
teacher-forced (matched) training. The defensible novelty is this
instantiation and its measured consequences, not the supervision device.

### (b) Prior work that separates functional correctness from fidelity under interventions

| Work | Separation shown |
|---|---|
| D'Amour et al. 2022; McCoy et al. 2020 | Equal held-out performance, divergent stress-test behaviour |
| Geiger et al. 2021/2022/2025 | Behavioural accuracy versus interchange-intervention accuracy |
| Saldyt and Kambhampati 2025 | Effective versus faithful learned GNN algorithms (BFS, DFS, Bellman-Ford) |
| Rodionov and Prokhorenkova 2023 | Output success without trajectory imitation |
| Najafgholizadeh and Afsharchi 2026 (concurrent) | All training hints fitted, rule still wrong out of distribution |
| Zhong et al. 2023 | Same function, different algorithms |
| Lowe et al. 2019 | Apparent communication without listener use |

**Verdict on (b).** The *conceptual* separation is well established and must
not be claimed. What remains specific is:
1. The separation is measured at a **declared distributed interface**
   (messages), where the alignment problem of causal abstraction does not
   arise.
2. The gap is **decomposed into semantic components**. The eligibility gate
   is redundant on native runs; occupied-option rejection is learned only
   with CF. The eligibility-only baseline already reaches 99.28% balanced
   accuracy (BA), so most of the raw gap is a rule that native executions
   never exercise.
3. There is an **overlap audit of sufficient input types**: 99.28–99.54% of
   diagnostic occurrences have training-seen sufficient types.

These points make the result an unusually well-controlled *instance* of
underspecification. That is a fair contribution claim.

### Other threats by contribution

- **C1.** Precedents: CLRS and the generalist learner (many teachers, one
  core); SALSA-CLRS (distributed randomized MIS); Qharabagh et al. (LOCAL
  learnability); NPI (interpreter plus learned modules); Langedal–Manne and
  Zhao et al. (learned modules in distributed skeletons); Tolstaya and Gama
  (imitation of decentralized experts). What remains: the specific
  distributed coloring rules, declared interpreter boundaries, and complete
  certification provenance. This is **a coverage and engineering result,
  not a novelty claim**.
- **C4.** Precedents: Yehudai et al. (local-structure shift); Mirjanić et al.
  (out-of-range latent failure); Najafgholizadeh and Afsharchi (range-driven
  mechanism, replayed); theory for deterministic algorithms (Nerem, Wittig,
  Qharabagh). What remains: the **safety-versus-progress taxonomy** of
  transfer failures. Examples are a correct veto with collapsed proposal
  diversity (progress) versus an omitted scheduled repair or forbidden write
  (safety), with replayed first-failure batches. I did not find this
  distributed-computing lens applied to learned-rule transfer in NAR.
- **C5.** Precedents: Angluin 1980 and Hella et al. 2015 (indistinguishability);
  Sato 2021 and Abboud 2021 (randomness as a symmetry breaker); Rodionov 2025
  and Cai 2017 (finite exhaustive verification). **Nothing new remains** beyond
  the concrete JP control. Demote it to supporting material.

---

## 4. Contribution verdicts

| Claim (current framing) | Verdict | Required citations | Recommended wording |
|---|---|---|---|
| C1 "Functional module learning across heterogeneous teachers; 21/21" | **Keep as a supporting result; reword.** Not a novelty claim. | `minder2023salsa`, `ibarz2022generalist`, `qharabagh2026exact`, `reed2016neural`, `zhao2020scheduling`, `tolstaya2020learning` | "Within declared interpreters that supply schedules, arithmetic and argmax, small MAX-MPNN modules can be trained to reproduce the learned decisions of seven coloring rules; each of 105 selected checkpoints passes 1,000 native runs." |
| C2 "Continuation contracts / typed oracle" | **Keep as specification; reword. Drop any implication of a new supervision method.** | `goldberg2012dynamic`, `galashov2022augmentation`, `laskey2017dart`, `geiger2022inducing`, `angluin1988queries`, `bevilacqua2023causal` | "We specify, for each teacher, a dynamic-oracle-style continuation contract: which message interventions are admissible at a legal teacher state, and the teacher's confirmation response. Training on such labels is an instance of expert queries at synthetic states (APC) or of interchange-intervention training with a fixed interface alignment." |
| C3 "Per-type continuation diagnostics with label-budget controls" | **Keep: the central empirical result. Reword "controls" to "accounting".** Drop efficiency and causal-effect language (already mostly done). | `damour2022underspecification`, `saldyt2025mind`, `lowe2019pitfalls`, `gardner2020contrast`, `huang2020counterfactually`, `williams1989learning`, `bengio2015scheduled`, `spencer2021feedback`, `najafgholizadeh2026premature` (concurrent) | "Perfect native agreement coexists with near-chance continuation BA (60.1% passive, 53.3% matched) versus 99.997% with CF supervision (more labels). The gap lies in rules that native runs never exercise." |
| C4 "Frozen transfer with reproducible safety/progress failure witnesses" | **Keep; reword around the safety/progress taxonomy.** No "first" claims. | `yehudai2021size`, `mirjanic2024latent`, `najafgholizadeh2026premature`, `nerem2026extrapolate`, `wittig2026which`, `qharabagh2026exact` | "Transfer mostly preserves functional success. The 15 failing cells replay to distinct mechanisms that aggregate success conflates: progress failures under correct vetoes, safety failures from omitted repairs or wrong confirmations, and scale-sensitive preference comparison." |
| C5 "Information obstructions" and finite-type MAX certificate | **Demote to remarks or appendix.** Do not list them as contributions. | `angluin1980local`, `hella2015weak`, `sato2021random`, `abboud2021surprising`, `rodionov2025discrete`, `cai2017making` | "Proposition 1 is the standard indistinguishability argument applied to our interface. The finite-type certificate is a sufficient check in the spirit of discrete NAR; selected transfer cores fail it despite successful rollouts." |
| Thesis "functional success, continuation fidelity and generalization are distinct dimensions" | **Reword.** As a general statement it is known (underspecification; IIA versus accuracy; effective versus faithful). | as for C3 | "For learned distributed rules these dimensions separate *empirically and in specific, replayable ways*, so each must be reported." |
| Title term "Counterfactual Fidelity" | **Keep only with a definition**, or rename to "continuation fidelity". | `geiger2022inducing` | Define it as the teacher's response under do(message), with state and private draws held fixed. That makes it a unit-level counterfactual of a deterministic local rule. Note the different usage in the counterfactual-data-augmentation literature. |
| "Our specific increment is the combination of …" (Intro, l. 96–98) | **Drop.** "Combination" claims read as weak to reviewers. | — | Replace with the formulation in Section 5. |

**Do not claim:** that intervening on algorithmic inputs or messages is new;
that teacher-labelled off-trajectory supervision is new; that "functional
success does not imply fidelity" is new; first neural coloring; first
learning of distributed algorithms; first size or degree transfer; any
learnability guarantee; any label-efficiency or causal effect of CF; any
asynchronous or deployment robustness.

---

## 5. Recommended "what does the reader learn" formulation

> Prior work has shown that neural networks can be trained to execute local
> and distributed algorithms, and that task accuracy, or even trace
> agreement, need not reveal which rule was learned. We show how these
> notions come apart for learned distributed symmetry-breaking rules. A
> network can reproduce every native decision of a degree-three
> Barenboim–Elkin base-case coloring teacher (Linial and Kuhn–Wattenhofer
> reductions), yet confirm most proposals the teacher would veto under
> admissible message interventions (balanced accuracy 53–60%). Supervising
> that confirmation step on teacher-labelled interventions raises it above
> 99.9%, at a higher label cost. Under frozen transfer to larger degrees,
> sizes and palettes, the remaining failures replay to distinct safety and
> progress mechanisms that aggregate success rates hide. Learned distributed
> rules should therefore be reported along three axes: native functional
> success, fidelity under declared interventions, and transfer.

**Relevance (for Pedro's "why should anyone care").**
- **NAR and learned-distributed-policy work** (e.g., Tolstaya/Gama, Zhao,
  Langedal–Manne) still evaluates mostly by task success or trace accuracy.
  Our result is a concrete case where both are perfect and the rule is
  still wrong on inputs that any deviating neighbour would produce.
- **Theory now proves exact learnability from instruction-style supervision**
  (Qharabagh et al.; Wittig et al.; Nerem et al.). Our experiments are
  *consistent with* the practical message: trace-only supervision
  underdetermines the local rule, while supervision covering message
  configurations closes the gap.
- **One link would make the relevance concrete if the data support it.**
  Check whether continuation fidelity predicts transfer *safety* failures.
  For example, the secondary matched CT E1 checkpoints show incorrect
  simultaneous confirmations in transfer. **This link is not established in
  the current manuscript and must not be asserted without an analysis.**

**Suggested replacement for "Relation to previous work"** (about 170 words; adjust keys after the bib merge):

> Neural execution of algorithms from traces is established
> \citep{velickovic2020execution,velickovic2022clrs,ibarz2022generalist},
> including randomized distributed rules such as Luby-type MIS
> \citep{minder2023salsa}. Learned modules inside parallel or distributed
> coloring and scheduling skeletons \citep{langedal2025ordering,zhao2020scheduling}
> and imitation-trained decentralized GNN controllers \citep{gama2022synthesizing}
> also exist. Exact learnability of LOCAL algorithms from local-instruction
> tables is now proved \citep{qharabagh2026exact}, and size extrapolation is
> proved for deterministic algorithms \citep{nerem2026extrapolate,wittig2026which}.
> Supervising a student with teacher labels off the native trajectory is
> classical: dynamic oracles, expert queries at perturbed states, and
> interchange-intervention training
> \citep{goldberg2012dynamic,ross2011dagger,laskey2017dart,galashov2022augmentation,geiger2022inducing}.
> Hint-ReLIC instead enforces invariance under augmentations that preserve
> the next step \citep{bevilacqua2023causal}. That accuracy underdetermines
> mechanism is likewise known
> \citep{damour2022underspecification,rodionov2023without,saldyt2025mind,lowe2019pitfalls}.
> We claim none of these ideas. We contribute an audited instance for
> distributed coloring: typed message-intervention contracts, a paired
> passive/teacher-forced/intervention comparison showing that perfect native
> agreement leaves confirmation behaviour underdetermined, and replayable
> safety and progress failure witnesses under frozen transfer. Concurrent
> work reports a related gap between fitting every hint and correct
> execution \citep{najafgholizadeh2026premature}.

---

## 6. Bibliography actions

- **New file** `paper/ucgnn/revision-references.bib`: 49 entries, all
  VERIFIED, with no key collisions with the four existing files. It is not
  yet referenced by `\bibliography{...}` in `main.tex`. Add it when an editor
  is authorized.
- **Must cite** (threat-closing): `qharabagh2026exact`, `minder2023salsa`,
  `geiger2022inducing`, `galashov2022augmentation`, `goldberg2012dynamic`,
  `laskey2017dart`, `damour2022underspecification`, `saldyt2025mind`,
  `najafgholizadeh2026premature`, `rodionov2025discrete`,
  `nerem2026extrapolate`, `xu2020reason`, `yehudai2021size`,
  `angluin1980local`, `hella2015weak`.
- **Should cite** (accurate positioning): `ibarz2022generalist`,
  `mirjanic2024latent`, `wu2022causal`, `geiger2025causal`,
  `sutter2025nonlinear`, `lowe2019pitfalls`, `williams1989learning`,
  `bengio2015scheduled`, `spencer2021feedback`, `huang2020counterfactually`,
  `kaushik2020learning`, `angluin1988queries`, `daume2009searn`,
  `chang2015lols`, `tolstaya2020learning`, `gama2022synthesizing`,
  `sato2021random`, `abboud2021surprising`, `cai2017making`,
  `reed2016neural`, `barenboim2009distributed`, `clopper1934use`,
  `kingma2015adam`.
- **Optional** (context): `engelmayer2024parallel`, `mahdavi2023towards`,
  `kujawa2024multiple`, `he2026minar`, `mccoy2020berts`, `zhong2023clock`,
  `gardner2020contrast`, `ribeiro2020checklist`, `schuetz2022coloring`,
  `li2022rethinking`, `parkar2025unsupervised`.
- **Fix existing keys:** `lemos2019coloring`, `velickovic2021nar`,
  `wittig2026which`, `langedal2025ordering` (Section 1 BibTeX), plus DOIs or
  pages for `kuhn2006complexity`, `schneider2010symmetry`,
  `linial1992locality`, `gebremedhin2000scalable` and `rodionov2023without`.
- **Update `docs/NOVELTY_AND_RELATED_WORK.md`.** Its claim that no preprint
  is presented as accepted is now stale for Wittig et al., which is an ICML
  2026 oral. Its matrix also lacks the threats listed in Section 3.

---

## 7. Search log and limitations

- **Primary sources used:**
  - arXiv API and abstract pages and PDFs (text extracted locally);
  - PMLR (v15, v37, v78, v100, v139, v162, v198, v202, v231, v267, v336);
  - JMLR; ACL Anthology; NeurIPS proceedings;
  - icml.cc (Wittig oral);
  - Crossref DOI records (ICTAI, Patterns, JCSS, IPL, SISC, SICOMP, PODC,
    STOC, Concurrency, TEST, SAGMB, SDM, IJCAI, Distributed Computing,
    PRR, TSP, Machine Learning, Neural Computation, Biometrika, ACL/EMNLP,
    BlackboxNLP, Insights, AAMAS, ICASSP);
  - mlanthology, for ICLR metadata only.
- **Unavailable:** OpenReview (bot checks); SIAM epubs (403); ScienceDirect
  content (paywall). Metadata came from Crossref instead.
- **Query themes (30 Oct 2026):**
  - learning distributed or LOCAL algorithms with GNNs, including MIS,
    coloring and symmetry breaking (2024–2026);
  - counterfactual or interventional supervision in NAR;
  - teacher queries on alternative inputs and messages;
  - mechanistic or faithfulness evaluation of NAR;
  - interchange intervention and causal abstraction;
  - dynamic oracles;
  - expert-labelled perturbations in imitation learning, with label-budget
    controls;
  - provable size extrapolation (2025–2026);
  - neural graph coloring (2024–2026);
  - CONGEST/GNN complexity;
  - emergent-communication measurement.
- **Negative results.** No search found a paper that imitates distributed
  coloring teachers, or that combines typed message interventions with
  teacher-labelled confirmation targets for a learned distributed rule. This
  is a focused search, not proof of absence. Very recent arXiv postings
  (September–October 2026) are the most likely gap. Recheck
  `najafgholizadeh2026premature`'s citing papers and new SALSA-CLRS
  follow-ups before submission.
- **Content details marked UNVERIFIED:**
  - whether Luby (1993) uses participation probability 1/2;
  - the Rodionov–Prokhorenkova 2023 page range;
  - the contents of the Tan–Sim–Chua survey;
  - the Holm (1979) record, not reopened;
  - the exact OpenReview IDs for Loukas 2020 and Xu 2020.
