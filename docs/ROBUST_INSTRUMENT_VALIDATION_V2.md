# Independent validation round two: unknown pair location

Development relocation probe on fresh synthetic controls found first-version
omnibus detections of 0/10 at m=250 and 3/10 at m=1000 when the dependent
leaf pair was relocated to (137,181). A relocated XOR triple gave 1/10 and
0/10. These outcomes are retained in remote-development-v1.json. They show
a limitation of fixed sampled interaction regions, not equality of laws.

Version two retains the full first-version procedure and adds selection-only
pair search. On raw color and pooled-median binary representations, compute
the difference between selection-sample covariance matrices, after pooled
selection standardization. Add the eight largest absolute upper-triangular
entries of each matrix to the candidate categorical pair regions. The usual
selection score chooses four interaction witnesses. No evaluation rows or
source labels enter this search. All components still receive joint held-out
permutation calibration. The original algorithm is preserved in stats_v1.py
and its first validation outcomes remain separate.

Run development2 first. Freeze this version before validation2 is generated.
Fresh validation2 namespaces contain five null laws with 400 trials each,
and twelve alternative classes at all five m, with 30 trials per cell.
The original ten classes remain; add remote pair and remote triple controls.
Compare first-version omnibus on exactly the same new arrays. Newly sampled
null2 banks provide another 400 Luby/Luby and 400 actual-checkpoint self
trials for each of the three architectures. No first-round null bank is
reused as fresh data. The second-round gate retains all original targets,
adding >=80% lower Wilson power for remote pairs at m=1000. No 80% target
is imposed on arbitrary unknown triples: searching all third-order regions
has not been implemented. Their known-TV failures must be retained and
disclosed as a finite-sample and representation limitation.

Run sensitivity2 using new seed namespaces and the same prospective stress
grid as round one. Null acceptance and global-loss criteria are unchanged.
The main method is fixed before this validation; it is not selected using
new neural teacher-versus-model comparisons. After acceptance, commit a
separate final confirmation protocol before generating its first bank.
