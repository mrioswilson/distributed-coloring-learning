# Architecture audit: local palettes (21 October 2026)

This audit precedes new outcomes. Source: unchanged `src/dcl/universal/model.py`
and `src/dcl/paperstudy/models.py`. The primary 30 networks are the six paper
treatments, seeds 41000–41004, epoch **120**, not the separately trained historical
E1 seeds 0–4. NORMALIZED has the E1 architecture; the identities are not interchangeable.

## Exact interface and computation

`UniversalColorGNN.forward` (`universal/model.py:137`) receives permanent colors
`colors[N]` (-1 pending), independent private `rp[N], rc[N]`, directed incidences
`edge_index[2,E]`, and integer **per-node** `palette[N]`. No degree, available-set
oracle, identifier, teacher action, or graph embedding is supplied. Constructor
dimensions do not depend on K. `edge_chunk` is a memory parameter, not an input
feature. `_slots` (`model.py:83`) validates own fixed colors and computes a storage
maximum; this maximum is not concatenated to features.

Writing u_v=1[q_v=-1], o_vc=1[q_v=c], the first communication is

    m¹_vc = MLP₂→₃₂→₃₂([u_v,o_vc]),
    h_vc = MLP₃₃→₆₄→₃₂([u_v, R_{w∈N(v)} m¹_wc]),
    α_vc = sigmoid(Linear₃₂→₁(h_vc)) 1[c<K_v].

Both message/state MLPs end in ReLU. R is MAX for NORMALIZED, RAW, DeepSets and
Transformer, SUM for SUM, and arithmetic neighbor MEAN for MEAN. Empty aggregates
are zero; MEAN divides by max(actual degree,1), with no additional normalization
of SUM. All these availability values are **learned**, not exact availability masks.

For NORMALIZED, D_v=max(Σ_{c<K_v} α_vc,0.5), L_vc=Σ_{j<c}α_vj/D_v,
U_vc=L_vc+α_vc/D_v. The scorer is MLP₃₆→₆₄→₁ without final bias, on
[h,α,L,U,rc]. RAW replaces only L,U by unnormalized endpoints; it is NOT historical
A1/RAW_PREFIX (which has an additional total-mass input and different shape).
NORMALIZED/RAW/SUM/MEAN have **16,035 parameters**.

DeepSets (`paperstudy/models.py:57–65`) uses x_c=c/max(K_v−1,1), z=1/K_v,
φ=MLP₃₅→₁₆→₁₆([h,α,x,z]), masked mean b_v=Σ_{c<K_v}φ/K_v, then
MLP₅₂→₆₄→₁([h,α,x,z,rc,b_v]); **17,907 parameters**.
Transformer (`models.py:67–71`) projects [h,α,x,z,rc] (36 dimensions) to 16,
applies one 2-head TransformerEncoderLayer (feedforward width 32, ReLU,
dropout 0, post-layer-normalization), then MLP₁₆→₃₂→₁; **16,995 parameters**.
Attention is between a single node's colors, never graph-wide. Invalid keys are
masked, invalid output scores set to −1e9. Diagnostic prefixes in `stage1` have
no path into either prefix-free scorer. Both use functional positions and
inverse **own** K; there are no learned absolute-position arrays.

**Why distinguish identically represented free colors?** Actual DeepSets and
Transformer receive c/(K_v−1). Their h and α may be equal on all free slots, but
their scalar position is different. Their candidate maps are not color-permutation
equivariant. The phrase "no prefixes" does not mean "no order information".

All participation heads receive **only [rp,u]**, MLP₂→₃₂→₁, threshold strictly >0.
Candidate heads receive rc and first-communication state, never rp or neighbor
randomness. In evaluation mode dropout is disabled (configured 0). Therefore,
conditional on fixed state, participation and candidate are independent when
the two inputs are independent. Across nodes private input draws also ensure
independence of these first-stage decisions. Learned COMMIT decisions are not
independent because they observe neighbors' first-stage actions.

Second communication (`model.py:116–135`; `models.py:108–125`):

    m²_wc=MLP₄→₃₂→₃₂([u_w,1[q_w=c],B_w,1[C_w=c]]),
    z_vc=MLP₃₆→₆₄→₃₂([u_v,1[q_v=c],B_v,1[C_v=c],R_w m²_wc]),
    logit J_v=MLP₃₅→₆₄→₁([z_v,C_v,B_v,u_v,1[C_v≥0]]).

Here z_v,C_v denotes the gathered 32-vector, not an extra scalar candidate.
Fixed nodes have gathered zero and candidate −1; no fixed color is updated.

## Heterogeneous lists and storage

The original code already accepts vector K_v, but uses rectangular tensors of
width max_v K_v. A neighbor with fixed color outside the receiver's palette
has own-bit zero at each valid receiver color; it is still a fixed neighbor.
A neighbor proposing outside the receiver's palette likewise has offer-bit zero.
The pending/participating bits must **not** disappear. Those are legal neutral
message types at the queried receiver color, not proposals in nonexistent slots.
Multiplying all sender messages by 1[c<K_sender] would erase them and change the
historical policy. Only receiver colors are scored, normalized or attended.
Neither communication message contains position or K: functional positions enter
only the local palette scorer. Different local positional coordinates therefore
do not require aligning or transforming neighbors' embeddings.

The extension adapter groups **receivers by their own K**, evaluates first-stage
messages on those receiver colors, and scores each group at exactly that K.
It computes second-stage messages only at the receiver's chosen candidate (the
other z slots never enter COMMIT). This is algebraic restriction/gather of the
same modules, with no new inputs, weights, oracle masks or inference shield.
It is a zero-shot extension in real arithmetic, with floating-point equivalence
tested separately. Processing chunks affect runtime but are not features.

Padding-only changes of the original rectangular forward can alter floating-point
reduction order or attention kernels even with exact-zero masks. The adapter's
own-K reductions eliminate functional dependence on another graph's maximum.
Tests cover arbitrary output padding, mixed neighbor lists, isolated K=1,
node relabeling, disjoint-batch composition, distinct other-graph maxima,
individual/grouped execution, and original homogeneous/heterogeneous forwards.
Float32 comparisons use 3e-5 absolute/relative tolerance and exact hard actions
away from ties; explicit boundary probes report discrepancies rather than hiding
them. A failure of action equivalence is retained with logits/margins.
Invalid diagnostic endpoints are not compared: the original rectangular code
can leave L=U=1 outside the list whereas the adapter returns zero there. Neither
has a candidate path. Candidate argmax is restricted to valid colors, independently
of the finite negative sentinel used for padded output storage. This hardening
has identical decisions whenever the original argmax is in its declared list.

## Randomness and numerical reference

The new evaluator draws float64 uniform rp,rc using NumPy PCG64, then clips their
float32 conversion to nextafter(float32(1),0); otherwise a value below 1 can round
to 1. The exact mathematical teacher law is uniform on sorted available colors.
We separately compare floor(float64 rc × |A|) with floor(float32(rc) × |A|)
computed in float64, and probe float32 interval neighbors and endpoints. Argmax
ties choose the lowest slot. Confirmed-color histograms are separate from candidate
histograms and from candidates conditional on participation.

All 30 checkpoint hashes and source/environment hashes are frozen in the study
manifest. Original modules and checkpoints remain unchanged. M6B's learned
dimensions fix K, so M6B is excluded rather than padded into a different model.

The historical slot construction hardcodes float32 (`.float()`); simply calling
the entire historical model `.double()` is not a supported float64 execution.
The secondary precision probes use dtype-aware receiver-query modules with the
same coefficients. Their 257-point deterministic grids compare float32 and
float64 evaluation with **identical converted rc**, separately from changing
rc's precision. No discrepancy on such grids is a universal numerical certificate.
`participation-exact-real.json` integrates the 35 frozen ReLU participation heads
using rational arithmetic at u=1. Its probabilities concern real evaluation of
the stored coefficients, not rounding of intermediate GPU operations.
