from fractions import Fraction
import numpy as np
import torch

from dcl.paperstudy.certification import affine_interval,commit_certificate,realizable_star_probe
from dcl.paperstudy.evaluation import SpecialistAdapter,rollout
from dcl.paperstudy.models import make_model
from dcl.m4_data import _edge_index
from dcl.study.evaluation import run_rollout as specialist_rollout
from dcl.universal.evaluation import run_rollout
import networkx as nx


def test_interval_encloses_exact_binary_rational_affine_values():
    rng=np.random.default_rng(541)
    w=rng.normal(size=(8,13)).astype(np.float32).astype(np.float64)
    x=rng.normal(size=(4,13)).astype(np.float32).astype(np.float64)
    b=rng.normal(size=8).astype(np.float32).astype(np.float64)
    lo,hi=affine_interval(x,x,w,b)
    for i in range(4):
        for j in range(8):
            exact=Fraction(float(b[j]))+sum(Fraction(float(a))*Fraction(float(c)) for a,c in zip(w[j],x[i]))
            assert Fraction(float(lo[i,j])) <= exact <= Fraction(float(hi[i,j]))
    low,high=affine_interval(x,x,w,b,True)
    values=(torch.tensor(x,dtype=torch.float32)@torch.tensor(w,dtype=torch.float32).T+torch.tensor(b,dtype=torch.float32)).numpy()
    assert np.all(values>=low) and np.all(values<=high)


def test_instrumented_rollout_same_decisions_as_original():
    torch.set_num_threads(1)
    torch.manual_seed(99)
    model=make_model("normalized")
    graph=nx.cycle_graph(12)
    edges=_edge_index(graph)
    a,ca=run_rollout(model,12,edges,4,123,max_rounds=9)
    b,cb=rollout(model,12,edges,4,123,max_rounds=9)
    np.testing.assert_array_equal(ca,cb)
    for key in ("rounds","success","unsafe_commits","candidate_correct","uncolored_nodes"):
        assert a[key]==b[key]
    assert len(b["round_trace"])==b["rounds"]


def test_specialist_adapter_preserves_hard_protocol():
    from dcl.m4_models import M4B
    torch.manual_seed(11)
    model=M4B(num_colors=4)
    graph=nx.cycle_graph(10)
    a=specialist_rollout(model,graph,321,4,max_rounds=9)
    b,colors=rollout(SpecialistAdapter(model),10,_edge_index(graph),4,321,max_rounds=9)
    np.testing.assert_array_equal(a["colors"],colors)
    for key in ("rounds","success","unsafe_commits","candidate_correct"):
        assert a[key]==b[key]


def test_interval_certificate_rejects_incorrect_constant_commit_and_sum_scope():
    model=make_model("normalized")
    with torch.no_grad():
        for p in model.commit_head.parameters():
            p.zero_()
        model.commit_head[2].bias.fill_(1)
    certificate=commit_certificate(model)
    assert not certificate["certified_exact_rule"]
    assert not certificate["certified_no_false_accept"]
    assert len(certificate["certified_unsafe_masks"])==48
    assert not commit_certificate(make_model("sum"))["applicable"]


def test_full_realizable_stars_equal_the_type_reduction():
    torch.set_num_threads(1)
    for treatment in ("normalized","raw_endpoints","sum","mean","transformer"):
        model=make_model(treatment)
        rows=realizable_star_probe(model)
        assert len(rows)==64 and all(r["same_decision"] for r in rows)
        assert max(r["absolute_logit_difference"] for r in rows)<2e-5
