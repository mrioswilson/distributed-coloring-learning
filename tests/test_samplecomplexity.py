"""Scientific regression tests: budgets, leakage, strict gates and native clocks."""
import copy
import json

import networkx as nx
import numpy as np
import pytest
import torch

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, TEACHERS, teacher_choices, LINIAL, DETECT
from dcl.multiteacher_e1.experiment import predictions
from dcl.samplecomplexity.data import acquire, load, components
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.model import relational_arrays
from dcl.samplecomplexity.protocol import (cases, five_seed_gate, freeze, horizon, passed_run, summarize_runs, source_hashes)
from dcl.samplecomplexity.training import initialize, passive_loss, train


@pytest.fixture
def config():
    c=json.loads(__import__("pathlib").Path("configs/sample-complexity.json").read_text())
    c.update(namespace="test-only", training_sizes=[8], pilot_runs=7, development_runs=7,
             updates=3, batch_size=4, fidelity_decisions=16)
    return c


def valid_row(i=0):
    return dict(case_id=str(i), proper=True, palette=True, invariants=True, terminated=True,
                no_technical_error=True, no_cutoff=True, terminal_correct=True, failure=None, native_steps=1)


@pytest.mark.parametrize("field", ["proper","palette","invariants","terminated","no_technical_error","no_cutoff","terminal_correct"])
def test_gate_requires_every_condition(field):
    row=valid_row(); assert passed_run(row)
    row[field]=False; assert not passed_run(row)
    del row[field]; assert not passed_run(row)


def test_all_1000_runs_and_all_five_seeds_are_required():
    rows=[valid_row(i) for i in range(1000)]
    s=summarize_runs(rows,[str(i) for i in range(1000)])
    assert s["pass_all"] and s["lower_one_sided_95"] == pytest.approx(.9970087504549047)
    five={i:s for i in range(5)}
    assert five_seed_gate(five,list(range(5)))["status"]=="PASS"
    rows[-1]["no_cutoff"]=False
    five[4]=summarize_runs(rows,[str(i) for i in range(1000)])
    assert five_seed_gate(five,list(range(5)))["seeds_pass"]==4
    assert five_seed_gate(five,list(range(5)))["status"]=="NOT CERTIFIED"
    assert not summarize_runs(rows[:999],[str(i) for i in range(1000)])["pass_all"]
    assert not summarize_runs(rows+[rows[0]],[str(i) for i in range(1000)])["pass_all"]


@pytest.mark.parametrize("teacher",TEACHERS)
def test_exact_teachers_pass_prospective_horizons(teacher):
    suite=[(nx.path_graph(12),dict(case_id="path",tape_seed=101))]
    rows,_=evaluate(None,teacher,suite)
    assert passed_run(rows[0]), rows[0]


def test_jp_increasing_ids_requires_n_rounds_beyond_old_128():
    p=ColoringProgram(nx.path_graph(129),"JP",1,max_rounds=horizon("JP",129))
    p.ids=np.arange(129)
    while not p.finished:
        state=p.snapshot(); p.apply(state,teacher_choices(state,p.neighbors,"JP"))
    assert p.failure is None and p.steps==129


def test_zero_action_policy_is_a_cutoff_and_never_success():
    class Zero(torch.nn.Module):
        def forward(self,*args,**kw): raise RuntimeError("synthetic technical failure")
    rows,_=evaluate(Zero(),"LUBY",[(nx.path_graph(4),dict(case_id="error",tape_seed=1))])
    assert rows[0]["failure"]=="technical_error" and not passed_run(rows[0])


def test_waiting_policy_reaches_cutoff_without_replacement(monkeypatch):
    import dcl.samplecomplexity.evaluation as evaluation
    monkeypatch.setattr(evaluation,"horizon",lambda teacher,n:2)
    monkeypatch.setattr(evaluation,"predictions",lambda model,arrays,phases:
                        (np.zeros(len(phases),int),np.zeros(len(phases),int),np.zeros(len(phases),bool)))
    suite=[(nx.path_graph(4),dict(case_id="original-tape",tape_seed=11))]
    rows,_=evaluate(torch.nn.Identity(),"LUBY",suite)
    assert len(rows)==1 and rows[0]["case_id"]=="original-tape"
    assert rows[0]["native_steps"]==2 and rows[0]["no_cutoff"] is False
    assert not passed_run(rows[0])


def test_be_auxiliary_violation_fails_before_final_coloring(monkeypatch):
    import dcl.samplecomplexity.evaluation as evaluation
    monkeypatch.setattr(evaluation,"predictions",lambda model,arrays,phases:
                        (np.zeros(len(phases),int),np.zeros(len(phases),int),np.zeros(len(phases),bool)))
    rows,_=evaluate(torch.nn.Identity(),"BE_BASE_D3",[(nx.path_graph(4),dict(case_id="be",tape_seed=11))])
    assert rows[0]["failure"]=="invariant" and rows[0]["invariants"] is False
    assert rows[0]["native_steps"]==1 and not passed_run(rows[0])


def test_interactive_labels_preserve_bad_trajectories_and_budget(config,tmp_path):
    model=initialize("JP","normalized",83101).eval()
    m=acquire(config,"JP",83101,90,tmp_path/"interactive",policy=model)
    assert m["queries"]<=90 and m["queries"]==sum(m["by_component"].values())
    assert m["identity"]["interface"]=="components-interactive-v1"
    assert all("failure" in r for r in m["trajectories"])


@pytest.mark.parametrize("teacher",TEACHERS)
def test_query_counts_all_phases_and_replay_acquires_nothing(config,tmp_path,teacher,monkeypatch):
    folder=tmp_path/teacher
    m=acquire(config,teacher,101,600,folder)
    assert m["queries"]<=600 and m["queries"]==sum(m["by_component"].values())
    arrays,labels,accounting=load(folder,100)
    assert accounting["queries"]<=100
    assert accounting["queries"]==sum(len(components(int(p))) for p in labels[:,0])
    if teacher=="BE_BASE_D3": assert "linial_point" in m["by_component"]
    if teacher=="SPECULATIVE_GM": assert "conflict_mark" in m["by_component"]
    def forbidden(*a,**k): raise AssertionError("Teacher called during optimization")
    monkeypatch.setattr("dcl.samplecomplexity.data.teacher_choices",forbidden)
    model=initialize(teacher,"normalized",101)
    phase=int(labels[0,0]); use=labels[:,0]==phase
    loss=passive_loss(model,tuple(a[use] for a in arrays),labels[use]); loss.backward()
    assert torch.isfinite(loss)
    assert json.loads((folder/"manifest.json").read_text())["queries"]==m["queries"]


@pytest.mark.parametrize("teacher",["JP","SPECULATIVE_GM"])
@pytest.mark.parametrize("architecture",["normalized","deepsets","transformer"])
def test_order_preserving_ids_give_identical_policy_actions(teacher,architecture):
    p=ColoringProgram(nx.cycle_graph(8),teacher,19)
    p.ids=np.array([1,3,5,7,9,11,13,15]); p.order=np.argsort(p.ids)
    state=p.snapshot(); arrays=encode_roots(state,p.neighbors,np.arange(8))
    state.ids=state.ids*100+13
    other=encode_roots(state,p.neighbors,np.arange(8))
    model=initialize(teacher,architecture,31).eval()
    with torch.inference_mode():
        first=predictions(model,arrays,np.full(8,state.phase))
        second=predictions(model,other,np.full(8,state.phase))
    assert all(np.array_equal(a,b) for a,b in zip(first,second))


def test_anonymous_interface_cannot_distinguish_swapped_jp_ids():
    p=ColoringProgram(nx.path_graph(2),"JP",1)
    state=p.snapshot(); state.ids=np.array([1,2])
    before=teacher_choices(state,p.neighbors,"JP").color.copy()
    state.ids=state.ids[::-1].copy()
    after=teacher_choices(state,p.neighbors,"JP").color
    assert not np.array_equal(before,after)
    assert np.array_equal(state.colors,np.array([-1,-1]))


def test_train_is_reproducible_from_fresh_initialization(config,tmp_path):
    acquire(config,"LUBY",42,300,tmp_path/"data")
    a=train(config,"LUBY","normalized",42,300,tmp_path/"data",tmp_path/"a",dev=False)
    b=train(config,"LUBY","normalized",42,300,tmp_path/"data",tmp_path/"b",dev=False)
    assert a["initial_hash"]==b["initial_hash"] and a["weights_hash"]==b["weights_hash"]
    assert a["accounting"]==b["accounting"]
    assert a["accounting"]["queries"]==300
    assert a["parameter_updates"]==3


def test_splits_are_independent_and_sizes_disjoint(config):
    dev=cases(config,"development",83101); final=cases(config,"final",83101)
    assert {m["tape_seed"] for _,m in dev}.isdisjoint(m["tape_seed"] for _,m in final)
    assert set(config["training_sizes"]).isdisjoint(config["development_sizes"])
    assert set(config["training_sizes"]).isdisjoint(config["final_sizes"])
    assert set(config["development_sizes"]).isdisjoint(config["final_sizes"])
    for g,m in final:
        assert max(dict(g.degree()).values())<=3
        if m["family"]=="bipartite": assert nx.is_bipartite(g)


def test_frozen_objects_cannot_be_replaced(tmp_path):
    freeze(tmp_path/"frozen.json",dict(budget=1000))
    freeze(tmp_path/"frozen.json",dict(budget=1000))
    with pytest.raises(RuntimeError): freeze(tmp_path/"frozen.json",dict(budget=3000))
