"""Explicit private-tape/ID interventions with replayable native-state witnesses.

This training/data library is deliberately absent from autonomous inference.
It supplements multicf's finite proposal bank: downstream availability,
comparisons, polynomial equalities and schedule messages are always re-encoded.
"""
import copy

import networkx as nx
import numpy as np

from dcl.multiteacher.programs import ColoringProgram, Choices, ID_SPACE, available, teacher_choices
from dcl.samplecomplexity.protocol import horizon
from dcl.multicf.states import (trajectory, replay, state_hash, serial_choices,
                               validate_choice, alternate_choice)


def assign_ids_at_initialization(program, ids):
    """Rebuild all ID-dependent state before the first snapshot, never mid-run."""
    if program.steps or program.index:
        raise ValueError("ID intervention requires a fresh native history")
    ids=np.asarray(ids)
    if ids.shape!=(program.n,) or ids.dtype.kind not in "iu" or len(np.unique(ids))!=program.n:
        raise ValueError("Distinct integer IDs required")
    if np.any((ids<0)|(ids>=ID_SPACE)):
        raise ValueError("ID outside native space")
    program.ids=ids.astype(np.int64).copy();program.order=np.argsort(program.ids)
    program.blocks=np.array_split(program.order,min(4,program.n))
    program.slots=np.zeros(program.n,np.int64)
    if program.teacher=="SPECULATIVE_GM":
        for block in program.blocks:program.slots[block]=np.arange(len(block))
    program.width=max(map(len,program.blocks))
    if program.teacher=="BE_BASE_D3":program.aux=program.ids.copy()


def _with_ids(graph,teacher,tape_seed,ids,alternative_seed):
    p=ColoringProgram(graph,teacher,tape_seed,max_rounds=horizon(teacher,len(graph)))
    assign_ids_at_initialization(p,ids)
    rng=np.random.default_rng(alternative_seed)
    prefix=[]
    while not p.finished:
        s=p.snapshot()
        witness=dict(teacher=teacher,n=len(graph),edges=[list(e) for e in sorted(graph.edges())],
            tape_seed=tape_seed,native_step=p.steps,state_sha256=state_hash(s),
            prefix=copy.deepcopy(prefix),initial_ids=np.asarray(ids).tolist(),prefix_kind="ID-reassigned-legal-replay")
        yield p,s,teacher_choices(s,p.neighbors,teacher),witness
        chosen=alternate_choice(p,s,rng);validate_choice(p,s,chosen)
        p.apply(s,chosen)
        prefix.append(dict(state_sha256=state_hash(s),choices=serial_choices(chosen)))
    if p.failure:raise RuntimeError("Intervened legal history failed")


def retape(state,teacher,kind):
    """Intervene only in current private inputs; persistent state remains fixed."""
    s=copy.deepcopy(state)
    if teacher in ("LUBY","JOHANSSON"):
        if kind=="participation_boundary":
            a=np.float32(.5)
            s.draws[:,0]=np.resize(np.array([np.nextafter(a,np.float32(0)),a,np.nextafter(a,np.float32(1))]),len(s.colors))
        elif kind=="equal_offers":
            s.draws[:,0]=0.;s.draws[:,1]=0.
        elif kind=="candidate_boundary":
            values=[0.]
            for d in (2,3,4):
                for j in range(1,d):
                    a=np.float32(j/d)
                    values.extend([np.nextafter(a,np.float32(0)),a,np.nextafter(a,np.float32(1))])
            s.draws[:,1]=np.resize(np.array(values,np.float32),len(s.colors))
        else:raise ValueError("Unknown private-draw intervention")
    elif teacher=="COLOR_TRIALS":
        if kind=="all_preferences_tie":s.preferences[:]=2
        elif kind=="alternating_preferences":
            s.preferences[:]=np.arange(len(s.colors))[:,None]%4
        elif kind=="one_private_advantage":
            s.preferences[:]=2
            roots=np.flatnonzero(s.permission)
            if len(roots):s.preferences[roots[0],0]=3
        else:raise ValueError("Unknown preference intervention")
    else:raise ValueError("This teacher has no native private tape to intervene in")
    return s


def contexts(graph,teacher,tape_seed=73101):
    """Hard contexts valid for each teacher, with distinct causal operations."""
    if teacher in ("LUBY","JOHANSSON","COLOR_TRIALS"):
        kinds=(("participation_boundary","candidate_boundary","equal_offers") if teacher!="COLOR_TRIALS" else
               ("all_preferences_tie","alternating_preferences","one_private_advantage"))
        for p,s,_,witness in trajectory(graph,teacher,tape_seed):
            for kind in kinds:
                changed=retape(s,teacher,kind)
                proof=dict(base=witness,intervention=kind,state_sha256=state_hash(changed))
                yield p,changed,teacher_choices(changed,p.neighbors,teacher),proof
    else:
        # All comparisons, blocks, auxiliary colors and subsequent native phases
        # descend from the new IDs; none is kept from the original assignment.
        for ids in (np.arange(len(graph),dtype=np.int64)+1,np.arange(len(graph),dtype=np.int64)[::-1]+1000):
            for p,s,exact,witness in _with_ids(graph,teacher,tape_seed,ids,171):
                proof=dict(base=witness,intervention="initial-IDs-and-legal-prefix",state_sha256=state_hash(s))
                yield p,s,exact,proof


def replay_context(proof):
    witness=proof["base"]
    if "initial_ids" not in witness:
        p,s=replay(witness)
        s=retape(s,p.teacher,proof["intervention"])
    else:
        g=nx.Graph();g.add_nodes_from(range(witness["n"]));g.add_edges_from(witness["edges"])
        p=ColoringProgram(g,witness["teacher"],witness["tape_seed"],max_rounds=horizon(witness["teacher"],len(g)))
        assign_ids_at_initialization(p,witness["initial_ids"])
        for event in witness["prefix"]:
            s=p.snapshot()
            if state_hash(s)!=event["state_sha256"]:raise ValueError("Intervened prefix hash differs")
            raw=event["choices"];c=Choices(np.asarray(raw["color"],int),np.asarray(raw["auxiliary"],int),np.asarray(raw["mark"],bool))
            validate_choice(p,s,c);p.apply(s,c)
        s=p.snapshot()
        if state_hash(s)!=witness["state_sha256"]:raise ValueError("Intervened native snapshot differs")
    if state_hash(s)!=proof["state_sha256"]:raise ValueError("Causal intervention snapshot differs")
    return p,s
