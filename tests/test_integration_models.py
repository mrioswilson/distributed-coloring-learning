import networkx as nx
import numpy as np
import pytest
import torch

from dcl.integration.models import ARCHITECTURES, RootedPalettePolicy
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, TEACHERS
from dcl.multiteacher_e1.model import RootedE1, native_prediction
from dcl.multiteacher_e1.supervision import tensor_arrays


def test_normalized_adapter_is_exactly_existing_rooted_e1():
    p = ColoringProgram(nx.path_graph(4), "LUBY", 8)
    arrays = tensor_arrays(encode_roots(p.snapshot(), p.neighbors, np.arange(4)))
    torch.manual_seed(7); a = RootedE1("LUBY")
    torch.manual_seed(7); b = RootedPalettePolicy("LUBY", "normalized")
    x, y = a(*arrays), b(*arrays)
    for v, w in zip(native_prediction(x), native_prediction(y)):
        assert torch.equal(v, w)
    assert torch.equal(x.scores, y.scores) and torch.equal(x.commit, y.commit)


@pytest.mark.parametrize("teacher", TEACHERS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_all_native_action_interfaces_and_gradients(teacher, architecture):
    torch.set_num_threads(1)
    p = ColoringProgram(nx.path_graph(4), teacher, 81)
    s = p.snapshot()
    arrays = tensor_arrays(encode_roots(s, p.neighbors, np.arange(4)))
    model = RootedPalettePolicy(teacher, architecture)
    out = model(*arrays, phase=s.phase)
    assert out.scores.shape[0] == 4 and torch.isfinite(out.commit).all()
    (out.scores.sum() + out.commit.sum()).backward()
    assert model.core.scorer[0].weight.grad is not None


def test_no_communication_predictions_ignore_changed_neighbor_information():
    p = ColoringProgram(nx.path_graph(3), "LUBY", 88)
    s = p.snapshot(); s.colors[2] = 0
    a = tensor_arrays(encode_roots(s, p.neighbors, np.array([1])))
    s.colors[2] = 1
    b = tensor_arrays(encode_roots(s, p.neighbors, np.array([1])))
    model = RootedPalettePolicy("LUBY", "no_communication")
    x, y = model(*a), model(*b)
    assert torch.equal(x.scores, y.scores) and torch.equal(x.commit, y.commit)


def test_new_teacher_training_saves_and_reuses_exact_final_checkpoint(tmp_path):
    import json
    import time
    from dcl.integration.study import setup, sources, train_cell
    from dcl.multiteacher.data import build_pool
    config = json.loads(open("configs/integration-balanced.json").read())
    config.update(n=8, graphs=2, val_graphs=1, batch_decisions=4)
    setup(config)
    pool = build_pool(config, tmp_path, "LUBY", "train")
    validation = build_pool(config, tmp_path, "LUBY", "validation")
    design = dict(updates=2, source_hashes=sources())
    a = train_cell(config, design, tmp_path, "LUBY", "deepsets", 123, pool, validation, time.perf_counter() + 30)
    b = train_cell(config, design, tmp_path, "LUBY", "deepsets", 123, pool, validation, time.perf_counter() + 30)
    assert a == b and a["supervised_root_decisions"] == 8
