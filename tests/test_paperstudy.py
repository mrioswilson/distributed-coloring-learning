from dataclasses import replace
import copy

import numpy as np
import pytest
import torch

from dcl.paperstudy.config import Run, cohort_plan
from dcl.paperstudy.models import PRIMARY, make_model, neighbor_reduce
from dcl.paperstudy.teacher import cyclic_law, cyclic_offer, teacher_trajectory
from dcl.paperstudy.training import initialize, weights_hash
from dcl.universal.model import UniversalColorGNN, count_parameters
from dcl.universal.rules import available_matrix, exact_commit_labels


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def inputs(k=4, isolated=False):
    colors = torch.tensor([-1, 0, -1, min(2, k-1)])
    edges = torch.empty((2, 0), dtype=torch.long) if isolated else torch.tensor([[0,1,0,2,0,3],[1,0,2,0,3,0]])
    return colors, torch.tensor([.2,.4,.8,.1]), torch.tensor([.1,.8,.9,.3]), edges, torch.full((4,), k)


def test_isolated_normalization_same_tensors_and_only_feature_change():
    norm, _, order1, cf1 = initialize(Run("normalized", 41000), "cpu")
    raw, _, order2, cf2 = initialize(Run("raw_endpoints", 41000), "cpu")
    assert count_parameters(norm) == count_parameters(raw) == 16035
    assert weights_hash(norm.state_dict()) == weights_hash(raw.state_dict())
    assert np.array_equal(order1.permutation(111), order2.permutation(111))
    assert cf1.bit_generator.state == cf2.bit_generator.state
    captures = []
    for model in (norm, raw):
        model.scorer.register_forward_pre_hook(lambda m, args: captures.append(args[0].detach().clone()))
        model(*inputs())
    a, b = captures
    torch.testing.assert_close(a[..., :33], b[..., :33], rtol=0, atol=0)
    torch.testing.assert_close(a[..., -1], b[..., -1], rtol=0, atol=0)
    scale = a[..., 32].sum(1).clamp_min(.5)
    torch.testing.assert_close(a[..., 33:35] * scale[:, None, None], b[..., 33:35])
    assert not torch.allclose(a[..., 33:35], b[..., 33:35])
    for module in ("message2", "state2", "commit_head", "participation"):
        assert weights_hash(getattr(norm, module).state_dict()) == weights_hash(getattr(raw, module).state_dict())


def test_original_normalized_exact_forward_and_gradients():
    torch.manual_seed(91)
    original = UniversalColorGNN()
    model = make_model("normalized")
    model.load_state_dict(original.state_dict())
    a, b = original(*inputs()), model(*inputs())
    for key in ("candidate_logits", "commit_logit", "availability", "lo", "hi", "participate", "candidate"):
        assert torch.equal(getattr(a, key), getattr(b, key)), key
    for m, out in ((original, a), (model, b)):
        (out.candidate_logits.sum() + out.commit_logit.sum() + out.participation_logit.sum()).backward()
    for (name, pa), (name_b, pb) in zip(original.named_parameters(), model.named_parameters()):
        assert name == name_b
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)


def test_neighbor_reductions_multiplicity_scale_isolation_and_empty():
    z = torch.tensor([[1., 3.], [2., 1.], [0., 0.]])
    edges = torch.tensor([[0,1,0],[2,2,2]])
    assert torch.equal(neighbor_reduce(z, edges, "max")[2], torch.tensor([2.,3.]))
    assert torch.equal(neighbor_reduce(z, edges, "sum")[2], torch.tensor([4.,7.]))
    torch.testing.assert_close(neighbor_reduce(z, edges, "mean")[2], torch.tensor([4/3,7/3]))
    for kind in ("max","sum","mean"):
        assert not neighbor_reduce(z, edges, kind)[0].any()
        torch.testing.assert_close(neighbor_reduce(z, edges, kind), neighbor_reduce(z, edges, kind, 1))
        assert not neighbor_reduce(z, torch.empty((2,0),dtype=torch.long), kind).any()
    doubled = torch.cat((edges, edges), 1)
    assert torch.equal(neighbor_reduce(z, edges, "max"), neighbor_reduce(z, doubled, "max"))
    torch.testing.assert_close(neighbor_reduce(z, edges, "mean"), neighbor_reduce(z, doubled, "mean"))
    torch.testing.assert_close(2*neighbor_reduce(z, edges, "sum"), neighbor_reduce(z, doubled, "sum"))


@pytest.mark.parametrize("kind", PRIMARY)
@pytest.mark.parametrize("k", (4,11,201))
def test_all_models_variable_palette_no_oracle_gradients_and_isolated(kind, k, monkeypatch):
    import dcl.universal.rules as rules
    def forbidden(*a, **kw):
        raise AssertionError("oracle invoked from model")
    monkeypatch.setattr(rules, "available_matrix", forbidden)
    monkeypatch.setattr(rules, "teacher_offer", forbidden)
    model = make_model(kind)
    parameter_count = count_parameters(model)
    x = inputs(k, isolated=True)
    out = model(*x)
    assert count_parameters(model) == parameter_count
    assert out.candidate_logits.shape == (4,k)
    assert torch.isfinite(out.candidate_logits).all() and torch.isfinite(out.commit_logit).all()
    loss = out.candidate_logits.sum() + out.commit_logit.sum() + out.participation_logit.sum()
    loss.backward()
    # No edges means messages are not consumed and have no gradient path.
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    out = model(*inputs(k))
    (out.candidate_logits.sum() + out.commit_logit.sum() + out.participation_logit.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("kind", PRIMARY)
def test_padding_matches_individual_and_node_equivariance(kind):
    model = make_model(kind).eval()
    a, b = inputs(4), inputs(11)
    combined = [torch.cat((a[j],b[j])) for j in (0,1,2)]
    combined.extend((torch.cat((a[3],b[3]+4),1), torch.cat((a[4],b[4]))))
    with torch.inference_mode():
        joint, out = model(*combined), model(*a)
    torch.testing.assert_close(joint.candidate_logits[:4,:4], out.candidate_logits, atol=2e-5, rtol=2e-5)
    assert (joint.candidate_logits[:4,4:] < -1e8).all()
    perm = torch.tensor([2,0,3,1])
    inverse = torch.argsort(perm)
    x = (a[0][perm], a[1][perm], a[2][perm], inverse[a[3]], a[4][perm])
    with torch.inference_mode():
        relabeled = model(*x)
    torch.testing.assert_close(relabeled.candidate_logits, out.candidate_logits[perm])
    torch.testing.assert_close(relabeled.commit_logit, out.commit_logit[perm])


@pytest.mark.parametrize("kind", ("deepsets", "transformer"))
def test_prefix_free_scores_do_not_call_cumsum(kind, monkeypatch):
    model = make_model(kind)
    h = torch.randn(2,11,32)
    alpha = torch.rand(2,11)
    def forbidden(*args, **kwargs):
        raise AssertionError("prefix used in palette decisions")
    monkeypatch.setattr(torch, "cumsum", forbidden)
    monkeypatch.setattr(torch.Tensor, "cumsum", forbidden)
    out = model.palette_scores(h, alpha, torch.rand(2), torch.full((2,),11), torch.ones(2,11,dtype=torch.bool))
    assert out.shape == (2,11)


def test_cyclic_teacher_distribution_differs_and_stays_safe():
    # Center has available {0,2,3}; the cyclic gap gives (1,2,1)/4.
    colors = np.array([-1,1,-1,-1])
    edges = inputs()[3].numpy()
    rc = np.zeros(4)
    selected = []
    for value in (.125,.375,.625,.875):
        rc[0] = value
        offer, _ = cyclic_offer(colors, rc, edges, 4)
        selected.append(offer[0])
    law = cyclic_law(np.array([1,0,1,1],bool))
    np.testing.assert_allclose(law, [1/4,0,1/2,1/4])
    np.testing.assert_allclose(np.bincount(selected,minlength=4)/4,law)
    assert not np.allclose(law[[0,2,3]],1/3)
    for cyclic in (False,True):
        examples = teacher_trajectory(4, edges, 4, 19, cyclic)
        final = examples[-1].colors.copy()
        for e in examples:
            valid = available_matrix(e.colors, e.edge_index, e.palette)
            mask = e.colors < 0
            assert valid[np.arange(4)[mask],e.y_offer[mask]].all()
            final[e.y_commit > .5] = e.y_offer[e.y_commit > .5]
        assert (final >= 0).all()
        src,dst = edges
        assert (final[src] != final[dst]).all()


def test_confirmatory_plan_pairing_and_no_historical_streams():
    cases = cohort_plan()
    assert len(cases) == 304
    assert len({c["case_id"] for c in cases}) == 304
    for case in cases:
        assert case["n"] > case["degree"]
        assert case["namespace"] == 202610210211
    assert len({c["graph_id"] for c in cases}) == 288


def test_exact_resume_and_pairing_on_real_training(tmp_path, monkeypatch):
    from dcl.paperstudy.training import train
    def tiny(self):
        from dcl.universal.config import UConfig
        return UConfig(seed=self.seed,graphs_per_degree=(3,),tapes=1,epochs=3,warmup=1,
                       val_graphs_per_degree=2,val_rollouts=1,val_sizes=(40,),val_every=1,
                       max_rounds=8,batch_size=8,checkpoint_epochs=(1,3))
    monkeypatch.setattr(Run,"training_config",tiny)
    run = Run("normalized",41000)
    train(run,tmp_path/"full",device="cpu")
    train(run,tmp_path/"resumed",device="cpu",stop_after_epoch=1)
    train(run,tmp_path/"resumed",device="cpu")
    a = torch.load(tmp_path/"full"/"runs"/run.id/"last.pt",weights_only=False)
    b = torch.load(tmp_path/"resumed"/"runs"/run.id/"last.pt",weights_only=False)
    assert weights_hash(a["model_state_dict"]) == weights_hash(b["model_state_dict"])
    assert a["updates"] == b["updates"]
    raw = Run("raw_endpoints",41000)
    train(raw,tmp_path/"full",device="cpu")
    c = torch.load(tmp_path/"full"/"runs"/raw.id/"last.pt",weights_only=False)
    assert a["data_hash"] == c["data_hash"] and a["initial_state_hash"] == c["initial_state_hash"]
    assert a["updates"] == c["updates"]
    for ha,hc in zip(a["history"],c["history"]):
        for key in ("minibatch_order_hash","counterfactual_rng_start_hash","counterfactual_rng_end_hash"):
            assert ha[key] == hc[key]
