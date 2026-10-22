"""Frozen historical checkpoints; read-only CPU integration, no training."""
import json
from pathlib import Path
import numpy as np
import torch
from dcl.globaldist.core import *
from dcl.globaldist.sampling import sample
from dcl.paperstudy.training import load_model
from dcl.localpalettes.adapter import LocalPaletteAdapter

def test_all_fifteen_checkpoint_identities_actions_reset():
    torch.set_num_threads(1)
    previous=json.loads(Path('/workspaces/AlgoLearning/ucgnn-local-palette-study/results/ucgnn-local-palettes/frozen-checkpoints.json').read_text())
    previous={s['label']:s for s in previous}
    selected=specs();assert len(selected)==15
    for s in selected:
        assert s['sha256']==previous[s['label']]['sha256']
        assert s['seed'] in range(41000,41005) and s['epoch']==120
        model,_=load_model(s['checkpoint'],'cpu');assert not model.training
        colors=torch.tensor([-1,5,-1,0,-1,1,-1]);palette=torch.tensor([3,6,2,4,7,2,1])
        edges=torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]])
        rp=torch.full((7,),.3);rc=torch.linspace(.07,.93,7)
        with torch.inference_mode():old=model(colors,rp,rc,edges,palette);new=LocalPaletteAdapter(model)(colors,rp,rc,edges,palette)
        assert torch.equal(old.candidate,new.candidate)
        assert torch.equal(old.participate,new.participate)
        assert torch.equal(old.commit_logit>0,new.commit_logit>0)
        # Actual frozen coefficients: vertex equivariance on a legal partial clique.
        ce=torch.as_tensor(edge_array(nx.complete_graph(5)))
        cc=torch.tensor([-1,-1,0,-1,2]);cp=torch.full((5,),5)
        crp=torch.tensor([.13,.39,.55,.77,.91]);crc=torch.tensor([.17,.31,.53,.72,.89])
        order=torch.tensor([3,1,4,0,2]);inverse=order.argsort()
        adapter=LocalPaletteAdapter(model)
        base=adapter(cc,crp,crc,ce,cp)
        perm=adapter(cc[order],crp[order],crc[order],inverse[ce],cp[order])
        assert torch.equal(base.candidate,perm.candidate[inverse])
        assert torch.equal(base.participate,perm.participate[inverse])
        assert torch.equal(base.commit_logit>0,(perm.commit_logit>0)[inverse])
        torch.testing.assert_close(base.candidate_logits,perm.candidate_logits[inverse],atol=3e-5,rtol=3e-5)
        g=small_graphs()[1]
        a=sample(g,model,s['label'],'integration-reset',8,batch_nodes=12)
        sample(small_graphs()[3],model,s['label'],'integration-interleave',2)
        b=sample(g,model,s['label'],'integration-reset',8,batch_nodes=24)
        assert np.array_equal(a['colors'],b['colors'])
        assert not a['status'].any()
