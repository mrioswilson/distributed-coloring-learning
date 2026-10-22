import itertools
from dcl.sizedegree.capacity import bound

def test_prefix_list_bound_for_every_assignment_on_small_lists():
    for degrees in ([1,2,1],[2,2,2],[1,4,1,1,1],[2,5,2,2,2,2]):
        upper=bound(degrees)['used_color_upper_bound']
        assert all(len(set(x))<=upper for x in itertools.product(*(range(d+1) for d in degrees)))

def test_heterogeneous_support_forces_primary_occupancy_event():
    assert bound([100]+[33]*44+[2]*83)['A_deterministically_true']
    assert bound([100]*10+[33]*358+[2]*656)['used_color_upper_bound']==44
    assert not bound([100]*128)['A_deterministically_true']
