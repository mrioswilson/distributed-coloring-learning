"""A two-message GNN evaluated on complete rooted two-hop neighborhoods.

The dense rooted representation is equivalent to two rounds on the full graph;
it also makes single-vertex steps of sequential programs practical to evaluate.
Node-local polynomial evaluation and native phase/clock arithmetic are supplied
representations, not neural claims about learning finite-field arithmetic.
"""
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from .programs import DEGREE, ID_SPACE, K, LINIAL, KW, ASSIGN, DETECT, REPAIR, MAX_FIELD, polynomial_values

POLY_SLICE = slice(19, 32)
FEATURES = 42
RELATIONS = 19


@lru_cache(maxsize=2)
def polynomial_table(field):
    return polynomial_values(np.arange(ID_SPACE), field).astype(np.int8)


def features_at(state, neighbors, indices):
    n = len(state.colors)
    indices = np.asarray(indices)
    ix = np.minimum(indices, n - 1)
    shape = ix.shape
    old, aux = state.colors[ix], state.aux[ix]
    phase = np.broadcast_to(np.eye(6, dtype=np.float32)[state.phase], shape + (6,))
    colors = np.eye(K + 1, dtype=np.float32)[np.clip(old + 1, 0, K)]
    valid_colors = (old >= -1) & (old < K)
    colors[~valid_colors] = 0
    scale = max(1, state.bound)
    poly = np.zeros(shape + (MAX_FIELD,), np.int8)
    if state.phase == LINIAL:
        poly[..., :state.field] = polynomial_table(state.field)[aux]
    offsets = (state.bound - 1 - aux) * K
    # Stable positions: colors[0:5], ID[5], slot[6], permission[7], phase[8:14],
    # auxiliary[14], field[15], target[16], bound[17], aux-target[18],
    # polynomial evaluations[19:32], draws[32:34], preferences[34:38],
    # mark[38], degree[39], offset[40], phase-is-KW[41].
    def col(value):
        return np.broadcast_to(value, shape)[..., None]
    x = np.concatenate((colors,
        col(state.ids[ix] / ID_SPACE), col(state.slots[ix] / max(1, n)), col(state.permission[ix]), phase,
        col(np.clip(aux / scale, -4, 4)), col(state.field / MAX_FIELD), col(state.target / scale),
        col(state.bound / ID_SPACE), col(np.clip((aux - state.target) / scale, -4, 4)), poly / (MAX_FIELD - 1),
        state.draws[ix], state.preferences[ix] / DEGREE, col(state.marks[ix]),
        col((neighbors[indices] < n).sum(-1) / DEGREE), col(np.clip(offsets / scale, -4, 4)),
        col(state.phase == KW)), axis=-1).astype(np.float32)
    x[indices == n] = 0
    return x


def relations_at(state, centers, adjacent, center_x, adjacent_x):
    n = len(state.colors)
    a, b = np.minimum(centers, n - 1), np.minimum(adjacent, n - 1)
    eq_color = (state.colors[a, None] == state.colors[b]) & (state.colors[a, None] >= 0)
    eq_slot = state.slots[a, None] == state.slots[b]
    if state.phase not in (ASSIGN, DETECT, REPAIR):
        eq_slot[:] = False
    eq_poly = center_x[..., None, POLY_SLICE] == adjacent_x[..., POLY_SLICE]
    offsets = (state.bound - 1 - state.aux[a]) * K
    options = offsets[..., None] + np.arange(K)
    eq_option = state.aux[b, None] == options[..., None, :]
    if state.phase != LINIAL:
        eq_poly[:] = False
    if state.phase != KW:
        eq_option[:] = False
    relations = np.concatenate((eq_color[..., None], eq_slot[..., None], eq_poly, eq_option), axis=-1)
    mask = (adjacent < n) & (centers[..., None] < n)
    relations &= mask[..., None]
    return relations.astype(np.uint8)


def encode_snapshot(state, neighbors):
    centers = np.arange(len(neighbors))
    adjacent = neighbors
    x = features_at(state, neighbors, centers)
    relations = relations_at(state, centers, adjacent, x, x[adjacent])
    return x, relations


def encode_roots(state, neighbors, roots):
    roots = np.asarray(roots)
    centers = np.concatenate((roots[:, None], neighbors[roots]), axis=1)
    adjacent = neighbors[centers]
    x = features_at(state, neighbors, centers)
    y = features_at(state, neighbors, adjacent)
    relations = relations_at(state, centers, adjacent, x, y).astype(np.float32)
    return x, y, relations, adjacent < len(state.colors)


def rooted_arrays(x, relations, neighbors, roots):
    n = len(neighbors) - 1
    roots = np.asarray(roots)
    centers = np.concatenate((roots[:, None], neighbors[roots]), axis=1)
    adjacent = neighbors[centers]
    return (x[centers], x[adjacent], relations[centers].astype(np.float32), adjacent < n)


def tensor_batch(arrays):
    return tuple(torch.from_numpy(a) for a in arrays)


class LocalGNN(nn.Module):
    def __init__(self, hidden=64, message=32):
        super().__init__()
        self.message1 = nn.Sequential(nn.Linear(2 * FEATURES + RELATIONS + 1, message), nn.ReLU(),
                                      nn.Linear(message, message), nn.ReLU())
        self.node1 = nn.Sequential(nn.Linear(FEATURES + 2 * message, hidden), nn.ReLU())
        self.message2 = nn.Sequential(nn.Linear(2 * hidden + RELATIONS + 1, message), nn.ReLU(),
                                      nn.Linear(message, message), nn.ReLU())
        self.node2 = nn.Sequential(nn.Linear(hidden + 2 * message, hidden), nn.ReLU())
        self.color_head = nn.Linear(hidden, K + 1)
        self.auxiliary_head = nn.Linear(hidden, MAX_FIELD**2)
        self.mark_head = nn.Linear(hidden, 1)

    @staticmethod
    def aggregate(messages, mask):
        messages = messages * mask[..., None]
        return torch.cat((messages.amax(-2), messages.sum(-2)), dim=-1)

    def forward(self, centers, adjacent, relations, masks):
        # IDs are exactly representable /65536. Their differences supply a
        # relational scalar; no priority comparison is supplied as a boolean.
        difference = adjacent[..., 5:6] - centers[:, :, None, 5:6]
        own = centers[:, :, None, :].expand_as(adjacent)
        message1 = self.message1(torch.cat((adjacent, own, relations, difference), dim=-1))
        h = self.node1(torch.cat((centers, self.aggregate(message1, masks)), dim=-1))
        # Layer one representations for padding vertices must be ignored.
        first_neighbors = masks[:, 0]
        pair = torch.cat((h[:, 1:], h[:, :1].expand(-1, DEGREE, -1),
                          relations[:, 0], difference[:, 0]), dim=-1)
        message2 = self.message2(pair)
        root = self.node2(torch.cat((h[:, 0], self.aggregate(message2, first_neighbors)), dim=-1))
        return self.color_head(root), self.auxiliary_head(root), self.mark_head(root).squeeze(-1)
