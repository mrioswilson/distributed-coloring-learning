"""Explicit relational priority adapter; no teacher routine in inference."""
import torch

from dcl.integration.models import RootedPalettePolicy
from dcl.multiteacher.programs import KW


def relational_arrays(arrays, teacher, phase):
    x, y, relations, masks = arrays
    if teacher in ("JP", "SPECULATIVE_GM"):
        x, y = x.clone(), y.clone()
        # Each center receives edge order, without dependence on ID spacing.
        relation = torch.sign(y[..., 5] - x[..., None, 5])
        relative = torch.sign(x[..., 5] - x[:, :1, 5])
        x[..., 5] = relative
        y[..., 5] = relative[..., None] + relation
    if phase == KW:
        x = x.clone()
        x[..., 18] = torch.sign(x[..., 18])
    return x, y, relations, masks


class QueryPolicy(RootedPalettePolicy):
    def first(self, arrays, phase):
        return super().first(relational_arrays(arrays, self.teacher, phase), phase)
