"""Palette architectures on Pedro's unchanged native state/action interface.

Native schedules, polynomial evaluation and context residuals are supplied by
RootedE1 and remain explicit. These CPU models do not introduce an action shield.
"""
import torch

from dcl.multiteacher_e1.model import RootedE1
from dcl.paperstudy.models import MechanismUCGNN, LocalPaletteTransformer

ARCHITECTURES = ("normalized", "deepsets", "transformer", "small", "no_communication")


class RootedPalettePolicy(RootedE1):
    def __init__(self, teacher, architecture="normalized"):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        hidden, message = (8, 4) if architecture == "small" else (64, 32)
        super().__init__(teacher, hidden, message)
        self.architecture = architecture
        if architecture in ("deepsets", "transformer"):
            original = self.core.state_dict()
            replacement = (LocalPaletteTransformer(hidden, message) if architecture == "transformer"
                           else MechanismUCGNN("deepsets", hidden, message))
            # Keep all compatible message/availability/COMMIT initial weights.
            shared = {k: v for k, v in original.items()
                      if k in replacement.state_dict() and replacement.state_dict()[k].shape == v.shape}
            replacement.load_state_dict(shared, strict=False)
            self.core = replacement

    def _scores(self, h, valid, rc):
        if self.architecture not in ("deepsets", "transformer"):
            return super()._scores(h, valid, rc)
        shape, k, width = h.shape[:-2], h.shape[-2], h.shape[-1]
        flat_h, flat_valid = h.reshape(-1, k, width), valid.reshape(-1, k)
        alpha = torch.sigmoid(self.core.availability(flat_h).squeeze(-1)) * flat_valid
        # Only each receiver's actual palette enters its mean/attention mask.
        palettes = flat_valid.sum(-1)
        scores = self.core.palette_scores(flat_h, alpha, rc.reshape(-1), palettes, flat_valid)
        return scores.reshape(*shape, k), alpha.reshape(*shape, k)

    def first(self, arrays, phase):
        if self.architecture == "no_communication":
            x, y, relations, masks = arrays
            arrays = (x, torch.zeros_like(y), torch.zeros_like(relations), torch.zeros_like(masks))
        return super().first(arrays, phase)
