"""β → condition features for the β-conditional VAE.

The same formula is re-implemented in static/latent_pad.js (`betaFeatures`), driven by the
`cond` dict stored in the checkpoint and the exported manifest. Keep the two in sync.
"""
import math

import torch


def make_cond(beta_min, beta_max):
    """log β, centred and scaled so [beta_min, beta_max] maps to [-1, 1]."""
    lo, hi = math.log(beta_min), math.log(beta_max)
    return {"type": "log_beta", "center": (lo + hi) / 2, "scale": (hi - lo) / 2,
            "beta_min": beta_min, "beta_max": beta_max, "dim": 1}


def beta_features(beta, cond):
    """beta: float or [B] tensor -> [1, 1] or [B, 1] float tensor."""
    assert cond["type"] == "log_beta", cond["type"]
    b = torch.as_tensor(beta, dtype=torch.float32).reshape(-1, 1)
    return (torch.log(b) - cond["center"]) / cond["scale"]


def sample_log_uniform(n, beta_min, beta_max, generator=None):
    u = torch.rand(n, generator=generator)
    return torch.exp(math.log(beta_min) + u * (math.log(beta_max) - math.log(beta_min)))
