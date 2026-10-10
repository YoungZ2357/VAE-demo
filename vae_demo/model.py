import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import beta_features


class VAE(nn.Module):
    """MLP VAE. With `cond` (see conditioning.make_cond), both encoder and decoder take a
    condition vector c = beta_features(β, cond) concatenated to their input: the β-conditional VAE."""

    def __init__(self, input_dim=784, hidden_dims=(512, 256), latent_dim=2, cond=None):
        super().__init__()
        h1, h2 = hidden_dims
        cond_dim = cond["dim"] if cond else 0
        self.input_dim, self.hidden_dims = input_dim, tuple(hidden_dims)
        self.latent_dim, self.cond, self.cond_dim = latent_dim, cond, cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)
        self.decoder = nn.Sequential(  # outputs logits
            nn.Linear(latent_dim + cond_dim, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def cond_vec(self, beta):
        """Condition tensor for β (float or [B] tensor), or None for an unconditional model."""
        if not self.cond:
            return None
        return beta_features(beta, self.cond).to(next(self.parameters()).device)

    @staticmethod
    def _cat(a, c):
        return a if c is None else torch.cat([a, c.expand(a.size(0), -1)], dim=1)

    def encode(self, x, c=None):
        h = self.encoder(self._cat(x.view(x.size(0), -1), c))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode_logits(self, z, c=None):
        return self.decoder(self._cat(z, c))

    def decode(self, z, c=None):
        """Pixel probabilities in [0, 1] — use this for visualization / demo."""
        return torch.sigmoid(self.decode_logits(z, c))

    def forward(self, x, c=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode_logits(z, c), mu, logvar  # logits


def vae_loss(logits, x, mu, logvar, beta=1.0):
    """Sum over pixels / latent dims, mean over batch.

    beta may be a scalar or a [B] tensor (per-sample β, for the conditional VAE).
    Returned recon / kl are unweighted batch means."""
    x = x.view(x.size(0), -1)
    b = x.size(0)
    recon_i = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)  # [B]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
    kl_i = kl_per_dim.sum(1)
    loss = (recon_i + beta * kl_i).sum() / b
    return loss, recon_i.sum() / b, kl_i.sum() / b, kl_per_dim.mean(0)
