"""2-D latent β-VAE on MNIST with an interactive latent pad.

Typical notebook use:

    from vae_demo import train_sweep, LatentPad
    runs = train_sweep([0.25, 1, 4], out="runs")   # loads existing runs/beta_*/vae.pt
    LatentPad.from_runs(runs)
"""
from .conditioning import beta_features, make_cond
from .data import get_device, get_loaders
from .export import build_bundle, export_bundle
from .model import VAE, vae_loss
from .runs import Run, list_runs, load_run, save_run
from .train import (evaluate, evaluate_run, save_run_figures, train_beta, train_conditional,
                    train_epoch, train_sweep)
from .widget import LatentPad

__all__ = [
    "VAE", "vae_loss", "beta_features", "make_cond", "get_device", "get_loaders",
    "Run", "list_runs", "load_run", "save_run", "evaluate", "evaluate_run", "train_epoch",
    "train_beta", "train_conditional", "train_sweep", "save_run_figures",
    "build_bundle", "export_bundle", "LatentPad",
]
