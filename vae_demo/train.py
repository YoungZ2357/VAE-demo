"""
Train β-VAEs with a 2-D latent on MNIST.

Usage:
    python -m vae_demo.train --betas 0.25 1 4 --epochs 30
    python -m vae_demo.train --conditional --beta-min 0.05 --beta-max 16 --epochs 60

Outputs (under --out, default ./runs):
    beta_<b>/vae.pt, history.json, eval.npz   one plain run per β (see runs.py)
    beta_<b>/*.png, grid_uint8.npy            first-pass figures (with --figures)
    cbeta/...                                 the β-conditional run (with --conditional)
    compare.png, beta_summary.png             cross-β figures
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .conditioning import make_cond, sample_log_uniform
from .data import get_device, get_loaders
from .model import VAE, vae_loss
from .runs import Run, list_runs, load_run, save_run


def train_epoch(model, loader, opt, beta, device):
    """beta: a float, or a callable n -> [n] tensor of per-sample β's (conditional VAE)."""
    model.train()
    tot, n = np.zeros(3), 0
    for x, _ in loader:
        x = x.to(device)
        b = x.size(0)
        if callable(beta):
            beta_i = beta(b).to(device)
            logits, mu, logvar = model(x, model.cond_vec(beta_i))
            loss, recon, kl, _ = vae_loss(logits, x, mu, logvar, beta_i)
        else:
            logits, mu, logvar = model(x)
            loss, recon, kl, _ = vae_loss(logits, x, mu, logvar, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += np.array([loss.item(), recon.item(), kl.item()]) * b
        n += b
    return dict(zip(["loss", "recon", "kl"], (tot / n).tolist()))


@torch.no_grad()
def evaluate(model, loader, beta, device):
    """Returns test metrics plus posterior means μ(x) and labels for plotting.
    For a conditional model, the model is conditioned on this β."""
    model.eval()
    c = model.cond_vec(beta)
    tot, kl_dims, n = np.zeros(3), 0.0, 0
    mus, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits, mu, logvar = model(x, c)  # sampled z: single-sample ELBO estimate
        loss, recon, kl, kld = vae_loss(logits, x, mu, logvar, beta)
        b = x.size(0)
        tot += np.array([loss.item(), recon.item(), kl.item()]) * b
        kl_dims = kl_dims + kld.cpu().numpy() * b
        n += b
        mus.append(mu.cpu())
        labels.append(y)
    metrics = dict(zip(["loss", "recon", "kl"], (tot / n).tolist()))
    metrics["kl_per_dim"] = (kl_dims / n).tolist()  # a dim near 0 => that dim collapsed
    return metrics, torch.cat(mus).numpy(), torch.cat(labels).numpy()


def default_eval_betas(cond, k=9):
    return np.geomspace(cond["beta_min"], cond["beta_max"], k).round(4).tolist()


def evaluate_run(run: Run, test_loader, eval_betas=None):
    """Fill run.latents / run.metrics / run.labels. Plain runs: at their own β.
    Conditional runs: at `eval_betas` (default: 9 log-spaced β's across the trained range)."""
    device = next(run.model.parameters()).device
    if run.kind == "plain":
        eval_betas = [run.beta]
    elif eval_betas is None:
        eval_betas = default_eval_betas(run.cond)
    for b in eval_betas:
        m, mu, labels = evaluate(run.model, test_loader, b, device)
        run.metrics[b], run.latents[b], run.labels = m, mu, labels
    return run


def _fit(model, train_loader, test_loader, beta_train, beta_eval, epochs, lr, device, log):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, opt, beta_train, device)
        te, _, _ = evaluate(model, test_loader, beta_eval, device)
        history.append({"epoch": epoch, "train": tr, "test": te})
        if log:
            kd = ", ".join(f"{v:.2f}" for v in te["kl_per_dim"])
            log(f"ep {epoch:3d} | train loss {tr['loss']:7.2f} | test recon {te['recon']:7.2f} "
                f"KL {te['kl']:5.2f} [{kd}] | {time.time() - t0:4.1f}s")
    return history


def train_beta(beta, train_loader, test_loader, epochs=30, lr=1e-3, seed=0,
               device=None, hidden_dims=(512, 256), log=print):
    """Train one plain β-VAE. Same seed for every β: identical init + data order, so
    differences come from β (it does NOT guarantee the latent layouts line up)."""
    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VAE(hidden_dims=hidden_dims, latent_dim=2).to(device)
    if log:
        log(f"\n=== β = {beta:g} ===")
    history = _fit(model, train_loader, test_loader, beta, beta, epochs, lr, device, log)
    return evaluate_run(Run(model=model, kind="plain", beta=beta, history=history), test_loader)


def train_conditional(train_loader, test_loader, beta_min=0.05, beta_max=16.0, epochs=60,
                      lr=1e-3, seed=0, device=None, hidden_dims=(512, 256), eval_betas=None,
                      log=print):
    """Train one VAE conditioned on β, with β ~ log-uniform[beta_min, beta_max] per sample."""
    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cond = make_cond(beta_min, beta_max)
    model = VAE(hidden_dims=hidden_dims, latent_dim=2, cond=cond).to(device)
    g = torch.Generator().manual_seed(seed)
    sampler = lambda n: sample_log_uniform(n, beta_min, beta_max, g)  # noqa: E731
    beta_mid = float(np.sqrt(beta_min * beta_max))  # per-epoch monitoring only
    if log:
        log(f"\n=== conditional, β ∈ [{beta_min:g}, {beta_max:g}] ===")
    history = _fit(model, train_loader, test_loader, sampler, beta_mid, epochs, lr, device, log)
    run = Run(model=model, kind="conditional", history=history)
    return evaluate_run(run, test_loader, eval_betas)


def save_run_figures(run: Run, test_loader, run_dir, lim=3.0, grid_n=50):
    from . import viz
    run_dir = Path(run_dir)
    for b in run.eval_betas:
        c = run.model.cond_vec(b)
        sfx = "" if run.kind == "plain" else f"_beta_{b:g}"
        tag = f"β={b:g}"
        viz.save_latent_plot(run.latents[b], run.labels, run_dir / f"latent{sfx}.png",
                             f"{tag}: test-set μ(x)")
        viz.save_manifold_plot(run.model, run_dir / f"manifold{sfx}.png",
                               f"{tag}: decoded latent grid", lim=lim, c=c)
        viz.save_recon_plot(run.model, test_loader, run_dir / f"recon{sfx}.png", tag, c=c)
        viz.save_prior_samples(run.model, run_dir / f"prior_samples{sfx}.png", tag, c=c)
        if run.kind == "plain":
            viz.export_grid(run.model, run_dir / "grid_uint8.npy", n=grid_n, lim=lim)


def train_sweep(betas, out="runs", epochs=30, lr=1e-3, seed=0, device=None,
                data_dir="./data", batch_size=128, num_workers=2, skip_existing=True,
                figures=False, log=print):
    """Train (or load, if `out/beta_<b>/vae.pt` exists and skip_existing) one run per β.
    Returns {β: Run}."""
    device = device or get_device()
    out = Path(out)
    loaders = None

    def get():
        nonlocal loaders
        if loaders is None:
            loaders = get_loaders(data_dir, batch_size, num_workers, device.type == "cuda")
        return loaders

    runs = {}
    for beta in betas:
        run_dir = out / f"beta_{beta:g}"
        if skip_existing and (run_dir / "vae.pt").exists():
            needs_eval = not (run_dir / "eval.npz").exists()
            runs[beta] = load_run(run_dir, device, get()[1] if needs_eval else None)
            if log:
                log(f"loaded {run_dir}")
            continue
        train_loader, test_loader = get()
        run = train_beta(beta, train_loader, test_loader, epochs, lr, seed, device, log=log)
        save_run(run, run_dir)
        if figures:
            save_run_figures(run, test_loader, run_dir)
        runs[beta] = run
    return dict(sorted(runs.items()))


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # duplicate OpenMP runtimes on Windows/conda
    import matplotlib
    matplotlib.use("Agg")
    from . import viz

    p = argparse.ArgumentParser()
    p.add_argument("--betas", type=float, nargs="+", default=[0.25, 1.0, 4.0])
    p.add_argument("--conditional", action="store_true",
                   help="train one β-conditional VAE instead of a sweep")
    p.add_argument("--beta-min", type=float, default=0.05)
    p.add_argument("--beta-max", type=float, default=16.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default="./runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--retrain", action="store_true", help="retrain even if a checkpoint exists")
    p.add_argument("--figures", action="store_true", help="also write per-run png figures")
    args = p.parse_args()

    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    if args.conditional:
        train_loader, test_loader = get_loaders(args.data_dir, args.batch_size,
                                                args.num_workers, device.type == "cuda")
        run = train_conditional(train_loader, test_loader, args.beta_min, args.beta_max,
                                args.epochs, args.lr, args.seed, device)
        save_run(run, out / "cbeta")
        if args.figures:
            save_run_figures(run, test_loader, out / "cbeta")
        viz.save_beta_summary(run, out / "cbeta" / "beta_summary.png")
        print(json.dumps({f"{b:g}": {k: round(v, 2) for k, v in m.items() if k != "kl_per_dim"}
                          for b, m in run.metrics.items()}, indent=1))
    else:
        runs = train_sweep(args.betas, out, args.epochs, args.lr, args.seed, device,
                           args.data_dir, args.batch_size, args.num_workers,
                           skip_existing=not args.retrain, figures=args.figures)
        viz.save_compare_plot(runs, out / "compare.png")
        if len(runs) > 1:
            viz.save_beta_summary(list_runs(out, device), out / "beta_summary.png")
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
