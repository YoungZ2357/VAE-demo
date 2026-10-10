"""Static matplotlib figures. `c` is the condition vector for a β-conditional model
(model.cond_vec(β)); leave it None for a plain VAE."""
import matplotlib.pyplot as plt
import numpy as np
import torch


def _device(model):
    return next(model.parameters()).device


@torch.no_grad()
def decode_grid(model, n, lim, c=None):
    """Decode an n×n regular grid over [-lim, lim]². Row 0 = top = largest z2."""
    model.eval()
    z1 = np.linspace(-lim, lim, n)
    z2 = np.linspace(lim, -lim, n)
    zz = np.array([[a, b] for b in z2 for a in z1], dtype=np.float32)
    imgs = model.decode(torch.from_numpy(zz).to(_device(model)), c).cpu().numpy()
    return imgs.reshape(n, n, 28, 28)


def tile(imgs):
    """[rows, cols, 28, 28] -> one 2-D canvas."""
    r, c, h, w = imgs.shape
    return imgs.transpose(0, 2, 1, 3).reshape(r * h, c * w)


def draw_latent(ax, mu, labels, lim):
    sc = ax.scatter(mu[:, 0], mu[:, 1], c=labels, cmap="tab10", s=2, alpha=0.6,
                    vmin=-0.5, vmax=9.5, rasterized=True)
    # For a 2-D standard normal, radius r encloses 1 - exp(-r²/2) of the mass.
    for r in (1, 2, 3):
        mass = 1 - np.exp(-r ** 2 / 2)
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls="--", lw=0.8, color="k", alpha=0.6))
        ax.text(r * 0.707, r * 0.707, f"{mass:.0%}", fontsize=7, alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    return sc


def draw_manifold(ax, model, n, lim, c=None):
    canvas = tile(decode_grid(model, n, lim, c))
    # extent puts the image in latent coordinates, so it lines up with the scatter
    ax.imshow(canvas, cmap="gray", extent=[-lim, lim, -lim, lim])
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")


@torch.no_grad()
def draw_recon(ax, model, x, c=None):
    """x: [n, 1, 28, 28] batch. Top row: input, bottom row: recon from μ (deterministic)."""
    model.eval()
    n = x.size(0)
    x = x.to(_device(model))
    mu, _ = model.encode(x, c)
    x_hat = model.decode(mu, c)
    imgs = torch.stack([x.view(n, 28, 28), x_hat.view(n, 28, 28)]).cpu().numpy()
    ax.imshow(tile(imgs), cmap="gray")
    ax.axis("off")


@torch.no_grad()
def draw_prior_samples(ax, model, n=10, seed=0, c=None):
    model.eval()
    g = torch.Generator().manual_seed(seed)  # same z's for every β → fair comparison
    z = torch.randn(n * n, 2, generator=g).to(_device(model))
    imgs = model.decode(z, c).view(n, n, 28, 28).cpu().numpy()
    ax.imshow(tile(imgs), cmap="gray")
    ax.axis("off")


def _save(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_latent_plot(mu, labels, path, title, lim=4):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = draw_latent(ax, mu, labels, lim)
    fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")
    ax.set_title(title)
    _save(fig, path)


def save_manifold_plot(model, path, title, n=20, lim=3, c=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_manifold(ax, model, n, lim, c)
    ax.set_title(title)
    _save(fig, path)


def save_recon_plot(model, loader, path, title, n=12, c=None):
    x, _ = next(iter(loader))
    fig, ax = plt.subplots(figsize=(n * 0.8, 2))
    draw_recon(ax, model, x[:n], c)
    ax.set_title(title + "   (top: input, bottom: recon from μ)")
    _save(fig, path)


def save_prior_samples(model, path, title, n=10, seed=0, c=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_prior_samples(ax, model, n, seed, c)
    ax.set_title(title + "   z ~ N(0, I)")
    _save(fig, path)


def plot_compare(runs, lim=3, n=15):
    """runs: {β: Run} of plain VAEs. Top: test-set μ(x); bottom: decoder over the same window."""
    betas = sorted(runs)
    fig, axes = plt.subplots(2, len(betas), figsize=(4.2 * len(betas), 8.4), squeeze=False)
    for j, b in enumerate(betas):
        r = runs[b]
        m = r.metrics[b]
        draw_latent(axes[0, j], r.latents[b], r.labels, lim)
        axes[0, j].set_title(f"β={b:g}  recon={m['recon']:.1f}  KL={m['kl']:.2f}")
        draw_manifold(axes[1, j], r.model, n, lim)
    fig.suptitle("Top: posterior means μ(x) on test set   Bottom: decoder over the same window")
    fig.tight_layout()
    return fig


def plot_prior_samples(runs, n=8, seed=0):
    """Decodes of the same z ~ N(0, I) under every β, side by side."""
    betas = sorted(runs)
    fig, axes = plt.subplots(1, len(betas), figsize=(3 * len(betas), 3.3), squeeze=False)
    for ax, b in zip(axes[0], betas):
        draw_prior_samples(ax, runs[b].model, n, seed)
        ax.set_title(f"β={b:g}")
    fig.suptitle("z ~ N(0, I), same samples for every β")
    fig.tight_layout()
    return fig


def plot_beta_summary(runs):
    """Rate–distortion: test recon / KL vs β. Accepts {β: Run} or one conditional Run."""
    if hasattr(runs, "metrics"):
        metrics = runs.metrics
    else:
        metrics = {b: r.metrics[b] for b, r in runs.items()}
    betas = sorted(metrics)
    recon = [metrics[b]["recon"] for b in betas]
    kl = [metrics[b]["kl"] for b in betas]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(betas, recon, "o-", color="tab:blue", label="recon (BCE, nats)")
    ax1.set_xscale("log")
    ax1.set_xlabel("β")
    ax1.set_ylabel("test recon", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(betas, kl, "s-", color="tab:red", label="KL (nats)")
    ax2.set_ylabel("test KL", color="tab:red")
    ax1.set_title("Rate–distortion trade-off across β")
    fig.tight_layout()
    return fig


def save_compare_plot(runs, path, lim=3, n=15):
    fig = plot_compare(runs, lim, n)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_beta_summary(runs, path):
    _save(plot_beta_summary(runs), path)


@torch.no_grad()
def export_grid(model, path, n=50, lim=3, c=None):
    """Precomputed decodes over the latent window (≈2 MB at n=50), uint8."""
    grid = decode_grid(model, n, lim, c)
    np.save(path, (grid * 255).round().astype(np.uint8))
