"""A run = one trained model + its evaluation, saved as a directory:

    vae.pt       {state_dict, kind, beta | cond, latent_dim, hidden_dims}
    history.json per-epoch train/test metrics
    eval.npz     test-set posterior means μ(x) and recon/KL at one or more evaluation β's

Plain runs (`kind="plain"`) are trained at a single β and evaluated at that β.
Conditional runs (`kind="conditional"`) take β as input and are evaluated at several β's.
Checkpoints without a `kind` key (the original reference runs) are treated as plain.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .model import VAE


@dataclass
class Run:
    model: VAE
    kind: str = "plain"                          # "plain" | "conditional"
    beta: float | None = None                    # plain only
    history: list = field(default_factory=list)
    latents: dict = field(default_factory=dict)  # eval β -> μ(x) [N, 2]
    metrics: dict = field(default_factory=dict)  # eval β -> {"recon", "kl", ...}
    labels: np.ndarray | None = None             # [N] digit labels for `latents`
    path: Path | None = None

    @property
    def cond(self):
        return self.model.cond

    @property
    def eval_betas(self):
        return sorted(self.metrics)

    @property
    def name(self):
        if self.kind == "plain":
            return f"beta_{self.beta:g}"
        return self.path.name if self.path else "cbeta"

    def __repr__(self):
        b = f"β={self.beta:g}" if self.kind == "plain" else \
            f"β∈[{self.cond['beta_min']:g}, {self.cond['beta_max']:g}]"
        return f"Run({self.kind}, {b}, evaluated at {len(self.metrics)} β)"


def save_run(run: Run, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    m = run.model
    ckpt = {"state_dict": m.state_dict(), "kind": run.kind, "latent_dim": m.latent_dim,
            "hidden_dims": m.hidden_dims, "input_dim": m.input_dim}
    if run.kind == "plain":
        ckpt["beta"] = run.beta
    else:
        ckpt["cond"] = m.cond
    torch.save(ckpt, run_dir / "vae.pt")
    (run_dir / "history.json").write_text(json.dumps(run.history, indent=2))
    save_eval(run, run_dir)
    run.path = run_dir


def save_eval(run: Run, run_dir):
    if not run.metrics:
        return
    betas = run.eval_betas
    np.savez_compressed(
        Path(run_dir) / "eval.npz",
        betas=np.array(betas, dtype=np.float64),
        mu=np.stack([run.latents[b] for b in betas]).astype(np.float32),
        labels=run.labels.astype(np.uint8),
        metrics=json.dumps({repr(b): run.metrics[b] for b in betas}),
    )


def load_run(run_dir, device="cpu", test_loader=None):
    """Load a run directory. If eval.npz is missing it is computed (needs `test_loader`,
    otherwise the MNIST test set under ./data is loaded) and written back."""
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "vae.pt", map_location=device, weights_only=False)
    kind = ckpt.get("kind", "plain")
    model = VAE(input_dim=ckpt.get("input_dim", 784), hidden_dims=tuple(ckpt["hidden_dims"]),
                latent_dim=ckpt["latent_dim"], cond=ckpt.get("cond"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    hist_path = run_dir / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    run = Run(model=model, kind=kind, beta=ckpt.get("beta"), history=history, path=run_dir)

    ev = run_dir / "eval.npz"
    if ev.exists():
        d = np.load(ev)
        metrics = json.loads(str(d["metrics"]))
        for b, mu in zip(d["betas"].tolist(), d["mu"]):
            run.latents[b] = mu
            run.metrics[b] = metrics[repr(b)]
        run.labels = d["labels"]
    else:
        from .train import evaluate_run  # local import: train imports this module
        if test_loader is None:
            from .data import get_loaders
            _, test_loader = get_loaders(num_workers=0)
        evaluate_run(run, test_loader)
        save_eval(run, run_dir)
    return run


def list_runs(out="runs", device="cpu", test_loader=None):
    """All plain runs under `out` as {β: Run}, sorted by β."""
    runs = {}
    for d in sorted(Path(out).glob("beta_*")):
        if (d / "vae.pt").exists():
            r = load_run(d, device, test_loader)
            runs[r.beta] = r
    return dict(sorted(runs.items()))
