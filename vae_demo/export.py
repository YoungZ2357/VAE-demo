"""Export runs as a framework-free "model bundle" consumed by static/latent_pad.js —
both by the notebook widget (in memory, via traitlets) and by the static site (files).

    manifest.json
    <name>/decoder.bin    decoder weights, per Linear layer: W [out, in] row-major, then b [out]
    <name>/mu_<i>.bin     float32 [n, 2] posterior means at eval β i (subsampled test set)
    <name>/labels.bin     uint8 [n]
    <name>/recon_mu_<i>.bin  float32 [k, 2] this model's encoder μ(x) for the shared recon images
    recon/images.bin      uint8 [k, 784] the same k test images for every model, as the encoders saw them
    recon/labels.bin      uint8 [k]

manifest:
    {"version": 1, "lim": 3, "latent_dim": 2, "image": [28, 28],
     "binarize": "none" | "dynamic" | "static",                 # MNIST preprocessing of every run
     "beta": {"mode": "discrete", "values": [...]}            # plain runs, snap to trained β
           | {"mode": "continuous", "min": .., "max": .., "scale": "log"},  # one conditional run
     "models": [{"name", "beta" | null, "cond" | null,
                 "decoder": {"path", "dtype", "input": {"z": 2, "cond": k},
                             "layers": [{"type": "linear", "in", "out", "w", "b"}, {"type": "relu"}, ...,
                                        {"type": "sigmoid"}]},       # w/b: element offsets
                 "evals": [{"beta", "recon", "kl", "mu", "n", "recon_mu"?}], "labels": path,
                 "history": {"epoch": [...], "train" | "test": {"recon": [...], "kl": [...]}} | null}],
     "recon": {"images", "labels", "n", "per_class"} | null}   # images grouped by digit, per_class each
"""
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .runs import Run

FORMAT_VERSION = 1


def _decoder_blob(model, dtype):
    parts, layers, off = [], [], 0
    for m in model.decoder:
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu().numpy().astype(dtype)
            b = m.bias.detach().cpu().numpy().astype(dtype)
            layers.append({"type": "linear", "in": m.in_features, "out": m.out_features,
                           "w": off, "b": off + w.size})
            off += w.size + b.size
            parts += [w.ravel(), b.ravel()]
        elif isinstance(m, nn.ReLU):
            layers.append({"type": "relu"})
        else:
            raise TypeError(f"decoder layer {type(m).__name__} not supported by the JS runtime")
    layers.append({"type": "sigmoid"})  # VAE.decode = sigmoid(decoder(z))
    return np.concatenate(parts).astype("<" + np.dtype(dtype).str[1:]).tobytes(), layers


def _history_columns(history):
    """history.json rows -> per-split columns for the training-curve chart."""
    if not history:
        return None
    cols = {"epoch": [h["epoch"] for h in history]}
    for split in ("train", "test"):
        cols[split] = {k: [round(h[split][k], 4) for h in history] for k in ("recon", "kl")}
    return cols


def _recon_images(binarize, per_class, data_dir):
    """The first `per_class` test images of each digit, grouped by digit: (indices, x [k, 784])."""
    from .data import get_test_set
    x, y = get_test_set(data_dir, binarize)  # the draw training evaluated on (seed 0)
    idx = np.concatenate([np.flatnonzero(y.numpy() == d)[:per_class] for d in range(10)])
    return idx, x[idx], y.numpy()[idx]


@torch.no_grad()
def _encoder_means(run, x, beta):
    p = next(run.model.parameters())
    mu, _ = run.model.encode(x.to(p.device, p.dtype), run.model.cond_vec(beta))
    return mu.cpu().numpy()


def build_bundle(runs, lim=3.0, n_points=3000, dtype="float32", seed=0, recon_per_class=3,
                 data_dir="./data"):
    """runs: {β: Run} of plain runs, a list of them, or a single conditional Run.
    recon_per_class: test images per digit pushed through every model's own encoder and decoder
    (0 to skip; needs MNIST under data_dir). Returns (manifest dict, {path: bytes})."""
    if isinstance(runs, Run):
        runs = [runs]
    elif isinstance(runs, dict):
        runs = [runs[k] for k in sorted(runs)]
    kinds = {r.kind for r in runs}
    if kinds == {"plain"}:
        beta = {"mode": "discrete", "values": [r.beta for r in runs]}
    elif kinds == {"conditional"} and len(runs) == 1:
        c = runs[0].cond
        beta = {"mode": "continuous", "min": c["beta_min"], "max": c["beta_max"], "scale": "log"}
    else:
        raise ValueError("bundle must be plain runs only, or exactly one conditional run")
    binarize = {r.binarize for r in runs}
    if len(binarize) > 1:
        raise ValueError(f"runs mix preprocessing {sorted(binarize)}; losses would not be comparable")

    rng = np.random.default_rng(seed)
    buffers, models = {}, []
    recon = None
    if recon_per_class:
        r_idx, r_x, r_y = _recon_images(next(iter(binarize)), recon_per_class, data_dir)
        buffers["recon/images.bin"] = (r_x * 255).round().to(torch.uint8).numpy().tobytes()
        buffers["recon/labels.bin"] = r_y.astype(np.uint8).tobytes()
        recon = {"images": "recon/images.bin", "labels": "recon/labels.bin", "n": len(r_idx),
                 "per_class": recon_per_class}
    for r in runs:
        if not r.metrics:
            raise ValueError(f"{r} has no evaluation; load it with load_run / evaluate_run")
        name = r.name
        blob, layers = _decoder_blob(r.model, dtype)
        buffers[f"{name}/decoder.bin"] = blob
        n_all = len(r.labels)
        idx = np.sort(rng.choice(n_all, min(n_points, n_all), replace=False))
        buffers[f"{name}/labels.bin"] = r.labels[idx].astype(np.uint8).tobytes()
        evals = []
        for i, b in enumerate(r.eval_betas):
            path = f"{name}/mu_{i}.bin"
            buffers[path] = r.latents[b][idx].astype("<f4").tobytes()
            m = r.metrics[b]
            evals.append({"beta": b, "recon": m["recon"], "kl": m["kl"], "mu": path, "n": len(idx)})
            if recon:
                mu = _encoder_means(r, r_x, b)
                full = r.latents[b]
                if len(full) == len(r.labels) and np.abs(full[r_idx] - mu).max() > 1e-3:
                    warnings.warn(f"{name}: recon images differ from the evaluated test set "
                                  "(another binarization seed?); μ(x) recomputed from the encoder")
                evals[-1]["recon_mu"] = f"{name}/recon_mu_{i}.bin"
                buffers[evals[-1]["recon_mu"]] = mu.astype("<f4").tobytes()
        models.append({
            "name": name, "beta": r.beta, "cond": r.cond,
            "decoder": {"path": f"{name}/decoder.bin", "dtype": dtype,
                        "input": {"z": r.model.latent_dim, "cond": r.model.cond_dim},
                        "layers": layers},
            "evals": evals, "labels": f"{name}/labels.bin",
            "history": _history_columns(r.history),
        })
    manifest = {"version": FORMAT_VERSION, "lim": lim, "latent_dim": 2, "image": [28, 28],
                "binarize": binarize.pop(), "beta": beta, "models": models, "recon": recon}
    return manifest, buffers


def export_bundle(runs, out_dir, **kw):
    """Write a bundle to `out_dir` (manifest.json + binaries). kw: see build_bundle."""
    out_dir = Path(out_dir)
    manifest, buffers = build_bundle(runs, **kw)
    for path, data in buffers.items():
        p = out_dir / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out_dir / "manifest.json"
