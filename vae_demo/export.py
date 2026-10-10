"""Export runs as a framework-free "model bundle" consumed by static/latent_pad.js —
both by the notebook widget (in memory, via traitlets) and by the static site (files).

    manifest.json
    <name>/decoder.bin    decoder weights, per Linear layer: W [out, in] row-major, then b [out]
    <name>/mu_<i>.bin     float32 [n, 2] posterior means at eval β i (subsampled test set)
    <name>/labels.bin     uint8 [n]

manifest:
    {"version": 1, "lim": 3, "latent_dim": 2, "image": [28, 28],
     "beta": {"mode": "discrete", "values": [...]}            # plain runs, snap to trained β
           | {"mode": "continuous", "min": .., "max": .., "scale": "log"},  # one conditional run
     "models": [{"name", "beta" | null, "cond" | null,
                 "decoder": {"path", "dtype", "input": {"z": 2, "cond": k},
                             "layers": [{"type": "linear", "in", "out", "w", "b"}, {"type": "relu"}, ...,
                                        {"type": "sigmoid"}]},       # w/b: element offsets
                 "evals": [{"beta", "recon", "kl", "mu", "n"}], "labels": path}]}
"""
import json
from pathlib import Path

import numpy as np
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


def build_bundle(runs, lim=3.0, n_points=3000, dtype="float32", seed=0):
    """runs: {β: Run} of plain runs, a list of them, or a single conditional Run.
    Returns (manifest dict, {path: bytes})."""
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

    rng = np.random.default_rng(seed)
    buffers, models = {}, []
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
        models.append({
            "name": name, "beta": r.beta, "cond": r.cond,
            "decoder": {"path": f"{name}/decoder.bin", "dtype": dtype,
                        "input": {"z": r.model.latent_dim, "cond": r.model.cond_dim},
                        "layers": layers},
            "evals": evals, "labels": f"{name}/labels.bin",
        })
    manifest = {"version": FORMAT_VERSION, "lim": lim, "latent_dim": 2, "image": [28, 28],
                "beta": beta, "models": models}
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
