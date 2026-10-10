# VAE-demo

A β-VAE with a **2-D latent** trained on MNIST, plus a draggable pad over the latent plane that
decodes to an image live (in the browser) and a β slider.

The point of the demo: the latent space is a map you can walk around, and raising β trades
reconstruction sharpness for how fully the prior's mass decodes to something plausible.

## Setup

```bash
uv sync
```

This installs the CUDA 12.8 build of PyTorch on Windows/Linux (see `[tool.uv.sources]` in
`pyproject.toml`; change the index for CPU-only or a different CUDA version).

## Notebook

```bash
uv run jupyter lab main.ipynb
```

`main.ipynb` loads the trained runs in `runs/` (no training needed) and shows the `LatentPad`
widget plus static comparison figures. The minimal version:

```python
from vae_demo import train_sweep, LatentPad
runs = train_sweep([0.05, 0.1, 0.25, 1, 4, 8, 16], out="runs")  # loads runs/beta_*/vae.pt if present
pad = LatentPad.from_runs(runs)
pad            # display; pad.z / pad.beta sync both ways
```

The widget carries every model's decoder weights (~15 MB in all). Save the notebook without widget
state, or clear its outputs, before you commit it.

## Training from the command line

```bash
uv run python -m vae_demo.train --betas 0.25 1 4 --epochs 30 --figures
uv run python -m vae_demo.train --conditional --beta-min 0.05 --beta-max 16 --epochs 60
```

Existing runs are skipped unless you pass `--retrain`. `--conditional` trains a single β-conditional
VAE (β is an input to both encoder and decoder, sampled log-uniformly per example) into
`runs/cbeta/`. Its pad gets a continuous β slider.

## Static site (GitHub Pages)

```bash
uv run python scripts/build_site.py                          # all plain runs -> docs/
uv run python scripts/build_site.py --conditional runs/cbeta # or the conditional model
uv run python -m http.server -d docs                         # preview at http://localhost:8000
```

The build exports the decoder weights as float16 (~7.7 MB for 7 β), and the page loads only the
current β's model. To publish, commit `docs/` and set **Settings → Pages → Source** to
`main` / `/docs`.

## Layout

| Path | What |
| --- | --- |
| `vae_demo/model.py` | `VAE` (optionally β-conditional), `vae_loss` (scalar or per-sample β) |
| `vae_demo/conditioning.py` | β → condition features (mirrored in the JS) |
| `vae_demo/train.py` | training loops, `train_sweep`, `train_conditional`, CLI |
| `vae_demo/runs.py` | `Run`, `save_run`, `load_run`, `list_runs` |
| `vae_demo/export.py` | model bundle (`manifest.json` + binaries) shared by widget and site |
| `vae_demo/viz.py` | matplotlib figures |
| `vae_demo/widget.py`, `vae_demo/static/` | `LatentPad` anywidget and the single JS/CSS front end |
| `site/index.html`, `scripts/build_site.py` | the standalone page and its builder |
| `runs/beta_*/` | trained checkpoints (`vae.pt`, `history.json`, `eval.npz`) |
