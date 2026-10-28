from pathlib import Path

import anywidget
import traitlets

from .export import build_bundle

_STATIC = Path(__file__).parent / "static"


class LatentPad(anywidget.AnyWidget):
    """Draggable pad over the 2-D latent plane, decoded live in the browser, plus a β slider.

    `z` and `beta` sync both ways: read them to see where the user is, set them to move the pad.

        pad = LatentPad.from_runs(list_runs("runs"))
        pad.z = [0.0, 2.5]; pad.beta = 4; pad.digits = [4, 9]
    """
    _esm = _STATIC / "latent_pad.js"
    _css = _STATIC / "latent_pad.css"

    manifest = traitlets.Dict().tag(sync=True)
    buffers = traitlets.Dict().tag(sync=True)      # path -> bytes (sent as binary buffers)
    z = traitlets.List([0.0, 0.0]).tag(sync=True)
    beta = traitlets.Float(1.0).tag(sync=True)
    background = traitlets.Unicode("scatter").tag(sync=True)  # "scatter" | "manifold" | "none"
    pixels = traitlets.Unicode("mean").tag(sync=True)  # "mean": p(x|z) | "sample": x ~ Bernoulli(p)
    digits = traitlets.List(list(range(10))).tag(sync=True)  # classes drawn in the μ(x) scatter

    @classmethod
    def from_runs(cls, runs, lim=3.0, n_points=3000, dtype="float32", beta=None, **kw):
        """runs: {β: Run} of plain runs, or one conditional Run (see export.build_bundle)."""
        manifest, buffers = build_bundle(runs, lim=lim, n_points=n_points, dtype=dtype)
        if beta is None:
            b = manifest["beta"]
            beta = (min(b["values"], key=lambda v: abs(v - 1)) if b["mode"] == "discrete"
                    else 1.0)
        return cls(manifest=manifest, buffers=buffers, beta=float(beta), **kw)
