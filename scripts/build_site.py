"""Build the static GitHub Pages site from trained runs.

    uv run python scripts/build_site.py                      # runs/ -> docs/, all plain runs
    uv run python scripts/build_site.py --conditional runs/cbeta
    uv run python scripts/build_site.py --betas 0.25 1 4 --out _site

Then serve locally with `python -m http.server -d docs`, or point GitHub Pages at main:/docs.
"""
import argparse
import shutil
from pathlib import Path

from vae_demo import export_bundle, list_runs, load_run

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "vae_demo" / "static"


def build_site(runs, out, dtype="float16", n_points=3000):
    out = Path(out)
    if (out / "bundle").exists():
        shutil.rmtree(out / "bundle")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "site" / "index.html", out / "index.html")
    for f in ("latent_pad.js", "latent_pad.css"):
        shutil.copy(STATIC / f, out / f)
    (out / ".nojekyll").touch()  # serve files as-is
    export_bundle(runs, out / "bundle", dtype=dtype, n_points=n_points)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"site written to {out.resolve()}  ({size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--betas", type=float, nargs="*", help="subset of plain runs (default: all)")
    p.add_argument("--conditional", help="path of a conditional run dir; replaces the plain runs")
    p.add_argument("--out", default=str(ROOT / "docs"))
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--points", type=int, default=3000, help="test-set μ(x) points per β")
    args = p.parse_args()

    if args.conditional:
        runs = load_run(args.conditional)
    else:
        runs = list_runs(args.runs)
        if args.betas:
            runs = {b: runs[b] for b in args.betas}
    build_site(runs, args.out, args.dtype, args.points)


if __name__ == "__main__":
    main()
