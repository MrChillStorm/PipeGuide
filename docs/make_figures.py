#!/usr/bin/env python3
"""Regenerate the README cross-section figures from the synthetic test bodies.

    pip install matplotlib
    python3 docs/make_figures.py

Pass a mesh path to plot your own model instead (needs pyvista):

    python3 docs/make_figures.py my_fuselage.stl
"""
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import pipefit  # noqa: E402
import synthetic  # noqa: E402

GREY, BLUE, RED, INK = "#c9ced6", "#1f6feb", "#d1242f", "#24292f"


def legacy_sections(mesh_path, out):
    subprocess.run([sys.executable, os.path.join(ROOT, "pipeguide.py"), mesh_path,
                    "-o", out, "--legacy"], check=True, capture_output=True)
    rows = []
    for m in re.finditer(r"<fuselage ([^>]*)/>", open(out).read()):
        d = dict(re.findall(r'(\w+)="\s*([-\d.]+)"', m.group(1)))
        rows.append(tuple(float(d[k]) for k in
                          ("ax", "ay", "az", "bx", "by", "bz", "width", "taper", "midpoint")))
    return rows


def style(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    if sub:
        ax.text(0.99, 1.02, sub, transform=ax.transAxes, ha="right", fontsize=9, color="#57606a")
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors="#57606a")


def cross_sections(prof, layouts, path, n=4):
    idx = np.linspace(0.12, 0.88, n)
    cols = [int(i * (len(prof.x) - 1)) for i in idx]
    fig, axes = plt.subplots(len(layouts), n, figsize=(2.6 * n, 2.3 * len(layouts)))
    for r, (name, secs, colour) in enumerate(layouts):
        arrs = pipefit._pipe_arrays(secs)
        for c, i in enumerate(cols):
            ax = axes[r][c]
            poly = prof.polys[i]
            ax.fill(poly[:, 0], poly[:, 1], color=GREY, lw=0)
            for cy, cz, rad in pipefit._circles_at(arrs, prof.x[i]):
                ax.add_patch(plt.Circle((cy, cz), rad, fill=False, ec=colour, lw=1.4))
            ax.set_aspect("equal")
            ax.autoscale()
            ax.margins(0.15)
            ax.axis("off")
            if c == 0:
                ax.set_title(name, loc="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    scratch = os.path.join(HERE, ".tmp")
    os.makedirs(scratch, exist_ok=True)
    if len(sys.argv) > 1:
        import pyvista as pv
        mesh = pv.read(sys.argv[1]).extract_surface(algorithm=None).triangulate()
        pts, tris = np.asarray(mesh.points), mesh.faces.reshape(-1, 4)[:, 1:]
        bodies = {"model": (pts, tris, sys.argv[1])}
    else:
        import pyvista as pv
        bodies = {}
        for name in ("glider", "flat_belly"):
            pts, tris = synthetic.BODIES[name](noise=0.002)
            path = os.path.join(scratch, f"{name}.ply")
            pv.PolyData(pts, np.hstack([np.full((len(tris), 1), 3), tris]).ravel()).save(path)
            bodies[name] = (pts, tris, path)

    for name, (pts, tris, path) in bodies.items():
        prof = pipefit.sample_profile(pts, tris)
        balanced = pipefit.fit_balanced(prof)
        bsecs = pipefit.chain_to_sections(balanced)
        old = legacy_sections(path, os.path.join(scratch, "legacy.xml"))
        i = balanced.info
        cross_sections(prof, [("Previous default", old, RED),
                              (f"Adaptive ({balanced.n_lobes} lobe{'s' if balanced.n_lobes > 1 else ''})",
                               bsecs, BLUE)],
                       os.path.join(HERE, f"{name}_sections.png"))
        print(name, len(old), "->", len(bsecs), "pipes")


if __name__ == "__main__":
    main()
