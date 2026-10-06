"""Synthetic fuselage meshes (FlightGear axes: +x forward, +y left, +z up).

Each body is a lofted ring family with an analytic cross-section, so tests can
compare a fit against known truth. Pure numpy, no pyvista needed.
"""
import numpy as np


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def loft(ring_fn, length=10.0, n_x=220, n_theta=64, noise=0.0, seed=0):
    """ring_fn(s) -> (cy, cz, a, b_top, b_bot, p): s in [0,1] tail->nose.

    Cross-section: superellipse with half-width a, upper half-height b_top,
    lower half-height b_bot and exponent p (2 = ellipse, >2 boxier).
    Returns (points[N,3], tris[T,3]).
    """
    s = np.linspace(0.0, 1.0, n_x)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    c, sn = np.cos(th), np.sin(th)
    pts = []
    for si in s:
        cy, cz, a, bt, bb, p = ring_fn(si)
        e = 2.0 / p
        y = cy + a * np.sign(c) * np.abs(c) ** e
        b = np.where(sn >= 0, bt, bb)
        z = cz + b * np.sign(sn) * np.abs(sn) ** e
        x = np.full_like(y, (si - 0.5) * length)
        pts.append(np.stack([x, y, z], 1))
    pts = np.concatenate(pts)
    tris = []
    for i in range(n_x - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            p00, p01 = i * n_theta + j, i * n_theta + j2
            p10, p11 = (i + 1) * n_theta + j, (i + 1) * n_theta + j2
            tris.append((p00, p10, p11))
            tris.append((p00, p11, p01))
    tris = np.array(tris)
    if noise:
        rng = np.random.default_rng(seed)
        pts = pts + rng.normal(0, noise, pts.shape)
    return pts, tris


def glider(**kw):
    """Slender body: rounded nose, canopy bump, long tapering tail boom."""
    def ring(s):
        nose = np.sqrt(np.clip(1 - ((s - 0.78) / 0.22) ** 2, 0, 1)) if s > 0.78 else 1.0
        boom = 0.09 + 0.41 * _smoothstep(0.15, 0.6, s)
        canopy = 0.35 * np.exp(-((s - 0.66) / 0.09) ** 2) * _smoothstep(0.45, 0.6, s)
        a = boom * nose
        bt = (boom + canopy) * nose
        bb = boom * 0.85 * nose
        cz = 0.05 * (1 - s) - 0.02
        return 0.0, cz, max(a, 1e-4), max(bt, 1e-4), max(bb, 1e-4), 2.4
    return loft(ring, length=8.0, **kw)


def liner(**kw):
    """Cylinder with ogive nose and conical tail - needs very few pipes."""
    def ring(s):
        r = 1.0
        if s > 0.88:
            r = np.sqrt(np.clip(1 - ((s - 0.88) / 0.12) ** 2, 0, 1))
        if s < 0.2:
            r = 0.25 + 0.75 * s / 0.2
        r = max(r, 1e-4)
        return 0.0, 0.0, r, r, r, 2.0
    return loft(ring, length=30.0, **kw)


def flat_belly(**kw):
    """Wide, flat-bellied body: strongly non-circular boxy sections."""
    def ring(s):
        nose = np.sqrt(np.clip(1 - ((s - 0.85) / 0.15) ** 2, 0, 1)) if s > 0.85 else 1.0
        w = (0.55 + 0.35 * _smoothstep(0.1, 0.5, s)) * nose
        h = (0.28 + 0.12 * _smoothstep(0.3, 0.6, s)) * nose
        return 0.0, 0.0, max(w, 1e-4), max(h, 1e-4), max(h * 0.7, 1e-4), 3.2
    return loft(ring, length=6.0, **kw)


BODIES = {"glider": glider, "liner": liner, "flat_belly": flat_belly}
