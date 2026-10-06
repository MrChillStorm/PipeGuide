"""Adaptive pipe fitting for PipeGuide.

Turns a closed triangle mesh into a chain of YASim fuselage pipes that follows
the model tightly without chasing tessellation noise.

Pipeline
--------
1. ``sample_profile``   slice the mesh at dense stations along x (exact
   triangle/plane intersection, vectorised) and reduce every slice to a round
   equivalent: centre + radius.
2. ``fit_chain``        fit a continuous piecewise-linear centre/radius curve.
   Knot positions come from an optimal dynamic-programming segmentation, are
   polished by local search, and the *number* of pipes is chosen automatically
   as the smallest that meets an error tolerance, so simple bodies get few
   pipes and complex ones get more. That is what avoids both over- and
   under-fitting.
3. ``chain_to_sections`` emit YASim ``<fuselage>`` tuples.
4. ``evaluate_sections`` independently measure how much model volume the pipes
   miss (under-fit) and how much empty volume they claim (over-fit).

Only numpy is required.
"""
from dataclasses import dataclass
from math import hypot, pi, sqrt

import numpy as np

FIT_MODES = ("area", "perimeter", "enclosing")


# --------------------------------------------------------------------------
# 1. Cross-section sampling
# --------------------------------------------------------------------------

def _slice_points(points, tris, xs):
    """Intersect every mesh edge with the planes x = xs[k].

    Returns (station_index, y, z) for all crossings, sorted by station. Each
    edge is visited once and expanded only over the stations it spans, so the
    cost is O(edges + crossings) instead of O(edges * stations).
    """
    ia = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    ib = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    lo_i, hi_i = np.minimum(ia, ib), np.maximum(ia, ib)
    keep = np.unique(lo_i.astype(np.int64) * len(points) + hi_i)
    ia, ib = keep // len(points), keep % len(points)

    pa, pb = points[ia], points[ib]
    xa, xb = pa[:, 0], pb[:, 0]
    lo, hi = np.minimum(xa, xb), np.maximum(xa, xb)

    dx = xs[1] - xs[0]
    k0 = np.maximum(np.ceil((lo - xs[0]) / dx), 0).astype(np.int64)
    k1 = np.minimum(np.floor((hi - xs[0]) / dx), len(xs) - 1).astype(np.int64)
    cnt = np.where(hi > lo, np.maximum(k1 - k0 + 1, 0), 0)
    total = int(cnt.sum())
    if total == 0:
        raise ValueError("Mesh produced no cross-sections.")

    edge = np.repeat(np.arange(len(cnt)), cnt)
    first = np.cumsum(cnt) - cnt
    station = k0[edge] + (np.arange(total) - np.repeat(first, cnt))
    t = (xs[station] - xa[edge]) / (xb[edge] - xa[edge])
    y = pa[edge, 1] + t * (pb[edge, 1] - pa[edge, 1])
    z = pa[edge, 2] + t * (pb[edge, 2] - pa[edge, 2])

    order = np.argsort(station, kind="stable")
    return station[order], y[order], z[order]


def _polygon_stats(poly):
    """Area, centroid and perimeter of a simple polygon (shoelace)."""
    y, z = poly[:, 0], poly[:, 1]
    y2, z2 = np.roll(y, -1), np.roll(z, -1)
    cross = y * z2 - y2 * z
    area = 0.5 * cross.sum()
    perim = np.hypot(y2 - y, z2 - z).sum()
    if abs(area) < 1e-18:
        return 0.0, poly.mean(0), perim
    cy = ((y + y2) * cross).sum() / (6 * area)
    cz = ((z + z2) * cross).sum() / (6 * area)
    return abs(area), np.array([cy, cz]), perim


def _envelope(yz, nbins):
    """Outer boundary of a slice as a star-shaped polygon.

    The farthest crossing in each angular bin around the (iterated) centroid
    is kept. Interior clutter (seats, bulkheads, ducts) never reaches the
    outer skin, so it is ignored, and gaps between sparse vertices become
    straight chords, which is exactly what the surface does there.
    """
    if len(yz) < 3:
        return None
    c = 0.5 * (yz.min(0) + yz.max(0))
    poly = None
    for _ in range(3):
        d = yz - c
        rad = np.hypot(d[:, 0], d[:, 1])
        ang = np.arctan2(d[:, 1], d[:, 0]) + pi
        b = np.minimum((ang * (nbins / (2 * pi))).astype(np.int64), nbins - 1)
        order = np.lexsort((rad, b))
        bs = b[order]
        last = np.r_[bs[1:] != bs[:-1], True]
        poly = yz[order[last]]
        if len(poly) < 3:
            return None
        area, cen, _ = _polygon_stats(poly)
        if area <= 0:
            break
        moved = hypot(*(cen - c))
        c = cen
        if moved < 1e-9 * max(np.ptp(yz[:, 0]), np.ptp(yz[:, 1]), 1e-12):
            break
    return poly


def _min_enclosing_circle(pts):
    """Smallest circle containing all points (randomised incremental)."""
    P = [tuple(p) for p in pts[np.random.default_rng(0).permutation(len(pts))]]

    def inside(c, r, p):
        return hypot(p[0] - c[0], p[1] - c[1]) <= r * (1 + 1e-12) + 1e-15

    def two(a, b):
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), hypot(a[0] - b[0], a[1] - b[1]) / 2

    def three(a, b, c):
        ax, ay = a
        bx, by = b[0] - ax, b[1] - ay
        cx, cy = c[0] - ax, c[1] - ay
        d = 2 * (bx * cy - by * cx)
        if abs(d) < 1e-18:
            return max((two(a, b), two(a, c), two(b, c)), key=lambda t: t[1])
        ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d
        uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d
        return (ax + ux, ay + uy), hypot(ux, uy)

    c, r = P[0], 0.0
    for i, p in enumerate(P):
        if inside(c, r, p):
            continue
        c, r = p, 0.0
        for j in range(i):
            q = P[j]
            if inside(c, r, q):
                continue
            c, r = two(p, q)
            for k in range(j):
                s = P[k]
                if not inside(c, r, s):
                    c, r = three(p, q, s)
    return np.array(c), r


@dataclass
class Profile:
    """Dense per-station description of the model's cross-sections."""
    x: np.ndarray            # (M,) station positions
    area: np.ndarray         # (M,) cross-section area
    perim: np.ndarray        # (M,) cross-section perimeter
    centroid: np.ndarray     # (M,2) area centroid (y, z)
    polys: list              # M star-shaped outline polygons (n,2), or None
    valid: np.ndarray        # (M,) bool: station had geometry
    x_range: tuple = (0.0, 0.0)  # true mesh extent along x
    _enc: np.ndarray = None  # lazily computed enclosing circles (M,3)

    def channels(self, fit="area"):
        """(M,3) array of [cy, cz, r] for the chosen round-equivalent."""
        if fit == "area":
            r = np.sqrt(self.area / pi)
        elif fit == "perimeter":
            r = self.perim / (2 * pi)
        elif fit == "enclosing":
            if self._enc is None:
                enc = np.zeros((len(self.x), 3))
                for i, poly in enumerate(self.polys):
                    if poly is not None:
                        c, rr = _min_enclosing_circle(poly)
                        enc[i] = (c[0], c[1], rr)
                self._enc = _fill_invalid(self.x, enc, self.valid)
            return self._enc.copy()
        else:
            raise ValueError(f"Unknown fit mode '{fit}' (use one of {FIT_MODES}).")
        return np.column_stack([self.centroid, r])


def _fill_invalid(x, values, valid):
    """Interpolate over stations without geometry (gaps in the mesh)."""
    if valid.all():
        return values
    out = values.copy()
    for c in range(values.shape[1]):
        out[:, c] = np.interp(x, x[valid], values[valid, c])
    return out


def sample_profile(points, tris, n_stations=480, n_bins=128, inset=2e-3):
    """Slice a triangle mesh at ``n_stations`` planes along x."""
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    length = xmax - xmin
    if length <= 0:
        raise ValueError("Mesh has zero length along x.")
    # Inset by about one station so flat caps are measured at their rim and
    # vertex jitter at the very tip cannot corrupt the first/last section.
    xs = np.linspace(xmin + inset * length, xmax - inset * length, n_stations)

    station, y, z = _slice_points(points, tris, xs)
    bounds = np.searchsorted(station, np.arange(n_stations + 1))
    yz_all = np.column_stack([y, z])

    area = np.zeros(n_stations)
    perim = np.zeros(n_stations)
    cen = np.zeros((n_stations, 2))
    polys = [None] * n_stations
    valid = np.zeros(n_stations, dtype=bool)
    for i in range(n_stations):
        yz = yz_all[bounds[i]:bounds[i + 1]]
        poly = _envelope(yz, n_bins)
        if poly is None:
            continue
        a, c, p = _polygon_stats(poly)
        area[i], cen[i], perim[i], polys[i], valid[i] = a, c, p, poly, True

    if valid.sum() < 3:
        raise ValueError("Fewer than 3 usable cross-sections; is the mesh closed and axis-aligned (+x forwards)?")
    area = np.interp(xs, xs[valid], area[valid])
    perim = np.interp(xs, xs[valid], perim[valid])
    cen = _fill_invalid(xs, cen, valid)
    return Profile(xs, area, perim, cen, polys, valid, x_range=(xmin, xmax))


# --------------------------------------------------------------------------
# 2. Piecewise-linear chain fit
# --------------------------------------------------------------------------

@dataclass
class Chain:
    """Continuous piecewise-linear centre/radius curve; K segments, K+1 knots."""
    x: np.ndarray    # (K+1,)
    cy: np.ndarray
    cz: np.ndarray
    r: np.ndarray
    rms: float       # relative RMS error against the dense profile
    p99_err: float   # relative error not exceeded at 99% of stations
    d: np.ndarray = None   # lobe spacing per knot (multi-lobe fits only)
    n_lobes: int = 1
    axis: str = "y"        # direction the lobes are spread along

    @property
    def n_segments(self):
        return len(self.x) - 1

    def extended(self, xmin, xmax):
        """Move the end knots out to the mesh extremes (values unchanged)."""
        x = self.x.copy()
        x[0], x[-1] = min(x[0], xmin), max(x[-1], xmax)
        return Chain(x, self.cy, self.cz, self.r, self.rms, self.p99_err,
                     self.d, self.n_lobes, self.axis)

    def at(self, xq):
        """(cy, cz, r) interpolated at positions xq."""
        return (np.interp(xq, self.x, self.cy),
                np.interp(xq, self.x, self.cz),
                np.interp(xq, self.x, self.r))


def _prefix(a):
    return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])


def _segment_costs(u, F, w, min_len):
    """cost[i, j] = weighted SSE of the best straight line through samples
    i..j (all channels together), via prefix sums, in O(1) per segment."""
    M = len(u)
    P0, P1, P2 = _prefix(w), _prefix(w * u), _prefix(w * u * u)
    Q = _prefix(w[:, None] * F)
    R = _prefix((w * u)[:, None] * F)
    T = _prefix((w[:, None] * F * F).sum(1))

    i = np.arange(M)[:, None]
    j = np.arange(M)[None, :]
    s0 = P0[j + 1] - P0[i]
    s1 = P1[j + 1] - P1[i]
    s2 = P2[j + 1] - P2[i]
    q = Q[j + 1] - Q[i]
    r = R[j + 1] - R[i]
    t = T[j + 1] - T[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        sxx = s2 - s1 * s1 / s0
        sxy = r - s1[..., None] * q / s0[..., None]
        sse = t - (q * q).sum(-1) / s0 - (sxy * sxy).sum(-1) / sxx
    sse = np.maximum(sse, 0.0)
    sse[(j - i + 1) < min_len] = np.inf
    sse[~np.isfinite(sse) & ((j - i + 1) >= min_len)] = np.inf
    return sse


def _segment_dp(cost, kmax):
    """Optimal split of the samples into k contiguous groups, k = 1..kmax."""
    M = cost.shape[0]
    D = np.full((kmax + 1, M), np.inf)
    A = np.zeros((kmax + 1, M), dtype=np.int64)
    D[1] = cost[0]
    for k in range(2, kmax + 1):
        cand = D[k - 1][:-1, None] + cost[1:, :]
        A[k] = np.argmin(cand, axis=0) + 1
        D[k] = cand.min(axis=0)
    return D, A


def _backtrack(A, k, M):
    """Start index of each group for the k-group solution."""
    starts = []
    j = M - 1
    for kk in range(k, 1, -1):
        i = int(A[kk][j])
        starts.append(i)
        j = i - 1
    return starts[::-1]


def _hat_basis(x, t):
    idx = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2)
    lam = (x - t[idx]) / (t[idx + 1] - t[idx])
    B = np.zeros((len(x), len(t)))
    rows = np.arange(len(x))
    B[rows, idx] = 1 - lam
    B[rows, idx + 1] = lam
    return B


def _solve_knots(B, f, w):
    A = B.T @ (B * w[:, None])
    A[np.diag_indices_from(A)] += 1e-10 * np.trace(A) / len(A)
    return np.linalg.solve(A, B.T @ (w * f))


def _refit(x, F, ws, t, tau, robust=True, iters=8):
    """Continuous least-squares knot values, optionally asymmetric/robust.

    Channel 2 (radius) uses expectile weights: tau = 0.5 is plain least
    squares, tau > 0.5 penalises pipes that are too small more than too
    large (hug the outside), tau < 0.5 the reverse. Huber weights stop
    isolated artefacts (a stray antenna, a bad vertex) from dragging a pipe.
    """
    B = _hat_basis(x, t)
    V = np.zeros((len(t), F.shape[1]))
    for c in range(F.shape[1]):
        asym = abs(tau - 0.5) > 1e-9 and c == 2
        w = ws.copy()
        for _ in range(iters if (robust or asym) else 1):
            V[:, c] = _solve_knots(B, F[:, c], w)
            e = F[:, c] - B @ V[:, c]
            w = ws.copy()
            if asym:
                w *= 2 * np.where(e > 0, tau, 1 - tau)
            if robust:
                # z is a relative error. The threshold has a floor so that
                # clean data (median error ~ 0) never marks real shape
                # changes as outliers.
                z = np.abs(e) * np.sqrt(ws)
                thr = max(2.5 * 1.4826 * np.median(z), 0.08)
                w *= np.minimum(1.0, thr / np.maximum(z, 1e-30))
    return V


def _polish_knots(x, F, ws, t, V, sweeps=3, n_cand=24):
    """Coordinate descent on interior knot positions.

    With its neighbours fixed, each knot's best position (and the optimal
    value there) is found in closed form per candidate, so a sweep is cheap.
    """
    t, V = t.copy(), V.copy()
    min_gap = 3.0 * (x[1] - x[0])
    for _ in range(sweeps):
        moved = False
        for j in range(1, len(t) - 1):
            tl, tr = t[j - 1], t[j + 1]
            sel = (x > tl) & (x < tr)
            if sel.sum() < 4:
                continue
            xs, Fs, wsel = x[sel], F[sel], ws[sel]
            cands = xs[(xs > tl + min_gap) & (xs < tr - min_gap)]
            if len(cands) == 0:
                continue
            cands = np.unique(np.append(
                cands[np.linspace(0, len(cands) - 1, min(n_cand, len(cands))).astype(int)], t[j]))
            best = (np.inf, t[j], V[j])
            for p in cands:
                left = xs <= p
                lam = np.where(left, (xs - tl) / (p - tl), (xs - p) / (tr - p))
                h = np.where(left, lam, 1 - lam)
                base = np.where(left[:, None], V[j - 1] * (1 - lam)[:, None], V[j + 1] * lam[:, None])
                hw = wsel * h
                den = (hw * h).sum()
                if den < 1e-12:
                    continue
                v = (hw[:, None] * (Fs - base)).sum(0) / den
                e = Fs - base - h[:, None] * v
                sse = (wsel[:, None] * e * e).sum()
                if sse < best[0] - 1e-15:
                    best = (sse, p, v)
            if abs(best[1] - t[j]) > 1e-12:
                moved = True
            t[j], V[j] = best[1], best[2]
        if not moved:
            break
    return t, V


def _errors(x, F, ws, t, V):
    e = F - _hat_basis(x, t) @ V
    per = np.sqrt(ws * (e * e).sum(1))
    return float(sqrt((ws * (e * e).sum(1)).sum() / ws.sum())), float(np.percentile(per, 99))


def fit_chain(x, F, kmax=63, tol=0.01, bias=0.0, center_weight=0.5,
              min_len=3, polish=True, min_radius_frac=0.25):
    """Fit a piecewise-linear chain with automatically chosen pipe count.

    x, F      dense stations and their [cy, cz, r] targets.
    kmax      upper bound on pipes.
    tol       target RMS error relative to the local radius (floored at
              ``min_radius_frac`` of the largest radius so tips stay sane).
              The fewest pipes meeting it are used; 0 forces ``kmax``.
    bias      -1..1. 0 is area-neutral (balanced over/under fit), >0 prefers
              pipes that contain the model, <0 pipes that sit inside it.
    """
    x = np.asarray(x, float)
    F = np.asarray(F, float)
    M = len(x)
    kmax = int(max(1, min(kmax, M // max(min_len, 1))))
    rmax = max(F[:, 2].max(), 1e-12)
    # 1/scale^2 so that sqrt(ws)*error is a dimensionless relative error.
    ws = 1.0 / np.maximum(F[:, 2], min_radius_frac * rmax) ** 2

    cw = np.sqrt(np.r_[center_weight, center_weight, 1.0,
                       [center_weight] * (F.shape[1] - 3)])
    Fs = (F - F.mean(0)) * cw          # scaled + centred for numerics
    Fm = F.mean(0)
    u = (x - x[0]) / (x[-1] - x[0])
    tau = 0.5 + 0.4 * float(np.clip(bias, -1, 1))

    cost = _segment_costs(u, Fs, ws, min_len)
    D, A = _segment_dp(cost, kmax)

    def build(k):
        starts = _backtrack(A, k, M)
        t = np.concatenate([[x[0]], [0.5 * (x[i - 1] + x[i]) for i in starts], [x[-1]]])
        Fl = F - Fm
        V = _refit(x, Fl * cw, ws, t, 0.5, robust=False)
        if polish and k > 1:
            t, V = _polish_knots(x, Fl * cw, ws, t, V)
        V = _refit(x, Fl * cw, ws, t, tau, robust=True)
        rms, p99 = _errors(x, Fl * cw, ws, t, V)
        d = (np.maximum(V[:, 3] / cw[3] + Fm[3], 0.0)
             if F.shape[1] > 3 else None)
        return Chain(t, V[:, 0] / cw[0] + Fm[0], V[:, 1] / cw[1] + Fm[1],
                     np.maximum(V[:, 2] / cw[2] + Fm[2], 0.0), rms, p99, d)

    if tol <= 0:
        return build(kmax)

    # Never ask for accuracy below the measurement noise: that would make the
    # chain trace tessellation jitter (over-fit) and burn pipes for nothing.
    d2 = (F[:-2, 2] - 2 * F[1:-1, 2] + F[2:, 2]) * np.sqrt(ws[1:-1])
    sigma = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / sqrt(6)
    tol_eff = max(tol, 1.5 * sigma)

    top = build(kmax)
    # Tolerance beyond what kmax pipes can reach: settle for the knee, the
    # fewest pipes within 10% of the best achievable error.
    rms_goal = max(tol_eff, top.rms * 1.1)
    p99_goal = max(4 * tol_eff, top.p99_err * 1.1)

    def ok(ch):
        return ch.rms <= rms_goal and ch.p99_err <= p99_goal

    # Smallest k that meets the goal (error is ~monotone in k).
    lo, hi, best = 1, kmax, top
    while lo < hi:
        mid = (lo + hi) // 2
        ch = build(mid)
        if ok(ch):
            hi, best = mid, ch
        else:
            lo = mid + 1
    return best if best.n_segments == hi else build(hi)


# --------------------------------------------------------------------------
# 2b. Multi-lobe sections (opt-in)
# --------------------------------------------------------------------------

def _lobe_offsets(n):
    """Lobe centre multipliers of the spacing d, e.g. n=3 -> -1, 0, +1."""
    return [k - (n - 1) / 2.0 for k in range(n)]


def lobe_axis(profile):
    """Spread lobes across the wider direction of the body ('y' or 'z')."""
    w = h = 0.0
    for poly, a in zip(profile.polys, profile.area):
        if poly is not None:
            w += a * np.ptp(poly[:, 0])
            h += a * np.ptp(poly[:, 1])
    return "y" if w >= h else "z"


def _union_radial(theta, rho, d, offsets, axis):
    """Radial function of a row of equal discs, seen from the row's centre."""
    proj = np.cos(theta) if axis == "y" else np.sin(theta)
    out = np.zeros(np.broadcast(rho, d, proj).shape)
    for m in offsets:
        o = m * d
        p = o * proj
        q = rho * rho - o * o + p * p
        t = np.where(q >= 0, p + np.sqrt(np.maximum(q, 0)), 0.0)
        out = np.maximum(out, t)
    return out


def lobe_channels(profile, n, axis, idx, n_theta=128, grid=22, passes=3):
    """Per-station [cy, cz, radius, spacing] of the best n-lobe round row.

    Minimises the symmetric difference with the true outline (exact for
    star-shaped outlines, via their radial function) with a zooming grid
    search over lobe radius and spacing.
    """
    theta = np.linspace(-pi, pi, n_theta, endpoint=False)
    dth = 2 * pi / n_theta
    offsets = _lobe_offsets(n)
    base = profile.channels("area")
    out = np.zeros((len(idx), 4))
    for row, i in enumerate(idx):
        out[row, :3] = base[i]
        poly = profile.polys[i]
        if poly is None or n == 1:
            continue
        c = profile.centroid[i]
        v = poly - c
        ang = np.arctan2(v[:, 1], v[:, 0])
        o = np.argsort(ang)
        rho_true = np.interp(theta, ang[o], np.hypot(v[o, 0], v[o, 1]), period=2 * pi)
        r0 = max(base[i, 2], 1e-9)
        # Outline half-extents along / across the lobe row. Matching them
        # breaks the near-degenerate trade-off between radius and spacing
        # (a slightly bigger, closer row looks almost identical), which would
        # otherwise make the estimate wander and cost needless pipes.
        ax_i = 0 if axis == "y" else 1
        ext_major = max(abs(v[:, ax_i].min()), abs(v[:, ax_i].max()))
        ext_minor = max(abs(v[:, 1 - ax_i].min()), abs(v[:, 1 - ax_i].max()))
        r_lo, r_hi, d_lo, d_hi = 0.25 * r0, 1.3 * r0, 0.0, 2.5 * r0
        for _ in range(passes):
            R = np.linspace(r_lo, r_hi, grid)[:, None, None]
            Dg = np.linspace(d_lo, d_hi, grid)[None, :, None]
            U = _union_radial(theta[None, None, :], R, Dg, offsets, axis)
            sd = (np.abs(rho_true ** 2 - U ** 2)).sum(-1) * dth / 2
            ext_u = (n - 1) / 2.0 * Dg[..., 0] + R[..., 0]
            sd = sd + 0.25 * r0 * (np.abs(ext_u - ext_major)
                                   + np.abs(R[..., 0] - ext_minor))
            a, b = np.unravel_index(np.argmin(sd), sd.shape)
            rs = np.linspace(r_lo, r_hi, grid)
            ds = np.linspace(d_lo, d_hi, grid)
            rstep, dstep = rs[1] - rs[0], ds[1] - ds[0]
            r_best, d_best = rs[a], ds[b]
            r_lo, r_hi = max(r_best - 2 * rstep, 1e-9), r_best + 2 * rstep
            d_lo, d_hi = max(d_best - 2 * dstep, 0.0), d_best + 2 * dstep
        out[row, 2], out[row, 3] = r_best, d_best
    return out


def fit_lobes(profile, n, axis=None, tol=0.01, kmax=63, bias=0.0, stride=2):
    """Fit a chain of n-lobe rows; returns a Chain with spacing ``d``."""
    axis = axis or lobe_axis(profile)
    idx = np.arange(0, len(profile.x), stride)
    F = lobe_channels(profile, n, axis, idx)
    ch = fit_chain(profile.x[idx], F, kmax=kmax, tol=tol, bias=bias)
    ch.n_lobes, ch.axis = n, axis
    return ch


def fit_auto(profile, max_lobes=4, min_gain=0.01, **kw):
    """Add lobes only while each extra one buys at least ``min_gain`` overlap."""
    best = fit_chain(profile.x, profile.channels("area"), **kw)
    best_ev = evaluate_sections(profile, chain_to_sections(best.extended(*profile.x_range)))
    for n in range(2, max_lobes + 1):
        cand = fit_lobes(profile, n, **kw)
        ev = evaluate_sections(profile, chain_to_sections(cand.extended(*profile.x_range)))
        if ev["iou"] - best_ev["iou"] < min_gain:
            break
        best, best_ev = cand, ev
    return best


# --------------------------------------------------------------------------
# 3. YASim output
# --------------------------------------------------------------------------

def chain_to_sections(chain, prune=0.1):
    """Chain -> YASim fuselage tuples (ax,ay,az,bx,by,bz,width,taper,midpoint).

    Uses PipeGuide's established axis convention (x and y negated for YASim).
    A linear cone is midpoint 1 (widest at b) or 0 (widest at a).

    Multi-lobe chains emit one pipe per lobe, but a lobe is pruned wherever
    its offset from the centre is below ``prune`` of the radius, since it
    would just duplicate the pipe next to it.
    """
    n = chain.n_lobes
    d = chain.d if chain.d is not None else np.zeros_like(chain.x)
    out = []

    def cone(ra, rb, pa, pb):
        wide, narrow = max(ra, rb), min(ra, rb)
        taper = narrow / wide if wide > 0 else 1.0
        mid = 1.0 if ra < rb else 0.0 if ra > rb else 0.5
        return (-chain.x[i], -pa[0], pa[1], -chain.x[i + 1], -pb[0], pb[1],
                2 * wide, taper, mid)

    def at(k, m):
        off = m * d[k]
        y = chain.cy[k] + (off if chain.axis == "y" else 0.0)
        z = chain.cz[k] + (off if chain.axis == "z" else 0.0)
        return (y, z)

    for i in range(chain.n_segments):
        ra, rb = float(chain.r[i]), float(chain.r[i + 1])
        offsets = _lobe_offsets(n)
        spread = max(d[i], d[i + 1])
        small = spread < prune * max(ra, rb)
        if small:
            # Lobes coincide here: one centred pipe stands in for all of them.
            out.append(cone(ra, rb, at(i, 0.0), at(i + 1, 0.0)))
            continue
        for m in offsets:
            if m != 0 and abs(m) * spread < prune * max(ra, rb):
                continue
            out.append(cone(ra, rb, at(i, m), at(i + 1, m)))
    return out


# --------------------------------------------------------------------------
# 4. Independent quality evaluation
# --------------------------------------------------------------------------

def _inside_polygon(px, py, poly):
    """Vectorised even-odd point-in-polygon."""
    y1, z1 = poly[:, 0], poly[:, 1]
    y2, z2 = np.roll(y1, -1), np.roll(z1, -1)
    inside = np.zeros(px.shape, dtype=bool)
    for a, b, c, d in zip(y1, z1, y2, z2):
        cond = (b > py) != (d > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = a + (py - b) * (c - a) / (d - b)
        inside ^= cond & (px < xint)
    return inside


def evaluate_sections(profile, sections, n_stations=96, grid=96):
    """Measure fit quality of YASim sections against the model.

    Works for any set of pipes (also overlapping, multi-pipe layouts): at
    every sampled station the union of pipe discs is compared with the true
    cross-section outline.

    Returns volume fractions relative to the model volume:
      under  model volume not covered by any pipe
      over   pipe volume outside the model
      iou    intersection-over-union of pipes and model
    plus ``n_pipes``.
    """
    S = np.array(sections, dtype=float)
    ax, ay, az, bx, by, bz, width, taper, mid = S.T
    # Back to model axes; end radii per YASim convention.
    xa, xb, ya, yb, za, zb = -ax, -bx, -ay, -by, az, bz
    wide, narrow = width / 2, width / 2 * taper
    ra = np.where(mid >= 1.0, narrow, wide)
    rb = np.where(mid >= 1.0, wide, narrow)
    ra = np.where(np.isclose(taper, 1.0), wide, ra)
    rb = np.where(np.isclose(taper, 1.0), wide, rb)

    idx = np.unique(np.linspace(0, len(profile.x) - 1, n_stations).astype(int))
    inter = under = over = union_model = 0.0
    for i in idx:
        poly = profile.polys[i]
        if poly is None:
            continue
        x = profile.x[i]
        lo, hi = np.minimum(xa, xb), np.maximum(xa, xb)
        act = np.where((x >= lo - 1e-9) & (x <= hi + 1e-9) & (hi > lo))[0]
        circles = []
        for k in act:
            f = (x - xa[k]) / (xb[k] - xa[k])
            circles.append((ya[k] + f * (yb[k] - ya[k]), za[k] + f * (zb[k] - za[k]),
                            ra[k] + f * (rb[k] - ra[k])))
        lo_y, hi_y = poly[:, 0].min(), poly[:, 0].max()
        lo_z, hi_z = poly[:, 1].min(), poly[:, 1].max()
        for cy, cz, r in circles:
            lo_y, hi_y = min(lo_y, cy - r), max(hi_y, cy + r)
            lo_z, hi_z = min(lo_z, cz - r), max(hi_z, cz + r)
        gy = np.linspace(lo_y, hi_y, grid, endpoint=False) + (hi_y - lo_y) / grid / 2
        gz = np.linspace(lo_z, hi_z, grid, endpoint=False) + (hi_z - lo_z) / grid / 2
        px, py = np.meshgrid(gy, gz)
        cell = (hi_y - lo_y) * (hi_z - lo_z) / grid ** 2
        in_model = _inside_polygon(px, py, poly)
        in_pipe = np.zeros(px.shape, dtype=bool)
        for cy, cz, r in circles:
            in_pipe |= (px - cy) ** 2 + (py - cz) ** 2 <= r * r
        inter += (in_model & in_pipe).sum() * cell
        under += (in_model & ~in_pipe).sum() * cell
        over += (in_pipe & ~in_model).sum() * cell
        union_model += in_model.sum() * cell
    model = max(union_model, 1e-30)
    return {"under": under / model, "over": over / model,
            "iou": inter / max(inter + under + over, 1e-30),
            "n_pipes": len(sections)}
