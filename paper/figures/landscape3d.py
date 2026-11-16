#!/usr/bin/env python3
"""3D conceptual landscapes that survive being opened in Inkscape.

Figure 1(a) contrasts three ways of spending one design budget on a toy
sequence-function landscape. It was drawn as a 2D filled contour seen from
above; the story draft wants it in 3D.

WHY NOT mplot3d
---------------
`Axes3D.plot_surface` is the obvious route and it breaks the one rule in
docs/FIGURES.md. It emits one filled path per surface quad, so a 60x60 grid is
~3,600 objects with no axis spines to grab, and its text is positioned by
projecting a data coordinate, which means every label moves if anything in the
view is nudged. The SVG looks right and cannot be edited.

WHAT THIS DOES INSTEAD
----------------------
A hand-rolled oblique projection, drawn as a ridgeline: ~44 profiles at
constant y, each ONE filled polyline, painted back to front so the near rows
occlude the far ones. That reads as a surface, costs ~44 objects instead of
~3,600, and leaves every label a real <text> placed in axis coordinates.

    iso(x, y, z) -> (x + SHEAR*y,  SKEW*y + RISE*z)

The same transform carries the design points, so a dot that should sit on a
peak sits on it. Nothing here is measured -- it is an argument about search
strategy made before any data, and the panel should be read that way.

The 2D version is still in `concept_library.py`. Swapping back is one import.
"""

import numpy as np

from palette import INK, SEQUENTIAL, fs

# --- the projection ------------------------------------------------------
#
# Oblique rather than true perspective: parallel lines stay parallel, so a
# peak at the back is directly comparable in height to one at the front. A
# vanishing point would make the back optima look smaller, which is an
# argument the panel is not trying to make.
SHEAR = 0.42      # how far right a unit of depth pushes
SKEW = 0.30       # how far up a unit of depth pushes
RISE = 0.46       # how tall a unit of fitness stands

N_ROWS = 44       # profiles drawn; the object count of the whole surface
N_COLS = 150      # samples along each profile -- vertices, not objects

FILL = "#FFFFFF"
EDGE = SEQUENTIAL[2]
EDGE_HI = SEQUENTIAL[4]
DOT = "#1F2937"


def iso(x, y, z):
    """-> (px, py) in panel coordinates. Vectorised over arrays."""
    return x + SHEAR * y, SKEW * y + RISE * z


def extent():
    """-> (xlim, ylim) that hold the whole projected unit cube.

    Computed from the transform rather than written down, so changing SHEAR
    or RISE cannot silently clip the surface.
    """
    xs, ys = [], []
    for x in (0.0, 1.0):
        for y in (0.0, 1.0):
            for z in (0.0, 1.0):
                px, py = iso(x, y, z)
                xs.append(px)
                ys.append(py)
    pad = 0.04
    # Extra room under the floor so the corner cue in `axis_cue` sits BELOW
    # the front row instead of inside it. The front row's baseline is y = 0
    # exactly, so a cue placed at the old bottom pad landed on the surface.
    return ((min(xs) - pad, max(xs) + pad),
            (min(ys) - 0.17, max(ys) + pad * 1.5))


def draw_surface(ax, fitness, zmax=None, rows=N_ROWS, lw=0.75):
    """Paint `fitness` as an isometric ridgeline, back row first.

    `fitness(X, Y) -> Z` is whatever the panel's landscape is. `zmax`
    normalises height; pass the SAME value to every panel in a row or the
    three landscapes will not be comparable, which is the panel's whole job.
    """
    gx = np.linspace(0.0, 1.0, N_COLS)
    ys = np.linspace(1.0, 0.0, rows)          # back to front: painter's order
    if zmax is None:
        zmax = float(np.max(fitness(*np.meshgrid(gx, ys))))

    for y in ys:
        z = fitness(gx, np.full_like(gx, y)) / zmax
        px, py = iso(gx, np.full_like(gx, y), z)
        # The polygon closes along the row's own baseline (z = 0), not the
        # bottom of the panel. Closing at the panel floor would make every
        # row a full-height silhouette and the far rows would be hidden
        # entirely rather than occluded where the near surface is taller.
        bx, by = iso(gx[::-1], np.full_like(gx, y), np.zeros_like(gx))
        ax.fill(np.concatenate([px, bx]), np.concatenate([py, by]),
                facecolor=FILL, edgecolor="none", zorder=2)
        ax.plot(px, py, color=EDGE, lw=lw, solid_capstyle="round", zorder=3)
    return zmax


def frame(ax):
    """Set limits and strip the axes. No ticks: there are no units."""
    (x0, x1), (y0, y1) = extent()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_points(ax, fitness, pts, zmax, ms=3.6, lift=0.012):
    """Designs, projected onto the surface so they sit where they belong.

    `lift` raises each dot a hair above the surface. Without it a dot on a
    steep face is half-swallowed by the profile line drawn through it.
    """
    pts = np.asarray(pts, float)
    z = fitness(pts[:, 0], pts[:, 1]) / zmax + lift
    px, py = iso(pts[:, 0], pts[:, 1], z)
    # Near points must occlude far ones, same as the surface rows.
    order = np.argsort(-pts[:, 1])
    ax.plot(px[order], py[order], "o", ms=ms, mfc=DOT, mec="white", mew=0.55,
            ls="none", zorder=6)


def draw_start(ax, fitness, start, zmax, ms=11, lift=0.02):
    """The parent design: where every campaign actually begins."""
    x, y = start
    z = fitness(np.array([x]), np.array([y]))[0] / zmax + lift
    px, py = iso(x, y, z)
    ax.plot([px], [py], marker="*", ms=ms, mfc="white", mec=DOT, mew=1.1,
            ls="none", zorder=7)


def axis_cue(ax, label="Sequence space"):
    """One corner cue naming the plane, instead of two labelled axes.

    Two axis labels on an oblique projection land at angles that do not match
    either axis, and there are no ticks for them to describe. A short L in the
    near corner says which way the plane runs and takes one text object.
    """
    (x0, x1), (y0, y1) = extent()
    ox, oy = x0 + 0.06, y0 + 0.035
    ax.plot([ox, ox + 0.18], [oy, oy], color=INK["muted"], lw=0.9,
            solid_capstyle="butt", zorder=8, clip_on=False)
    ax.plot([ox, ox + 0.18 * SHEAR], [oy, oy + 0.18 * SKEW],
            color=INK["muted"], lw=0.9, solid_capstyle="butt", zorder=8,
            clip_on=False)
    ax.text(ox + 0.21, oy - 0.005, label, fontsize=fs(7),
            color=INK["muted"], ha="left", va="center", zorder=8)


def title(ax, text):
    """Panel-row titles, black: they name the arms being contrasted.

    Deliberately not colour-coded. Tinting only "this work" would say the
    comparison had been decided before the reader got to it.
    """
    ax.set_title(text, fontsize=fs(9), pad=4, color="black")
