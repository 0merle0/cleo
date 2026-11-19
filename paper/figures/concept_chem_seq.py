#!/usr/bin/env python3
"""One panel: the ground-truth landscape over chemical x sequence space.

x is chemical space -- which reaction you are asking for -- and y is
sequence space. The surface is the activity that actually exists, and its
peaks are COLOURED BY REACTION: different chemistries are served by
different regions of sequence space, and one of them here is served by two
separate regions, because more than one solution usually exists.

The dots are an assay. They sample the plane without knowing where the peaks
are, which is the whole situation: the surface is real, it is not visible,
and the only access to it is a few hundred measurements.

WHY THE SURFACE IS AN IMAGE AND NOT STACKED CONTOURS. Four semi-transparent
`contourf` layers composite in the order they were drawn, so the last
reaction painted tints every peak under it and the colour of an overlap
depends on draw order rather than on the data. Here each reaction's field is
composited per pixel into one RGB array -- a peak's colour is its own
reaction's, and a genuine overlap mixes -- and that array is drawn as a
single rasterised image. Axes, dots, labels and the legend stay vector, so
the file is still editable.

    uv run python paper/figures/concept_chem_seq.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from figio import save  # noqa: E402
from palette import INK, PALETTE, fs  # noqa: E402

GRID = 600          # pixels per side of the composited image
N_DOTS = 170        # assayed variants
GAMMA = 0.85        # <1 lifts the shoulders so peaks read as peaks, not dots

# One entry per reaction: colour, and its peaks as (cx, cy, width, height).
#
# cx is position in CHEMICAL space and cy in SEQUENCE space, so a reaction
# occupies a narrow band in x -- it is one chemistry -- and may occupy more
# than one band in y, which is the interesting case: two unrelated families
# of sequence both catalyse it. Reaction C is drawn that way deliberately.
REACTIONS = [
    ("Reaction A", PALETTE["blue"],
     [(0.17, 0.72, 0.070, 1.00)]),
    ("Reaction B", PALETTE["yellow"],
     [(0.40, 0.30, 0.065, 0.86)]),
    ("Reaction C", PALETTE["green"],
     [(0.63, 0.80, 0.060, 0.78),
      (0.68, 0.22, 0.055, 0.62)]),
    ("Reaction D", PALETTE["purple"],
     [(0.87, 0.55, 0.068, 0.92)]),
]


def field(X, Y, peaks):
    """-> one reaction's activity field, normalised to [0, 1]."""
    z = np.zeros_like(X)
    for cx, cy, s, a in peaks:
        # Anisotropic: tighter in x than in y. A reaction is a narrow slice
        # of chemical space, while the sequences that catalyse it are a
        # broader neighbourhood -- a circular peak would say the two axes
        # mean the same kind of thing.
        z = np.maximum(z, a * np.exp(-(((X - cx) / (s * 0.85)) ** 2
                                       + ((Y - cy) / (s * 1.9)) ** 2) / 2))
    return z


def composite(X, Y):
    """-> (H, W, 3) RGB of every reaction's field, mixed per pixel.

    Painted onto white by repeated `over` compositing. Because the alpha is
    the reaction's own normalised activity, a pixel where two reactions
    overlap ends up between their colours instead of taking whichever was
    drawn last.
    """
    img = np.ones(X.shape + (3,))
    fields = []
    for _, colour, peaks in REACTIONS:
        z = field(X, Y, peaks)
        fields.append(z)
        a = (z ** GAMMA)[..., None]
        img = img * (1 - a) + np.array(to_rgb(colour))[None, None, :] * a
    return img, fields


def main():
    rng = np.random.default_rng(5)
    g = np.linspace(0, 1, GRID)
    X, Y = np.meshgrid(g, g)
    img, fields = composite(X, Y)

    fig = plt.figure(figsize=(5.0, 4.0))
    ax = fig.add_subplot(fig.add_gridspec(
        1, 1, left=0.105, right=0.985, top=0.965, bottom=0.215)[0, 0])

    # One image object. Dense colour as vector would be tens of thousands of
    # paths; the axes and everything on top of it stay editable.
    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), interpolation="bilinear",
              zorder=1, rasterized=True)

    # A faint outline per reaction, so a peak has an edge at print size
    # rather than fading into the page.
    for (_, colour, _), z in zip(REACTIONS, fields):
        ax.contour(X, Y, z, levels=[0.35], colors=[colour], linewidths=0.5,
                   alpha=0.35, zorder=2)

    # The assay: a sample over the plane that knows nothing about the peaks.
    P = rng.uniform(0.02, 0.98, size=(N_DOTS, 2))
    ax.plot(P[:, 0], P[:, 1], "o", ms=3.0, mfc="#1F2937", mec="white",
            mew=0.5, ls="none", zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Chemical space", fontsize=fs(8.5), labelpad=5)
    ax.set_ylabel("Sequence space", fontsize=fs(8.5), labelpad=5)
    for sp in ax.spines.values():
        sp.set_color(INK["muted"])
        sp.set_linewidth(0.9)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="s", ms=5.5, mfc=c, mec="none",
                      ls="none", label=name)
               for name, c, _ in REACTIONS]
    handles.append(Line2D([], [], marker="o", ms=4.0, mfc="#1F2937",
                          mec="white", mew=0.5, ls="none",
                          label="assayed variant"))
    # Below the x-label, not beside it: at bbox -0.045 the legend row and
    # the axis label occupied the same band and overlapped.
    ax.legend(handles=handles, fontsize=fs(6.6), loc="upper center",
              bbox_to_anchor=(0.5, -0.095), ncol=5, frameon=False,
              handletextpad=0.35, columnspacing=1.0, borderpad=0.0)

    save(fig, "concept_chem_seq", dpi=900, tight=False)


if __name__ == "__main__":
    main()
