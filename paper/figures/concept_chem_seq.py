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
from matplotlib.colors import LinearSegmentedColormap, to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from figio import save  # noqa: E402
from palette import INK, PALETTE, fs  # noqa: E402

GRID = 600          # pixels per side of the composited image
N_DOTS = 190        # assayed variants
GAMMA = 0.85        # <1 lifts the shoulders so peaks read as peaks, not dots

# Which reaction the dots are scored against. The panel shows four
# chemistries but a campaign is run for ONE of them, and the assay reports
# activity for that one -- so the dots are coloured by their value in this
# reaction's field and say nothing about the other three.
OF_INTEREST = 0

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
    """-> ((H, W, 4) RGBA of the landscape, [per-reaction fields]).

    RGBA with a real alpha channel, not RGB painted onto white. Compositing
    onto white would bake the page colour into the image, and the figure
    would then carry an opaque white rectangle that no amount of setting
    facecolors to "none" can remove -- the background would look transparent
    everywhere except under the plot.

    Colour is the activity-weighted mix of the reactions present at that
    pixel, so a peak keeps its own colour and a genuine overlap lands
    between the two. Opacity is the probabilistic union of the fields, which
    goes to zero off the peaks -- that is what makes the background
    transparent and the peaks solid.
    """
    cols = np.array([to_rgb(c) for _, c, _ in REACTIONS])
    fields = [field(X, Y, peaks) for _, _, peaks in REACTIONS]
    A = np.stack([z ** GAMMA for z in fields], axis=-1)       # (H, W, R)

    alpha = 1.0 - np.prod(1.0 - A, axis=-1)
    w = A.sum(axis=-1, keepdims=True)
    mix = np.where(w > 1e-9, (A[..., None] * cols).sum(axis=-2) / np.maximum(w, 1e-9), 1.0)

    img = np.concatenate([mix, alpha[..., None]], axis=-1)
    return img, fields


def main():
    rng = np.random.default_rng(5)
    g = np.linspace(0, 1, GRID)
    X, Y = np.meshgrid(g, g)
    img, fields = composite(X, Y)

    fig = plt.figure(figsize=(5.0, 4.3))
    ax = fig.add_subplot(fig.add_gridspec(
        1, 1, left=0.105, right=0.985, top=0.965, bottom=0.265)[0, 0])

    # One image object. Dense colour as vector would be tens of thousands of
    # paths; the axes and everything on top of it stay editable.
    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), interpolation="bilinear",
              zorder=1, rasterized=True)

    # A faint outline per reaction, so a peak has an edge at print size
    # rather than fading into the page.
    for (_, colour, _), z in zip(REACTIONS, fields):
        ax.contour(X, Y, z, levels=[0.35], colors=[colour], linewidths=0.5,
                   alpha=0.35, zorder=2)

    # The assay: a sample over the plane that knows nothing about the peaks,
    # coloured by what it MEASURED -- activity for the reaction of interest.
    # Grey is inactive, not missing: every dot is a variant that was built
    # and assayed, and most of them do nothing.
    name_i, colour_i, peaks_i = REACTIONS[OF_INTEREST]
    dot_cmap = LinearSegmentedColormap.from_list(
        "dot", ["#AEB5BC", "#7C8894", colour_i])

    P = rng.uniform(0.02, 0.98, size=(N_DOTS, 2))
    act = field(P[:, 0], P[:, 1], peaks_i)
    # A fractional power, so a dot on a peak's SHOULDER already shows
    # colour. Under a linear map only the handful of dots within a hair of a
    # summit tinted at all, and the panel read as grey dots with three
    # exceptions. The peaks are narrow in x by construction, so few random
    # dots land dead centre and the shoulders are where the story is.
    shade = np.clip(act, 0.0, 1.0) ** 0.40
    order = np.argsort(act)          # actives drawn last, over the greys
    # Dark thin edge rather than white: the fill runs from light grey, and a
    # white edge on a light fill disappears -- doubly so once the background
    # is transparent and the figure may land on any colour.
    ax.scatter(P[order, 0], P[order, 1], c=shade[order], cmap=dot_cmap,
               vmin=0.0, vmax=1.0, s=20, edgecolors="#374151",
               linewidths=0.45, zorder=4)

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
                      ls="none",
                      label=name + (" (assayed)" if i == OF_INTEREST else ""))
               for i, (name, c, _) in enumerate(REACTIONS)]
    # Below the x-label, not beside it: at bbox -0.045 the legend row and
    # the axis label occupied the same band and overlapped.
    ax.legend(handles=handles, fontsize=fs(6.6), loc="upper center",
              bbox_to_anchor=(0.5, -0.075), ncol=4, frameon=False,
              handletextpad=0.35, columnspacing=1.0, borderpad=0.0)

    # A strip for the dots. The reaction legend says which peak is which;
    # this says what the dot COLOURS mean, and those are different
    # questions -- one row conflating them would have to pretend a gradient
    # is a single swatch.
    cax = fig.add_axes([0.330, 0.060, 0.40, 0.022])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
               cmap=dot_cmap, origin="lower", extent=(0, 1, 0, 1),
               rasterized=True)
    cax.set_xticks([])
    cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_color(INK["muted"])
        sp.set_linewidth(0.6)
    cax.text(-0.04, 0.5, "inactive", transform=cax.transAxes,
             fontsize=fs(6.3), ha="right", va="center", color=INK["muted"])
    cax.text(1.04, 0.5, "active", transform=cax.transAxes, fontsize=fs(6.3),
             ha="left", va="center", color=INK["muted"])
    cax.set_title(f"measured activity, {name_i}", fontsize=fs(6.3),
                  color=INK["muted"], pad=3)

    # Transparent: no figure patch, no axes patch. The landscape image
    # carries its own alpha (see `composite`), so the peaks stay solid while
    # everything around them is see-through.
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    cax.patch.set_alpha(0.0)
    save(fig, "concept_chem_seq", dpi=900, tight=False)


if __name__ == "__main__":
    main()
