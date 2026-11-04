"""Schematic primitives shared by the paper's concept figures.

Figure 1's experimental loop and Figure 2's training loop are drawn with the
same boxes, arrows, sequence bars and structure placements. They lived in one
script while there was only one schematic; this module exists so the second
one does not fork them, because two copies of `box` drift and then the two
figures stop looking like one document.

Nothing here knows about a particular figure: callers pass their own
coordinates, colours and sizes.
"""

import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from palette import FRAGMENT, PALETTE, fs  # noqa: E402

PNG = HERE / "structures" / "png"


def trimmed(name):
    """-> RGBA array cropped to its non-transparent bounding box."""
    im = mpimg.imread(PNG / f"{name}.png")
    if im.shape[2] == 4:
        ys, xs = np.nonzero(im[:, :, 3] > 0.02)
        if len(ys):
            im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return im


def place(ax, name, xy, width_pts):
    """Draw a structure PNG `width_pts` wide, whatever its pixel size.

    Sized in POINTS, not by OffsetImage's zoom. zoom is per-pixel, so a figure
    written against it silently rescales every structure whenever the render
    resolution changes.
    """
    im = trimmed(name)
    ax.add_artist(AnnotationBbox(
        OffsetImage(im, zoom=width_pts / im.shape[1]), xy,
        frameon=False, box_alignment=(0.5, 0.5)))


# ---------------------------------------------------------------- panel (a)
def bar(ax, x, y, w, h, colors, shades=None, lw=0.7, edge=None):
    """A sequence drawn as k abutting blocks, optionally shade-varied."""
    n = len(colors)
    for i, c in enumerate(colors):
        a = x + w * i / n
        f = 1.0 if shades is None else shades[i]
        # Soft fill, saturated stroke. The soft palette sits in the 6-8 CVD
        # band and is only legal with a second cue -- the stroke is that cue.
        ax.add_patch(FancyBboxPatch(
            (a + 0.004, y), w / n - 0.008, h,
            boxstyle="square,pad=0",
            fc=_shade(c, f),
            ec=edge if edge else FRAGMENT[i % len(FRAGMENT)], lw=0.7,
            zorder=3))


def _shade(hex_c, f):
    """Lighten toward white by (1-f); f=1 is the base colour."""
    r, g, b = (int(hex_c.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
    return tuple(v + (1 - v) * (1 - f) for v in (r, g, b))


def box(ax, xy, w, h, label, sub=None, fc="white", title_size=8.5):
    ax.add_patch(FancyBboxPatch(xy, w, h,
                                boxstyle="square,pad=0",
                                fc=fc, ec=PALETTE["gray"], lw=1.1, zorder=3))
    ax.text(xy[0] + w / 2, xy[1] + h * (0.62 if sub else 0.5), label,
            ha="center", va="center", fontsize=fs(title_size), zorder=4)
    if sub:
        ax.text(xy[0] + w / 2, xy[1] + h * 0.26, sub, ha="center", va="center",
                fontsize=fs(6.8), zorder=4)


def arrow(ax, p0, p1, color=None, rad=0.0, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11,
        color=color or PALETTE["gray"], lw=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=2,
        shrinkA=2, shrinkB=2))
