"""Shared color palette for figures.

Mirrors the LaTeX colors in preamble.tex and the named colors in the author's
~/.pymolrc, so a figure, a structure render and the rendered paper all agree.

**This is the author's own "good"/IPD family, not an invented one.** It was
adopted after rendering twelve candidate palettes on the same structure; these
read best, and they also measure better than the set they replaced -- provided
they are used IN ORDER.

Order matters because a document uses N colors at a time, so what counts is
the worst pair among however many a figure uses, not the whole family's worst
pair. Greedily ordered from `good_blue`, measuring OKLab separation under
simulated protanopia, deuteranopia and tritanopia:

    1 good_blue       #4499E7
    2 golden_yellow   #FAC72C   first 2: CVD 31.9
    3 ipd_purple      #B41F85   first 3: CVD 16.9
    4 fire_red        #E44A3E   first 4: CVD 13.3   <- FRAGMENT
    5 vivid_orange    #FF8C00   first 5: CVD  5.8   below floor
    6 good_teal       #28B0C1   first 6: CVD  3.2   below floor
    7 soft_green      #AAC32F   first 7: CVD  1.9   below floor

The first four are comfortably clear of the dE 8 target. Past four the family
runs out: these are the conflicting pairs, and they cannot be fixed by
reordering, only by not using both members together.

    good_blue / good_teal        3.2
    soft_green / vivid_orange    1.9
    soft_green / golden_yellow   4.3
    golden_yellow / vivid_orange 5.8

So a five-or-more-way figure needs something other than hue to separate the
extra series -- position, a marker, a direct label. `soft_green` and
`good_teal` are the two to avoid pairing with anything; they are fine alone
against a neutral background.

For an ORDERED quantity -- k, a sweep, anything with a magnitude -- use
SEQUENTIAL. Categorical colors on an ordered axis throw away the ordering.
"""

# The family, in the order they should be used.
ORDERED = [
    ("blue",   "#4499E7"),   # good_blue
    ("yellow", "#FAC72C"),   # golden_yellow
    ("green",  "#AAC32F"),   # soft_green
    ("red",    "#E44A3E"),   # fire_red
    ("purple", "#B41F85"),   # ipd_purple / magenta -- the 5th slot
    ("orange", "#FF8C00"),   # vivid_orange   -- see conflicts above
    ("teal",   "#28B0C1"),   # good_teal      -- see conflicts above
]

PALETTE = dict(ORDERED)
PALETTE["gray"] = "#6B7280"

# Fragment slot identity, in order. Slot i gets FRAGMENT[i].
FRAGMENT = [hex for _, hex in ORDERED[:4]]

# One hue, light -> dark, for ordered quantities such as k.
#
# Built around PALETTE["blue"], which sits at index 2 EXACTLY -- the ramp is
# the categorical blue extended in both directions rather than a separate blue
# that happens to look similar. Same OKLab hue (249.1 deg) throughout, with
# chroma tapering toward both ends as a sequential ramp should. If the
# palette's blue changes, regenerate this rather than hand-editing it.
SEQUENTIAL = ["#BADBFD", "#7FBDFA", "#4499E7", "#216CAD", "#004A80", "#002D52"]
assert SEQUENTIAL[2] == PALETTE["blue"], "ramp must pass through the palette blue"

# Neutrals for context: un-highlighted structure, grid lines, rules.
INK = {"muted": "#9CA3AF", "grid": "#E5E7EB", "ghost": "#D1D5DB"}


# ---------------------------------------------------------------- typography
#
# Applied at import. Every figure script imports this module, so this is the
# one place that makes the document's figures agree on type -- the alternative
# is a style call each script can forget, which is how a document ends up with
# three fonts in it.
#
# Arial, because the journal asks for a sans and Arial is the one actually
# installed here (Helvetica is not; matplotlib would silently fall back to
# DejaVu Sans and the figures would not match the request). The fallbacks after
# it are metric-compatible, so a machine without Arial gets the same layout.
MIN_FONT = 7.0

# The floor above is in POINTS ON THE PRINTED PAGE, which is the only place a
# font size means anything. It is therefore only true if the figure is
# authored at the width it is printed at.
#
# This document is letter with 1-inch margins, so \textwidth is 6.5in. A
# figure built at figsize=(11, h) and included at width=\textwidth is scaled
# by 6.5/11 = 0.59 on its way in, and a label set at "7pt" arrives at 4.1pt.
# Every figure here is therefore sized PAGE_WIDTH_IN wide (or a stated
# fraction of it), and structures placed with `place` are sized in the same
# points.
PAGE_WIDTH_IN = 6.5

_SANS = ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]


def fs(size):
    """-> `size`, clamped up to MIN_FONT.

    Use this for EVERY fontsize in a figure rather than writing the number
    directly. A floor that lives in a comment gets broken the next time a
    label needs to squeeze into a corner; this one cannot be.
    """
    return max(float(size), MIN_FONT)


def _apply():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _SANS,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        # Black for everything that is text. Spines and grid stay neutral --
        # they are rules, not labels.
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "font.size": MIN_FONT,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": MIN_FONT,
    })


_apply()
