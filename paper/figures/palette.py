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
    ("purple", "#B41F85"),   # ipd_purple / magenta
    ("red",    "#E44A3E"),   # fire_red
    ("orange", "#FF8C00"),   # vivid_orange   -- see conflicts above
    ("teal",   "#28B0C1"),   # good_teal      -- see conflicts above
    ("green",  "#AAC32F"),   # soft_green     -- see conflicts above
]

PALETTE = dict(ORDERED)
PALETTE["gray"] = "#6B7280"

# Fragment slot identity, in order. Slot i gets FRAGMENT[i].
FRAGMENT = [hex for _, hex in ORDERED[:4]]

# One hue, light -> dark, for ordered quantities such as k.
SEQUENTIAL = ["#A5C3DF", "#7FA5CC", "#5A87B4", "#3C6A99", "#2A4F78", "#15314D"]

# Neutrals for context: un-highlighted structure, grid lines, rules.
INK = {"muted": "#9CA3AF", "grid": "#E5E7EB", "ghost": "#D1D5DB"}
