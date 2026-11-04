"""Shared color palette for figures.

Mirrors the LaTeX colors defined in preamble.tex. Import this from any
plotting script so figures stay visually consistent with the rendered paper.

**The four base colors are colorblind-safe as a set; keep them that way.**
They were checked by simulating protanopia, deuteranopia and tritanopia and
measuring OKLab separation between every pair, not by eye. The original
cleoOrange (#D97A29) and cleoGreen (#4A8C5A) failed that check: a protanope
saw orange and green at dE 5.1, below the hard floor of 6, and green/red sat
at 6.3 for a deuteranope. Both were nudged by the smallest amount that clears
dE 8 while staying recognisably the same hue, so the document did not have to
be recolored:

    cleoOrange  #D97A29 -> #D47F12
    cleoGreen   #4A8C5A -> #47966C

Blue and red were already safe and are unchanged. Worst pair is now orange vs
green at 8.1 (protan); every pair clears 15 for normal vision.

**FRAGMENT is four colors, and that is a decision, not an oversight.** Six
categorical hues that survive color-vision deficiency need high chroma and
come out neon, which is wrong for this document; every muted fifth color
tested against the base four collapsed below the floor (#7E57A8 at 4.8,
#6A4C93 at 1.8, #2F8E9E at 3.9). Four is also what the method recommends --
4-6 fragments of 20-50 aa -- and what Figure 1 shows. A figure that genuinely
needs five or six slots can lean on position along the chain as secondary
encoding, which fragments carry for free; it should not invent a fifth hue.

For an ORDERED quantity -- k itself, a sweep, anything with a magnitude --
use SEQUENTIAL, not FRAGMENT. Categorical colors on an ordered axis throw
away the ordering.
"""

PALETTE = {
    "blue":   "#2E5C8A",
    "orange": "#D47F12",
    "green":  "#47966C",
    "red":    "#B23A3A",
    "gray":   "#6B7280",
}

# Fragment slot identity, in order. Slot i gets FRAGMENT[i].
FRAGMENT = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]

# Softened fills for large areas -- a cartoon, a filled block -- where the
# saturated colors are too heavy. How soft is a measurement, not a taste:
# lightening in OKLab L and cutting chroma costs CVD separation fast.
#
#     +0.06 L, 0.88 chroma   CVD 6.8   <- this set, legal with 2nd encoding
#     +0.10 L, 0.80 chroma   CVD 6.0   on the hard floor
#     +0.18 L, 0.62 chroma   CVD 4.6   below it, no remedy
#     +0.24 L, 0.52 chroma   CVD 3.5   below it, no remedy
#
# 6.8 sits in the 6-8 band, which is permitted ONLY when something other than
# hue also carries identity. In the figures that use these, two things do:
# position along the chain, and the matching saturated FRAGMENT color as the
# stroke. Do not use these as the sole cue, and do not lighten them further --
# the next step down is already unreadable for a protanope.
FRAGMENT_SOFT = ["#456D97", "#E19547", "#63A781", "#BF5551"]

# The only fifth color that clears the floor against the base four (dE 8.1).
# Bright violet, and it does not sit comfortably with the others -- use it
# only when a fifth slot genuinely cannot be avoided.
FRAGMENT_5 = "#9B5DE5"

# One hue, light -> dark, for ordered quantities such as k.
SEQUENTIAL = ["#A5C3DF", "#7FA5CC", "#5A87B4", "#3C6A99", "#2A4F78", "#15314D"]

# Neutrals for context: un-highlighted structure, grid lines, rules.
INK = {"muted": "#9CA3AF", "grid": "#E5E7EB", "ghost": "#D1D5DB"}
