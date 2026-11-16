#!/usr/bin/env python3
"""Panel placeholders, so a figure's layout can be edited before its data exists.

The story draft points at ~26 panels, about a third of which have no data yet.
The choice is between leaving those figures unmade until the data lands, or
laying out the whole figure now with the empty slots drawn in. The second is
better: the page budget, the panel order and the aspect of every slot are
decisions that can be made and argued about today, by hand, in Inkscape --
and a slot that is already the right shape gets its real plot dropped into it
without the rest of the figure moving.

Every placeholder states, inside the box:

    the panel letter      so it can be referenced in prose immediately
    what it shows         one line, the claim the panel has to carry
    its status            have / derive / collect
    where the data is     the file or experiment, so filling it is mechanical

`paper/figure_plan_story.md` is the same information in table form and is the
authority; these boxes are its rendering.

A placeholder is NOT a blank box. A blank box tells a collaborator nothing and
tells a reviewer less. Anything shipped by accident with these still in it is
loudly and obviously unfinished, which is the intended failure mode.
"""

from matplotlib.patches import FancyBboxPatch

from palette import INK, PALETTE, fs

# have    -- data in hand, this is a build job
# derive  -- data exists, needs a new reduction or script
# collect -- the measurement or the run does not exist yet
STATUS = {
    "have": (PALETTE["green"], "data in hand"),
    "derive": (PALETTE["yellow"], "needs deriving"),
    "collect": (PALETTE["red"], "needs collecting"),
    "built": (PALETTE["blue"], "already built"),
}


def slot(ax, letter, what, status, source, lines=()):
    """Fill `ax` with a labelled placeholder for one panel.

    `what`   one line: the claim this panel carries.
    `status` a key of STATUS.
    `source` where the data comes from, or what has to happen to get it.
    `lines`  optional extra detail, one string per line.
    """
    colour, status_text = STATUS[status]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Rounded, dashed, tinted: three signals that this is scaffolding. A
    # square solid box reads as a real panel with an empty plot in it.
    ax.add_patch(FancyBboxPatch(
        (0.012, 0.012), 0.976, 0.976,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=0.9, linestyle=(0, (4, 3)),
        edgecolor=colour, facecolor=colour + "14",
        transform=ax.transAxes, zorder=0, clip_on=False))

    # Status chip, top right. Short, because the box is small and the colour
    # already carries most of it.
    ax.text(0.965, 0.945, status_text, transform=ax.transAxes,
            fontsize=fs(6.5), color=colour, ha="right", va="top",
            fontweight="bold", zorder=3)

    ax.text(0.05, 0.945, f"({letter})", transform=ax.transAxes,
            fontsize=fs(9), fontweight="bold", color="black",
            ha="left", va="top", zorder=3)

    ax.text(0.05, 0.795, what, transform=ax.transAxes, fontsize=fs(8),
            color="black", ha="left", va="top", zorder=3)

    # Line spacing in POINTS, converted to this box's own axes fraction.
    # Writing it as a fixed fraction instead made the step shrink with the
    # box, so a short placeholder ran its first body line through its title.
    h_in = ax.get_position().height * ax.figure.get_figheight()
    step = (fs(7) * 1.55 / 72.0) / max(h_in, 1e-6)

    y = 0.795 - step * 1.45 * (1 + what.count("\n"))

    # Auto-fit: a placeholder with a long list in a short box ran its last
    # lines through the source line and out of the box. Tighten the step
    # (and, only if that is not enough, the size) rather than overflowing --
    # an overflowing box hides the very detail it exists to carry.
    floor = 0.145
    if lines:
        need = step * len(lines)
        if need > y - floor:
            step = (y - floor) / len(lines)
    size = fs(7)
    if step * h_in * 72.0 < size * 1.12:
        size = max(5.5, step * h_in * 72.0 / 1.12)

    for line in lines:
        ax.text(0.05, y, line, transform=ax.transAxes, fontsize=size,
                color=INK["muted"], ha="left", va="top", zorder=3)
        y -= step

    ax.text(0.05, 0.055, source, transform=ax.transAxes, fontsize=fs(6.5),
            color=colour, ha="left", va="bottom", style="italic", zorder=3)


def panel_letter(fig, ax, letter, dx=-0.055, dy=0.035):
    """A panel letter for a REAL panel, in figure coordinates.

    Placed off the axes rather than inside them, so it does not land on data.
    `ax.get_tightbbox` under-measures rotated text (386.8 px reported against
    400.2 px actual for one y-label here), which is how panel letters end up
    sitting on y-labels -- so this takes an explicit offset rather than
    measuring.
    """
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, f"({letter})", fontsize=fs(9),
             fontweight="bold", ha="left", va="bottom", color="black")
