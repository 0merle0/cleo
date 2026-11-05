#!/usr/bin/env python3
"""Render the MoMI PETase parent, coloured by its five library fragments.

Two images:

    momi_fold   the whole 209-aa G20 parent, coloured by slot, catalytic
                triad as sticks, PET substrate in grey
    momi_site   the active site alone -- Ser66/His107/Asp122 and the
                substrate, each residue in its own slot's colour

FRAGMENT BOUNDARIES ARE MEASURED, NOT ASSUMED. Nothing in the data states
them, so they are recovered from the library itself: take designs that swap
exactly ONE slot away from the all-g20 parent and see which positions move.
See momi_fragments.py.

    slot 1    1-40      slot 2   45-84     slot 3   89-126
    slot 4  131-163     slot 5  168-209

The gaps (41-44, 85-88, 127-130, 164-167) are positions no design ever
mutates, so the cut points lie somewhere inside them and the windows above
are the tightest bounds the data supports.

THE TRIAD SPANS TWO SLOTS: Ser66 on slot 2, His107 and Asp122 on slot 3. So
recombining this library reassembles the catalytic machinery across a cut.

    uv run python paper/figures/structures/render_momi.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from palette import FRAGMENT, PALETTE  # noqa: E402
from render import STYLE, hexrgb, pymol_run  # noqa: E402

PDB = HERE.parents[2] / "templates" / "structure" / \
    "momi_petase_reference_structure.pdb"
OUT = HERE / "png"

# measured; see the module docstring
SLOTS = [(1, 40), (45, 84), (89, 126), (131, 163), (168, 209)]
# the palette carries four fragment hues; the fifth slot takes the magenta
# that the palette reserves for exactly this.
COLORS = list(FRAGMENT) + [PALETTE["purple"]]
TRIAD = {66: 1, 107: 2, 122: 2}          # residue -> slot index
PX = 2600


def base(obj="p"):
    lines = [f"load {PDB}, {obj}", "hide everything",
             f"show cartoon, {obj} and polymer", f"@{STYLE}",
             "set ray_opaque_background, 0", "set cartoon_transparency, 0"]
    for i, (lo, hi) in enumerate(SLOTS):
        r, g, b = hexrgb(COLORS[i])
        lines.append(f"set_color s{i}, [{r:.4f}, {g:.4f}, {b:.4f}]")
        lines.append(f"color s{i}, {obj} and polymer and resi {lo}-{hi}")
    # the linkers nobody mutates, in grey, so the five parts read as parts
    gaps = [(SLOTS[i][1] + 1, SLOTS[i + 1][0] - 1) for i in range(4)]
    sel = "+".join(f"{a}-{b}" for a, b in gaps)
    lines.append(f"color grey70, {obj} and polymer and resi {sel}")
    return lines


def run(lines, tag):
    f = OUT / f"_{tag}.pml"
    f.write_text("\n".join(lines) + "\n")
    r = pymol_run(f)
    if r.returncode != 0:
        sys.exit(f"pymol failed on {tag}:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    print(f"  rendered {tag}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = "+".join(str(n) for n in TRIAD)

    # --- the whole fold -------------------------------------------------
    lines = base()
    lines += ["set stick_radius, 0.16",
              f"show sticks, p and resi {sel} and not (name C+N+O)"]
    for n, s in TRIAD.items():
        lines.append(f"color s{s}, p and resi {n}")
    lines += ["show sticks, p and not polymer",
              "color grey45, p and not polymer",
              "set stick_radius, 0.12, p and not polymer",
              "orient p and polymer", "turn x, -10", "zoom p, 3",
              f"png {OUT / 'momi_fold.png'}, width={PX}, height={PX}, "
              "dpi=300, ray=1"]
    run(lines, "momi_fold")

    # --- the active site ------------------------------------------------
    # Sticks and the substrate only. The fold is in the panel beside it;
    # anything more here competes with the three colours that carry the
    # point, which is that the triad is split across two slots.
    lines = [f"load {PDB}, p", "hide everything", f"@{STYLE}",
             "set ray_opaque_background, 0", "set stick_radius, 0.20",
             "set dash_color, black", "set dash_width, 2.2",
             "set dash_gap, 0.24", "set label_size, 0",
             f"show sticks, p and resi {sel} and not (name C+N+O)"]
    for i, c in enumerate(COLORS):
        r, g, b = hexrgb(c)
        lines.append(f"set_color s{i}, [{r:.4f}, {g:.4f}, {b:.4f}]")
    for n, s in TRIAD.items():
        lines.append(f"color s{s}, p and resi {n}")
    lines += ["show sticks, p and not polymer",
              "color grey45, p and not polymer",
              "set stick_radius, 0.13, p and not polymer",
              # the two catalytic hydrogen bonds
              "distance hb1, p and resi 66 and name OG, "
              "p and resi 107 and name NE2",
              "distance hb2, p and resi 107 and name ND1, "
              "p and resi 122 and name OD2",
              "hide labels",
              f"orient p and (resi {sel} or not polymer)",
              "turn x, -8", f"zoom p and resi {sel}, 5.5",
              f"png {OUT / 'momi_site.png'}, width={PX}, height={PX}, "
              "dpi=300, ray=1"]
    run(lines, "momi_site")


if __name__ == "__main__":
    main()
