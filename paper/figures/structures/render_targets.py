#!/usr/bin/env python3
"""Render the three AME backbones the k sweep was run on, coloured at k=4.

One image per backbone: the fold coloured by its four fragments, the motif
(the residues RFdiffusion2 holds fixed) drawn as sticks in its own fragment's
colour, and the ligand in grey.

This is Figure 1's claim made on the real targets rather than on a borrowed
protein. The motif is what the benchmark scores, and under the same
equal-count split it does not sit inside one fragment:

    M0097   A26 -> 1, A59 -> 2, A61 -> 2, A93 -> 3     3 of 4 fragments
    M0255   A55 -> 2, A89 -> 2, A91 -> 3, A100 -> 3    2 of 4
    M0315   A90 -> 2, A124 -> 3                        2 of 4

So every recombination in this paper is cutting through an active site, not
around one. That is why retention is the quantity the figure reports.

    uv run python paper/figures/structures/render_targets.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from palette import FRAGMENT  # noqa: E402
from render import PX, STYLE, hexrgb, pymol_run  # noqa: E402

AME = HERE.parents[2] / "experiments" / "ame"
OUT = HERE / "png"

# backbone -> (stem, motif residue numbers)
TARGETS = {
    "M0097": ("run_M0097_1ctt_cond9_14", [26, 59, 61, 93]),
    "M0255": ("run_M0255_1mg5_cond17_66", [55, 89, 91, 100]),
    "M0315": ("run_M0315_1ey3_cond18_25", [90, 124]),
}
TARGET_PX = 2400


def find_pdb(stem):
    """-> the target PDB. The three do not all live in targets12."""
    for d in ("targets12", "targets"):
        p = AME / d / f"{stem}-atomized-bb-True.pdb"
        if p.exists():
            return p
    sys.exit(f"no target pdb for {stem}")


def ca_residues(pdb):
    return sorted({int(l[22:26]) for l in pdb.read_text().splitlines()
                   if l.startswith("ATOM") and l[12:16].strip() == "CA"})


def bounds4(pdb):
    r = ca_residues(pdb)
    e = [round(i * len(r) / 4) for i in range(5)]
    return [(r[e[i]], r[e[i + 1] - 1]) for i in range(4)]


def frag_of(bs, n):
    for i, (lo, hi) in enumerate(bs):
        if lo <= n <= hi:
            return i
    return None


def render(site):
    stem, motif = TARGETS[site]
    pdb = find_pdb(stem)
    bs = bounds4(pdb)
    lines = [f"load {pdb}, t", "hide everything", "show cartoon, t and polymer",
             f"@{STYLE}", "set ray_opaque_background, 0",
             "set cartoon_transparency, 0",
             "set stick_radius, 0.18"]
    for i, (lo, hi) in enumerate(bs):
        r, g, b = hexrgb(FRAGMENT[i])
        lines.append(f"set_color f{i}, [{r:.4f}, {g:.4f}, {b:.4f}]")
        lines.append(f"color f{i}, t and polymer and resi {lo}-{hi}")
    # the motif, as sticks in its own fragment's colour
    for n in motif:
        f = frag_of(bs, n)
        lines.append(f"show sticks, t and resi {n} and not (name C+N+O)")
        if f is not None:
            lines.append(f"color f{f}, t and resi {n}")
    # the ligand, grey, so the site reads as a site
    lines += ["show sticks, t and not polymer",
              "color grey45, t and not polymer",
              "set stick_radius, 0.13, t and not polymer"]
    sel = "+".join(str(n) for n in motif)
    lines += ["orient t and polymer", "turn x, -8", "zoom t, 3",
              f"png {OUT / f'target_{site}.png'}, width={TARGET_PX}, "
              f"height={TARGET_PX}, dpi=300, ray=1"]
    f = OUT / f"_t{site}.pml"
    f.write_text("\n".join(lines) + "\n")
    r = pymol_run(f)
    if r.returncode != 0:
        sys.exit(f"pymol failed on {site}:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    print(f"  rendered target_{site}  motif {sel} -> fragments "
          f"{sorted({frag_of(bs, n) + 1 for n in motif})}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for site in TARGETS:
        render(site)


if __name__ == "__main__":
    main()
