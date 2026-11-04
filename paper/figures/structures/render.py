#!/usr/bin/env python3
"""Render the PETase fold, split into k fragments, for the Figure 1 schematic.

Writes transparent PNGs that the figure script composites. PyMOL cannot emit
usable vector output for cartoons, so these are raster by necessity; everything
else in the figure stays vector, and `figio` still enforces SVG for the figure
itself. 1600px at the sizes used here is ~500 dpi on the page.

The style matches figs/frag_animation.gif -- flat fill with a thin dark outline
(ray_trace_mode 1) -- because that is the visual language the project already
uses for this idea, and a reader who has seen the repo should recognise it.

    uv run python paper/figures/structures/render.py            # all views
    uv run python paper/figures/structures/render.py --only whole
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from palette import FRAGMENT  # noqa: E402

PDB = CLEO / "templates" / "structure" / "momi_petase_reference_structure.pdb"
TEMPLATE = HERE / "render_fragments.pml.in"
OUT = HERE / "png"
L = 209                      # CA residues in the reference structure


def bounds(k):
    """-> k equal 1-indexed inclusive residue ranges, as the method splits."""
    edges = [round(i * L / k) for i in range(k + 1)]
    return [(edges[i] + 1, edges[i + 1]) for i in range(k)]


def pml(colors, sel_lines, out_png):
    t = TEMPLATE.read_text()
    cdefs = "\n".join(f"set_color frag{i}, [{r/255:.4f}, {g/255:.4f}, {b/255:.4f}]"
                      for i, (r, g, b) in enumerate(colors))
    return (t.replace("__PDB__", str(PDB))
             .replace("__OUT__", str(out_png))
             .replace("__COLORS__", cdefs)
             .replace("__SEL__", "\n".join(sel_lines)))


def run(script, tag):
    f = OUT / f"_{tag}.pml"
    f.write_text(script)
    r = subprocess.run(["pymol", "-cq", str(f)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"pymol failed on {tag}:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    print(f"  rendered {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cols = [tuple(int(FRAGMENT[i].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
            for i in range(a.k)]
    bs = bounds(a.k)

    jobs = {}
    # The whole fold, every fragment in its own colour.
    jobs["whole"] = [f"color frag{i}, resi {lo}-{hi}" for i, (lo, hi) in enumerate(bs)]
    # One image per fragment: that fragment coloured, the rest ghosted away.
    # Used for the parts rack, so each slot shows what it contributes.
    for i, (lo, hi) in enumerate(bs):
        jobs[f"frag{i+1}"] = [
            "color grey85, prot",
            "set cartoon_transparency, 0.82, prot",
            f"color frag{i}, resi {lo}-{hi}",
            f"set cartoon_transparency, 0.0, resi {lo}-{hi}",
        ]
    # The uncut parent, for the "before" state.
    jobs["grey"] = ["color grey75, prot"]

    for tag, sel in jobs.items():
        if a.only and tag != a.only:
            continue
        run(pml(cols, sel, OUT / f"{tag}.png"), tag)


if __name__ == "__main__":
    main()
