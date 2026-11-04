#!/usr/bin/env python3
"""Render a fold split into k fragments, for the Figure 1 schematic.

The subject is 1TIM (triosephosphate isomerase, chain A, 247 residues) --
deliberately NOT the PETase this project optimises. Figure 1 is about what a
combinatorial library IS; using the target here would imply the idea is
specific to it, and the PETase is saved for the results.

Style lives in style_modern.pml and is shared by every protein figure, so a
change there propagates rather than being re-tuned per figure.

PyMOL cannot emit usable vector output for cartoons, so these are PNGs at
2200px -- roughly 550 dpi at the size Figure 1 uses them. Everything else in
the figure stays vector.

    uv run python paper/figures/structures/render.py
    uv run python paper/figures/structures/render.py --only whole
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from palette import FRAGMENT  # noqa: E402

PDB = HERE / "pdb" / "2fx2_A.pdb"
STYLE = HERE / "style_modern.pml"
GHOST = HERE / "style_ghost.pml"
OUT = HERE / "png"
PX = 2200


def residues():
    """-> sorted CA residue numbers actually present (the file has a gap)."""
    nums = []
    for line in PDB.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            nums.append(int(line[22:26]))
    return sorted(set(nums))


def bounds(k):
    """-> k contiguous, near-equal residue ranges over the residues present.

    Split on the ORDER of residues rather than on numbering, so a gap in the
    PDB does not hand one fragment fewer residues than its neighbours.
    """
    r = residues()
    edges = [round(i * len(r) / k) for i in range(k + 1)]
    return [(r[edges[i]], r[edges[i + 1] - 1]) for i in range(k)]


def hexrgb(h):
    return tuple(int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))


def muted(hex_c, f=0.55):
    """-> `hex_c` blended toward light grey, for the recessive centre copy."""
    r, g, b = hexrgb(hex_c)
    return tuple(v + (0.88 - v) * f for v in (r, g, b))


def script(sel_lines, out_png, k, ghost=False):
    defs = []
    for i in range(k):
        r, g, b = hexrgb(FRAGMENT[i])
        defs.append(f"set_color frag{i}, [{r:.4f}, {g:.4f}, {b:.4f}]")
    return "\n".join([
        f"load {PDB}, prot",
        "hide everything",
        "show cartoon, prot",
        f"@{STYLE}",
        *(["@" + str(GHOST)] if ghost else []),
        *defs,
        *sel_lines,
        "orient prot",
        "turn x, -8",
        "zoom prot, 2",
        f"png {out_png}, width={PX}, height={PX}, dpi=300, ray=1",
    ])


def run(text, tag):
    f = OUT / f"_{tag}.pml"
    f.write_text(text)
    # -k: do NOT read ~/.pymolrc. A personal rc silently applies to every
    # render -- this one sets cartoon_transparency 0.45 -- so without it the
    # paper's figures depend on whose machine built them.
    r = subprocess.run(["pymol", "-cqk", str(f)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"pymol failed on {tag}:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    print(f"  rendered {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bs = bounds(a.k)
    print(f"{PDB.stem}, {len(residues())} residues -> {bs}")

    # ONE PyMOL session for every view. The exploded panel only works if the
    # isolated fragments share a camera with the whole fold -- rendered in
    # separate sessions they each get their own `orient`, and a reader cannot
    # map a fragment back to where it sits. So the view is set once, up front,
    # and nothing after it touches the camera.
    lines = [f"load {PDB}, prot", "hide everything", "show cartoon, prot",
             f"@{STYLE}",
             # The style sets an opaque background for standalone renders.
             # These are composited, so each frame must be transparent --
             # otherwise every image is a white square that paints over
             # whatever it overlaps, and alpha-based trimming sees ink
             # everywhere.
             "set ray_opaque_background, 0"]
    for i in range(a.k):
        r, g, b = hexrgb(FRAGMENT[i])
        lines.append(f"set_color frag{i}, [{r:.4f}, {g:.4f}, {b:.4f}]")
        r, g, b = muted(FRAGMENT[i])
        lines.append(f"set_color dim{i}, [{r:.4f}, {g:.4f}, {b:.4f}]")
    lines += ["orient prot", "turn x, -8", "zoom prot, 2"]

    def png(tag):
        return f"png {OUT / f'{tag}.png'}, width={PX}, height={PX}, dpi=300, ray=1"

    # Full colour, whole fold.
    lines += ["show cartoon, prot"]
    lines += [f"color frag{i}, resi {lo}-{hi}" for i, (lo, hi) in enumerate(bs)]
    lines.append(png("whole"))

    # The recessive centre of the exploded panel: same fold, colours pulled
    # toward grey so the isolated fragments around it carry the colour.
    lines += [f"color dim{i}, resi {lo}-{hi}" for i, (lo, hi) in enumerate(bs)]
    lines.append(png("dim"))

    # One fragment at a time, alone, in full colour and in the SAME frame.
    for i, (lo, hi) in enumerate(bs):
        lines += ["hide everything",
                  f"show cartoon, prot and resi {lo}-{hi}",
                  f"color frag{i}, resi {lo}-{hi}",
                  png(f"iso{i+1}")]

    # A colourless copy for the schematic's "backbone" input.
    lines += ["hide everything", "show cartoon, prot", f"@{GHOST}",
              "color grey72, prot", png("grey")]

    f = OUT / "_all.pml"
    f.write_text("\n".join(lines) + "\n")
    r = subprocess.run(["pymol", "-cqk", str(f)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"pymol failed:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    for t in ["whole", "dim", "grey"] + [f"iso{i+1}" for i in range(a.k)]:
        print(f"  rendered {t}")


if __name__ == "__main__":
    main()
