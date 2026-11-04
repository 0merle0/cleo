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

PDB = HERE / "pdb" / "1tim_A.pdb"
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
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bs = bounds(a.k)
    print(f"1TIM chain A, {len(residues())} residues -> {bs}")

    jobs = {
        "whole": [f"color frag{i}, resi {lo}-{hi}" for i, (lo, hi) in enumerate(bs)],
        "grey": ["color grey72, prot"],
    }
    for i, (lo, hi) in enumerate(bs):
        jobs[f"frag{i+1}"] = ["color grey90, prot",
                              "set cartoon_transparency, 0.85, prot",
                              f"color frag{i}, resi {lo}-{hi}",
                              f"set cartoon_transparency, 0.0, resi {lo}-{hi}"]

    for tag, sel in jobs.items():
        if a.only and tag != a.only:
            continue
        run(script(sel, OUT / f"{tag}.png", a.k, ghost=(tag == "grey")), tag)


if __name__ == "__main__":
    main()
