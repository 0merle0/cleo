#!/usr/bin/env python3
"""Render a fold split into k fragments, for the Figure 1 schematic.

The subject is 1TIM (triosephosphate isomerase, chain A, 247 residues) --
deliberately NOT the PETase this project optimises. Figure 1 is about what a
combinatorial library IS; using the target here would imply the idea is
specific to it, and the PETase is saved for the results.

Style lives in style_modern.pml and is shared by every protein figure, so a
change there propagates rather than being re-tuned per figure.

PyMOL cannot emit usable vector output for cartoons, so these are PNGs.
Everything else in the figure stays vector.

    uv run python paper/figures/structures/render.py
    uv run python paper/figures/structures/render.py --only whole
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from palette import FRAGMENT  # noqa: E402

# 2FX2, flavodoxin, chain A, 147 residues.
#
# Chosen by measuring secondary structure across a dozen candidates (39.5%
# helix, 32.7% strand), and at 147 aa its k=4 fragments are 37 residues --
# inside the 20-50 band the method recommends.
#
# It was briefly replaced by trypsin (2PTN) so that panel (c) could show a
# His/Asp/Ser catalytic triad, which flavodoxin does not have. That zoom was
# dropped, and the protein came back with it: trypsin's k=4 fragments are 56
# residues, outside the band. pdb/2ptn_A.pdb is kept -- the triad spans three
# fragments (His57/Asp102/Ser195 on fragments 1/2/4) and that is worth
# keeping to hand if a later figure wants it.
PDB = HERE / "pdb" / "2fx2_A.pdb"
STYLE = HERE / "style_modern.pml"
GHOST = HERE / "style_ghost.pml"
OUT = HERE / "png"
# 3000 px. The fold now prints ~0.9 inch wide beside its zoom, not 1.9, so
# this is still ~3300 dpi on the page. If an edge looks soft it is
# antialiasing, not pixel count: see style_modern.pml.
PX = 3000


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


def pymol_run(script_path):
    """Run `pymol -cqk` on a script, OUTSIDE whatever venv is active.

    The Debian `pymol` wrapper is a shell script that calls `python3`. Under
    `uv run` that resolves to the project venv, which has no pymol module, and
    every render dies with "No module named 'pymol'" even though PyMOL is
    installed system-wide. Stripping the venv from the child's environment
    lets the wrapper find the interpreter it was built against.

    -k: do NOT read ~/.pymolrc. A personal rc otherwise applies to every
    render -- the one on the original machine set cartoon_transparency 0.45 --
    so figures would silently depend on whose account built them.
    """
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    if venv:
        bad = str(Path(venv) / "bin")
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep) if p != bad)
    return subprocess.run(["pymol", "-cqk", str(script_path)],
                          capture_output=True, text=True, env=env)


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
    r = pymol_run(f)
    if r.returncode != 0:
        sys.exit(f"pymol failed on {tag}:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    print(f"  rendered {tag}")


# Variants for the lab-in-the-loop panel. Printed ~46 pt (0.64 in) wide, so
# 2000 px is ~3100 dpi there -- the 6000 px used for the hero structure would
# be eight times the pixels for a tenth the size.
VAR_PX = 2000

def shade(hex_c, f):
    """-> `hex_c` lightened toward white by (1-f). f=1 is the base colour."""
    r, g, b = hexrgb(hex_c)
    return tuple(v + (1 - v) * (1 - f) for v in (r, g, b))


def variants(n, k):
    """Render `n` copies of the fold, each a different combination of tints.

    Same backbone, same camera, same fragment boundaries -- only the per-slot
    tint changes, which is exactly what a recombinant is: one fragment per
    slot, drawn from different parents. Rendering them in one session keeps
    the camera identical, so the reader reads the colour difference and not a
    pose difference.
    """
    import numpy as np
    rng = np.random.default_rng(11)
    bs = bounds(k)
    lines = [f"load {PDB}, prot", "hide everything", "show cartoon, prot",
             f"@{STYLE}", "set ray_opaque_background, 0",
             "orient prot", "turn x, -8", "zoom prot, 2"]
    for v in range(n):
        # Spread the tints: every variant gets at least one near-full and one
        # clearly pale slot, so no two read as the same design at thumbnail
        # size.
        fs_ = rng.permutation(np.linspace(0.42, 1.0, k))
        for i, (lo, hi) in enumerate(bs):
            r, g, b = shade(FRAGMENT[i], fs_[i])
            lines.append(f"set_color v{v}s{i}, [{r:.4f}, {g:.4f}, {b:.4f}]")
            lines.append(f"color v{v}s{i}, resi {lo}-{hi}")
        lines.append(f"png {OUT / f'var{v+1}.png'}, width={VAR_PX}, "
                     f"height={VAR_PX}, dpi=300, ray=1")
    f = OUT / "_var.pml"
    f.write_text("\n".join(lines) + "\n")
    r = pymol_run(f)
    if r.returncode != 0:
        sys.exit(f"pymol failed:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    for v in range(n):
        print(f"  rendered var{v+1}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--variants", type=int, default=0,
                    help="render only N recombinant variants, then stop")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.variants:
        variants(a.variants, a.k)
        return
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
    r = pymol_run(f)
    if r.returncode != 0:
        sys.exit(f"pymol failed:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    for t in ["whole", "dim", "grey"] + [f"iso{i+1}" for i in range(a.k)]:
        print(f"  rendered {t}")


if __name__ == "__main__":
    main()
