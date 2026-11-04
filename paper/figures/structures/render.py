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

# Bovine trypsin, 2PTN chain A, 223 residues.
#
# It replaced 2FX2 (flavodoxin) when the figure needed a catalytic triad.
# Flavodoxin has none -- its functional site is a cofactor pocket -- so a
# His/Asp/Ser triad drawn on it would have been invented. Trypsin's is real
# and, better, it SPANS THE CUT: under the same equal-count split,
#
#     His57  -> fragment 1      His-Asp 2.67 A
#     Asp102 -> fragment 2      His-Ser 3.02 A
#     Ser195 -> fragment 4      Asp-Ser 6.77 A (they do not contact)
#
# Three of the four fragments carry one catalytic residue each, which is the
# figure's whole point about where boundaries fall.
#
# The cost: at 223 residues the k=4 fragments are 56 residues, just outside
# the 20-50 band the method recommends, where flavodoxin's were 37. The
# caption states the number rather than repeating the old claim.
PDB = HERE / "pdb" / "2ptn_A.pdb"
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

# --- the catalytic triad --------------------------------------------------
#
# Sticks only, no cofactor and no cartoon: the zoom is three side chains and
# the two hydrogen bonds between them. Each residue takes its own fragment's
# colour, so the three colours in the zoom ARE three fragments.
#
# An earlier version of this panel drew flavodoxin's FMN pocket with the
# cofactor in grey. It was replaced, not just recoloured -- see the note on
# PDB above for why the protein changed.
SITE_PX = 2200
TRIAD = [(57, 0, "HIS"),      # fragment 1
         (102, 1, "ASP"),     # fragment 2
         (195, 3, "SER")]     # fragment 4
# The two real catalytic hydrogen bonds. Asp102 and Ser195 are 6.77 A apart
# and are NOT joined -- the charge relay runs through the histidine, and
# drawing a third dash would assert a contact that is not there.
TRIAD_HB = [((57, "ND1"), (102, "OD2")),
            ((57, "NE2"), (195, "OG"))]


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


def site():
    """Render the catalytic triad: three side chains, two hydrogen bonds.

    Sticks only. No cartoon and no cofactor -- the panel beside it already
    shows the fold, and anything else in this frame competes with the three
    colours that carry the point.
    """
    sel = "+".join(str(n) for n, _, _ in TRIAD)
    lines = [f"load {PDB}, prot", "hide everything", f"@{STYLE}",
             "set ray_opaque_background, 0",
             "set stick_radius, 0.20",
             "set dash_color, black", "set dash_width, 2.4",
             "set dash_gap, 0.24", "set dash_length, 0.32",
             "set label_size, 0",
             f"show sticks, prot and resi {sel} and not (name C+N+O)"]
    for n, f, _ in TRIAD:
        r, g, b = hexrgb(FRAGMENT[f])
        lines.append(f"set_color tri{n}, [{r:.4f}, {g:.4f}, {b:.4f}]")
        lines.append(f"color tri{n}, prot and resi {n}")
    for i, ((n1, a1), (n2, a2)) in enumerate(TRIAD_HB):
        lines.append(f"distance hb{i}, prot and resi {n1} and name {a1}, "
                     f"prot and resi {n2} and name {a2}")
    lines += ["hide labels",
              f"orient prot and resi {sel}", "turn x, -8",
              f"zoom prot and resi {sel}, 1.6",
              f"png {OUT / 'triad.png'}, width={SITE_PX}, "
              f"height={SITE_PX}, dpi=300, ray=1"]
    f = OUT / "_site.pml"
    f.write_text("\n".join(lines) + "\n")
    r = pymol_run(f)
    if r.returncode != 0:
        sys.exit(f"pymol failed:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    print("  rendered triad")


def sitemark():
    """Render the triad as spheres in the WHOLE FOLD's camera.

    This is not a figure, it is a measurement. The zoom box drawn over the
    fold in panel (c) has to sit where the active site actually is, and the
    only reliable way to know that in image pixels is to render the site
    alone through the same camera and find its centroid. A hand-picked
    fraction of the frame goes silently wrong the moment the view changes.
    """
    sel = "+".join(str(n) for n, _, _ in TRIAD)
    lines = [f"load {PDB}, prot", "hide everything", f"@{STYLE}",
             "set ray_opaque_background, 0",
             f"show spheres, prot and resi {sel}",
             "color black, prot",
             "orient prot", "turn x, -8", "zoom prot, 2",
             f"png {OUT / 'sitemark.png'}, width={PX}, height={PX}, "
             "dpi=300, ray=1"]
    f = OUT / "_mark.pml"
    f.write_text("\n".join(lines) + "\n")
    r = pymol_run(f)
    if r.returncode != 0:
        sys.exit(f"pymol failed:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    print("  rendered sitemark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--variants", type=int, default=0,
                    help="render only N recombinant variants, then stop")
    ap.add_argument("--site", action="store_true",
                    help="render only the FMN pocket, then stop")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.variants:
        variants(a.variants, a.k)
        return
    if a.site:
        site()
        sitemark()
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
