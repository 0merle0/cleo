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

PDB = HERE / "pdb" / "2fx2_A.pdb"
STYLE = HERE / "style_modern.pml"
GHOST = HERE / "style_ghost.pml"
OUT = HERE / "png"
# 6000 px. The protein prints about 1.86 inches wide, so 3600 was already
# ~1930 dpi on the page and 6000 is ~3200 -- both far past what print needs.
# The reason to go higher is on-screen zoom, not the printed figure. If an
# edge looks soft it is antialiasing, not pixel count: see style_modern.pml.
PX = 6000


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

# --- the active site ------------------------------------------------------
#
# 2FX2 is a flavodoxin, and its cofactor pocket makes the figure's point
# without any invention: FMN is hydrogen-bonded by residues from THREE
# DIFFERENT QUARTERS of the chain. Measured on the deposited structure
# (chain A + FMN, pdb/2fx2_site.pdb), closest polar contact per residue:
#
#   fragment 1   Ser10 2.58, Thr11 2.73, Thr12 2.68, Gly13 3.16, Asn14 2.71
#   fragment 2   Ser58 2.61, Thr59 2.63, Trp60 3.32, Asp62 3.36
#   fragment 3   Asp95 3.15, Tyr100 3.13, Cys102 2.90
#   fragment 4   -- nothing within 4.5 A
#
# One residue per contacting fragment is drawn, so the three colours in the
# zoom are three fragments and nothing else. Fragment 4 contributes no
# contact, which the caption should say rather than the figure imply.
SITE_PDB = HERE / "pdb" / "2fx2_site.pdb"
SITE_PX = 2600
SITE_RES = [(12, 0, "OG1", "O3P"),    # Thr12  fragment 1
            (58, 1, "OG", "O1P"),     # Ser58  fragment 2
            (95, 2, "N", "O2")]       # Asp95  fragment 3, backbone donor


def site():
    """Render the FMN pocket: three fragments' residues, H-bonds drawn."""
    bs = bounds(4)
    # Only the three contacting loops are drawn, not the whole fold.
    #
    # The first attempt showed the entire cartoon at cartoon_transparency
    # 0.72 as a backdrop. Two things went wrong: ray_trace_mode 1 renders
    # transparency badly and left black patches across the ribbon, and even
    # faded the fold buried the cofactor it was supposed to frame. Three short
    # loops give the same context and leave the site visible.
    LOOPS = "9-16+56-63+93-102"
    lines = [f"load {SITE_PDB}, site", "hide everything",
             f"show cartoon, site and polymer and resi {LOOPS}", f"@{STYLE}",
             "set ray_opaque_background, 0",
             "set cartoon_transparency, 0",
             "set cartoon_tube_radius, 0.22",
             "set stick_radius, 0.16",
             "set dash_color, black", "set dash_width, 2.2",
             "set dash_gap, 0.22", "set dash_length, 0.30",
             "set label_size, 0", "set ray_trace_mode, 1"]
    for i, (lo, hi) in enumerate(bs):
        r, g, b = hexrgb(FRAGMENT[i])
        lines.append(f"set_color frag{i}, [{r:.4f}, {g:.4f}, {b:.4f}]")
        lines.append(f"color frag{i}, site and polymer and resi {lo}-{hi}")
    lines += ["show sticks, resn FMN", "color grey45, resn FMN",
              "set stick_radius, 0.13, resn FMN"]
    sel = "+".join(str(n) for n, _, _, _ in SITE_RES)
    lines += [f"show sticks, site and polymer and resi {sel}"]
    for n, f, at, fat in SITE_RES:
        lines.append(f"color frag{f}, site and resi {n}")
        lines.append(
            f"distance hb{n}, site and resi {n} and name {at}, "
            f"resn FMN and name {fat}")
    lines += ["hide labels",
              f"orient resn FMN or (site and resi {sel})",
              "turn x, -10", "turn y, 25", "zoom resn FMN, 6.0",
              f"png {OUT / 'site.png'}, width={SITE_PX}, "
              f"height={SITE_PX}, dpi=300, ray=1"]
    f = OUT / "_site.pml"
    f.write_text("\n".join(lines) + "\n")
    r = pymol_run(f)
    if r.returncode != 0:
        sys.exit(f"pymol failed:\n{r.stdout}\n{r.stderr}")
    f.unlink()
    print("  rendered site")


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
    ap.add_argument("--site", action="store_true",
                    help="render only the FMN pocket, then stop")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.variants:
        variants(a.variants, a.k)
        return
    if a.site:
        site()
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
