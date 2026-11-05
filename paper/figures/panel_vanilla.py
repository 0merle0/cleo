"""Vanilla ProteinMPNN against the post-trained policy, on recombinants.

Both arms are measured on the SAME quantity: sequences decoded from MPNN, cut
into k=4, shuffled across parents, reassembled, and folded. Not on parents.
A policy can look good on parents and still produce a library that falls
apart when its fragments are mixed, which is the failure this whole project
is about, so parents are not what either arm is scored on here.

    vanilla    r1panel/*_baseline  -- stock ProteinMPNN sampled at
               T = 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, POOLED
    post       fragk/*_fragk4      -- the same backbone after GRPO
               post-training against the RF3/Boltz/Protenix pool

Both recombinant sets are AF3-scored, and AF3 is in neither training loop.

THE VANILLA POINT IS NOT ONE TEMPERATURE, and it is not a Pareto front.
Matching every recombinant slot back to the parent carrying that exact
subsequence (1024 slots per backbone, 0 unmatched) gives the mix actually
used:

    M0097   T=0.1 28.9%  0.2 23.3%  0.3 21.7%  0.5 15.8%  0.7 9.2%  1.0 1.1%
    M0255   T=0.1 35.8%  0.2 18.8%  0.3 15.1%  0.5 20.5%  0.7 7.3%  1.0 2.3%
    M0315   T=0.1 22.7%  0.2 17.9%  0.3 23.4%  0.5 22.2%  0.7 13.2% 1.0 0.7%

The parent pool was uniform across the six temperatures; this skew toward low
T is entirely the passing filter, and the slot distribution tracks the
passing-parent distribution to within a point or two. So the grey marker is
the whole sweep's filtered union -- a STRONGER baseline than any single
temperature, not a weaker one.

A genuine per-temperature front cannot be recovered from this data: only
1-2% of recombinants are consistent with a single temperature, the other 98%
mix fragments across temperatures. Building one needs recombination re-run
from single-temperature fragment pools, about 4,600 AF3 folds.

DIVERSITY IS RAREFIED, which is the point of care in this panel. Distinct
substitutions are counted over PASSING recombinants, and an arm that passes
more often has more passing sequences to find substitutions in -- so counting
raw would hand the better arm a diversity win for free. Each backbone's two
arms are therefore subsampled to a common number of survivors before
counting, so a horizontal difference means genuinely wider sequence coverage
and not just a larger sample.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[1]
AME = CLEO / "experiments" / "ame"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AME))
from analyze_selection2 import as_matrix  # noqa: E402
from palette import INK, PALETTE, SEQUENTIAL, fs  # noqa: E402

BB = {"M0097": "run_M0097_1ctt_cond9_14",
      "M0255": "run_M0255_1mg5_cond17_66",
      "M0315": "run_M0315_1ey3_cond18_25"}
MARK = {"M0097": "o", "M0255": "s", "M0315": "^"}
# Grey for stock ProteinMPNN; for the post-trained arm a step DARKER than
# the palette blue (SEQUENTIAL[3], not PALETTE["blue"]), so a filled marker
# holds its weight against the pale centre of the gradient.
ARMC = {"vanilla": PALETTE["gray"], "post": SEQUENTIAL[3]}


def _disc(hex_c, px=160, light=0.62):
    """-> an RGBA radial-gradient disc: pale at the centre, `hex_c` at the rim.

    matplotlib has no gradient marker, so the marker is a small image. A disc
    is built once per colour and reused, which keeps the SVG to one embedded
    bitmap per arm rather than one per point.
    """
    r, g, b = (int(hex_c.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    y, x = np.mgrid[-1:1:px * 1j, -1:1:px * 1j]
    d = np.clip(np.hypot(x, y), 0, 1)
    inside = d <= 1.0
    # t=0 at the centre -> lightened; t=1 at the rim -> the colour itself
    t = d ** 1.35
    out = np.zeros((px, px, 4))
    for i, c in enumerate((r, g, b)):
        pale = c + (1 - c) * light
        out[..., i] = pale + (c - pale) * t
    # feather the last pixel ring so the disc does not alias against the axes
    out[..., 3] = np.clip((1.0 - d) * px / 6, 0, 1) * inside
    return out


_DISC = {}


def gradient_marker(ax, x, y, hex_c, size_pts=13.0, lw=1.1, zorder=5):
    """Draw one gradient-filled marker with an outline at a data point."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    if hex_c not in _DISC:
        _DISC[hex_c] = _disc(hex_c)
    im = _DISC[hex_c]
    ax.add_artist(AnnotationBbox(
        OffsetImage(im, zoom=size_pts / im.shape[1]), (x, y),
        frameon=False, box_alignment=(0.5, 0.5), pad=0, zorder=zorder,
        annotation_clip=False))
    # The outline is a hollow marker, not a patch. `ms` is in points, the
    # same unit the disc is placed at, so the ring registers on the disc
    # exactly -- a Circle patch would need its radius converted out of data
    # coordinates and would go elliptical whenever the axes aspect changed.
    ax.plot([x], [y], "o", ms=size_pts, mfc="none", mec=hex_c, mew=lw,
            zorder=zorder + 1)
    return size_pts


def _recomb(site, arm):
    """-> (scored recombinants, label) for one arm of one backbone."""
    bb = BB[site]
    if arm == "vanilla":
        f = AME / "r1panel" / "recomb" / f"{bb}_baseline_scored.csv"
    else:
        f = AME / "fragk" / "recomb" / f"{bb}_fragk4_scored.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)


def _subs(seqs, ref):
    M = as_matrix(list(seqs))
    return len({(j, M[i, j]) for i in range(M.shape[0])
                for j in range(M.shape[1]) if M[i, j] != ref[j]})


def collect(draws=200, rng=None):
    """-> rows of (site, arm, success, n_pass, subs_rarefied)."""
    rng = rng or np.random.default_rng(0)
    out = []
    for site in BB:
        arms = {a: _recomb(site, a) for a in ("vanilla", "post")}
        if any(v is None for v in arms.values()):
            continue
        keep = {a: d[d.rfd2_any_pass].drop_duplicates("sequence")
                .sequence.tolist() for a, d in arms.items()}
        n = min(len(v) for v in keep.values())
        # One reference per BACKBONE, not per arm: the column-wise consensus
        # of both arms' survivors pooled. A per-arm reference would measure
        # each arm's spread about its own centre and could not be compared.
        pooled = keep["vanilla"] + keep["post"]
        M = as_matrix(pooled)
        ref = np.array([max(set(c), key=list(c).count) for c in M.T])
        for a, d in arms.items():
            s = (float(np.mean([_subs(rng.choice(keep[a], n, replace=False),
                                      ref) for _ in range(draws)]))
                 if n else float("nan"))
            out.append(dict(site=site, arm=a,
                            success=100 * float(d.rfd2_any_pass.mean()),
                            n_pass=len(keep[a]), n_rare=n, subs=s))
    return pd.DataFrame(out)


def panel_vanilla(d, ax):
    """Success against diversity, vanilla and post-trained, per backbone."""
    for site in sorted(set(d.site)):
        g = d[d.site == site].set_index("arm")
        if len(g) < 2:
            continue
        v, p = g.loc["vanilla"], g.loc["post"]
        # The arrow is the move, and it is NOT the same move on every
        # backbone. M0097 and M0255 go up and to the right -- more passing
        # recombinants AND wider coverage. M0315 goes up and to the LEFT:
        # +48.8 points of success for 193 fewer distinct mutations. So the
        # panel shows that post-training buys success without collapsing
        # diversity on 2 of 3, and trades some diversity for it on the third.
        # Drawing the arrow rather than two loose points is what makes that
        # per-backbone difference visible instead of averaged away.
        ax.annotate("", xy=(p.subs, p.success), xytext=(v.subs, v.success),
                    arrowprops=dict(arrowstyle="-|>", color=INK["ghost"],
                                    lw=1.3, shrinkA=7, shrinkB=9))
        for arm, r in (("vanilla", v), ("post", p)):
            ax.plot(r.subs, r.success, MARK[site], ms=8, zorder=4,
                    mfc=ARMC[arm], mec=ARMC[arm], mew=1.6,
                    label=site if arm == "post" else None)
    ax.set_xlabel("distinct mutations (rarefied)")
    ax.set_ylabel("recombinants passing (\\%)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=INK["grid"], lw=0.7)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="none", ms=7,
                      mfc=ARMC["vanilla"], mec=ARMC["vanilla"],
                      label="vanilla, $T$ pooled"),
               Line2D([], [], marker="o", ls="none", ms=7,
                      mfc=ARMC["post"], mec=ARMC["post"],
                      label="post-trained")]
    handles += [Line2D([], [], marker=MARK[s], ls="none", ms=7, mfc="none",
                       mec=PALETTE["gray"], label=s) for s in sorted(BB)]
    ax.legend(handles=handles, frameon=False, fontsize=fs(6.6),
              loc="lower right", labelspacing=0.3, handletextpad=0.4)


if __name__ == "__main__":
    print(collect().to_string(index=False))
