"""One save path for every figure in the paper, so exports stay consistent.

**Rule: figure scripts export SVG, and only SVG.** The Makefile discovers
everything in `figures/svg/` and converts it to `figures/pdf/` with Inkscape at
build time, so SVG is the single source of truth and the PDFs are derived
artefacts that can be deleted and regenerated. A script that writes a PDF (or a
PNG) directly puts a file into the build that nothing knows how to rebuild.

Import and call `save(fig, "name")` rather than `fig.savefig(...)`:

    from figio import save
    save(fig, "ame_pca")        # -> figures/svg/ame_pca.svg

Beyond enforcing the format this also fixes `dpi`, which is not cosmetic. Dense
point clouds must be rasterised (`rasterized=True` on the scatter) or the vector
PDF carries one path per point -- the AME drift panel was 6.4 MB before, larger
than the rest of the paper combined. Rasterised artists are resolved at save
time, so the dpi set here is what determines their quality; axes, ticks and text
stay vector and remain selectable.
"""

from pathlib import Path

SVG_DIR = Path(__file__).resolve().parent / "svg"
DPI = 300


def save(fig, name, dpi=DPI, tight=True):
    """Write `fig` to figures/svg/<name>.svg. -> the path written.

    `name` may be given with or without a .svg suffix; any other suffix is
    rejected rather than silently corrected, since a script asking for .png is
    a script whose output the Makefile will not pick up.
    """
    stem = Path(name)
    if stem.suffix and stem.suffix.lower() != ".svg":
        raise ValueError(
            f"figures are exported as SVG only; got {stem.suffix!r} for {name!r}. "
            "The Makefile converts figures/svg/*.svg to PDF at build time."
        )
    out = SVG_DIR / f"{stem.stem}.svg"
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", dpi=dpi,
                **({"bbox_inches": "tight"} if tight else {}))
    print(f"wrote {out}")
    return out
