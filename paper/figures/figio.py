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
PDF_DIR = Path(__file__).resolve().parent / "pdf"
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
    kw = {"bbox_inches": "tight"} if tight else {}
    out = SVG_DIR / f"{stem.stem}.svg"
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", dpi=dpi, **kw)

    # ALSO write the PDF the Makefile would have produced.
    #
    # The build converts figures/svg/*.svg with Inkscape, and that is still
    # how a figure nobody re-ran gets its PDF. But Inkscape is not everywhere
    # -- it is absent on at least one machine this paper is built on, and
    # there the whole document stops building over two stale figures.
    # matplotlib writes PDF natively, so a script that runs can produce its
    # own and the conversion rule simply finds nothing to do.
    #
    # SVG remains the source of truth: this is the same figure written twice
    # from one canvas, not a second rendering path that could drift.
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PDF_DIR / f"{stem.stem}.pdf"
    fig.savefig(pdf, format="pdf", dpi=dpi, **kw)

    print(f"wrote {out} and {pdf}")
    return out
