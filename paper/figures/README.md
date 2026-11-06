# Figures

Each figure is one self-contained SVG in `svg/`, built from the script of the
same name. **The SVG is the editable deliverable** -- open it in Inkscape,
change what you like, save. Rebuild everything with:

    cd paper && make figures

## Editing in Inkscape

Three things were set up so this actually works:

**Text is text, not outlines.** matplotlib defaults to `svg.fonttype: "path"`,
which converts every label to filled shapes. The figure looks identical and is
completely uneditable -- fixing a typo means redrawing the glyph. `palette.py`
sets `"none"`, so labels come out as real `<text>` you can select and retype.

**The font will not shift on you.** With `fonttype: none` the SVG carries a
font *name* and the editor resolves it. If matplotlib laid out with a different
font than Inkscape renders, every label moves. Checked: Arial is installed at
`/usr/share/fonts/windows/arial.ttf` and is what matplotlib resolves, and the
fallback stack (Liberation, Nimbus, DejaVu) is metrically compatible, so a
machine without Arial substitutes and the layout still holds.

**Dense scatters are rasterised.** Figure 3's strip plots are ~3,300 points per
panel. As vectors that is ~6,600 separate objects in Inkscape -- slow to open
and miserable to select around. `rasterized=True` collapses each cloud to one
image object; axes, ticks and text stay vector and editable. Do this for any
new panel with more than a few hundred marks.

If you add a figure, call `figio.save(fig, "name")` rather than
`fig.savefig(...)`. It enforces SVG, warns when a figure is authored wider than
the 6.5 in page (a label that reads 7 pt in the script arrives at 4 pt if the
figure is scaled down), and writes the PDF alongside.

## What is here

| script | figure |
|---|---|
| `concept_library.py` | Fig 1 -- the conceptual panel |
| `fig2_libraries.py` | Fig 2 -- computational results, 3 backbones |
| `fig3_momi.py` | Fig 3 -- the PETase library |
| `figS_posthoc.py` | Fig S1 -- trained-at-k vs post-hoc |
| `concept_loop.py` | Fig 2 panel a, imported by fig2 |
| `panel_vanilla.py`, `ame_fragk.py`, `schematic.py` | panel/helper modules |
| `palette.py`, `figio.py` | house typography and the one save path |
| `momi_libraries_snapshot.py`, `momi_fragments.py` | Fig 3 data prep |
| `panel_cost.py` | prints the cost numbers quoted in the text |

Superseded exploration is in `archive/`, with a README explaining what each
script was and which ones have open questions attached.

## Figure 3 is gated to the second PETase library

`momi_fragments.py` and `fig3_momi.py` both filter to `second_big_library`.
This is not cosmetic: the two libraries share 5 parts out of 3,629, and when
panel b was drawn over both, slots 2 and 3 read 8.1 and 7.0 mean mutations
against the true 4.4 and 4.4. Every table `fig3_momi.py` reads goes through one
`only()` helper that exits if the filter leaves nothing, so a new panel cannot
quietly skip the step.
