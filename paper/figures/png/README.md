Derived PNG renders, for viewing figures on GitHub and in review threads where
SVG does not display.

**Nothing here is a source file.** The SVGs in `../svg/` are authoritative (see
`../figio.py`); the paper build converts those to PDF, and never reads this
directory. Regenerate rather than edit:

    uv run python paper/figures/ame_frontier.py
    inkscape --export-type=png --export-dpi=200 \
      -o paper/figures/png/ame_frontier.png paper/figures/svg/ame_frontier.svg
