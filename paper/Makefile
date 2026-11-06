.PHONY: pdf clean-pdf notebook library campaigns all svg2pdf clean watch figures

LATEX = latexmk -pdf -interaction=nonstopmode -halt-on-error
SVG_DIR = figures/svg
PDF_DIR = figures/pdf

# Figures are SVG-only by rule. Every figure script writes into $(SVG_DIR) via
# figures/figio.py::save(), which rejects any other extension; this target then
# derives the PDFs LaTeX includes. SVG stays the single source of truth and
# figures/pdf/ is disposable -- `make clean` removes it, a rebuild restores it.
# A script writing a PDF directly would put a file into the build that nothing
# knows how to regenerate.
#
# Dense point clouds must set rasterized=True on the artist. As vectors they
# carry one path per point: the AME drift panel was 6.4 MB before rasterising
# and is 1.3 MB after, with axes and text still vector and selectable.
#
# Find all SVGs and map them to PDF outputs.
SVGS := $(wildcard $(SVG_DIR)/*.svg)
PDFS := $(patsubst $(SVG_DIR)/%.svg,$(PDF_DIR)/%.pdf,$(SVGS))

# `make pdf` builds the paper only. The experimental record is a separate
# document with a separate reader -- see the header of notebook.tex for why the
# two are not one file. `make all` builds both.
# `make figures` rebuilds every figure the paper uses, from its script, into
# figures/svg/. That is the editable deliverable: each figure is one
# self-contained SVG that opens in Inkscape with its text as real text.
#
# Order matters only for fig3: momi_fragments regenerates the per-part table
# it reads, and that table is gated to the second PETase library.
FIG_SCRIPTS = momi_fragments concept_library concept_loop fig2_libraries \
              fig3_momi figS_posthoc

figures:
	@for f in $(FIG_SCRIPTS); do \
	  printf '  %-20s ' $$f; \
	  (cd .. && uv run python paper/figures/$$f.py >/dev/null 2>&1) \
	    && echo OK || echo FAILED; \
	done
	@echo "figures/svg/ is the editable set; open any of them in Inkscape."

all: pdf notebook library campaigns

pdf: svg2pdf
	$(LATEX) main.tex

# The forward-building document: title, abstract, and the figures that are
# settled. See the header of clean.tex for why it is separate from main.tex.
clean-pdf: svg2pdf
	$(LATEX) clean.tex

notebook: svg2pdf
	$(LATEX) notebook.tex

# The library arc is its own record: a different question from the AME
# programme, and the arc that retired three of its claims.
library: svg2pdf
	$(LATEX) library_notebook.tex

# The narrative deck: one panel per campaign, for choosing what the paper
# argues before it argues it. See the header of campaigns.tex.
campaigns: svg2pdf
	$(LATEX) campaigns.tex

svg2pdf: $(PDFS)

$(PDF_DIR)/%.pdf: $(SVG_DIR)/%.svg | $(PDF_DIR)
	@which inkscape > /dev/null || (echo "inkscape not found — install it or convert SVGs by hand into $(PDF_DIR)/"; exit 1)
	inkscape "$<" --export-type=pdf --export-filename="$@"

$(PDF_DIR):
	mkdir -p $(PDF_DIR)

watch:
	$(LATEX) -pvc main.tex

clean:
	latexmk -C
	rm -rf $(PDF_DIR)
