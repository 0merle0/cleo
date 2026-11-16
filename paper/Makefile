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
#
# Two levels, because $(SVG_DIR)/main/ holds the paper's four numbered
# figures and $(SVG_DIR)/ holds the panel sources they are assembled from. A
# single-level wildcard silently skipped main/, which is the half that goes
# in the document -- so both are globbed and the PDF tree mirrors the SVG one.
SVGS := $(wildcard $(SVG_DIR)/*.svg) $(wildcard $(SVG_DIR)/main/*.svg) \
        $(wildcard $(SVG_DIR)/fig1_panels/*.svg) \
        $(wildcard $(SVG_DIR)/fig2_panels/*.svg)
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
#
# The fig1..fig4 set is the story draft's figures, blocked out in the draft's
# own four-figure scheme: real panels where the data exists, labelled
# placeholders where it does not. See figure_plan_story.md. The older
# concept_*/fig2_libraries/fig3_momi scripts stay because their panels are
# what gets dropped into those slots.
FIG_SCRIPTS = momi_fragments concept_library concept_loop fig2_libraries \
              fig3_momi figS_posthoc \
              fig1_method fig2_petase_design fig3_platform fig4_rounds \
              fig1_panels fig2_panels

figures:
	@for f in $(FIG_SCRIPTS); do \
	  printf '  %-20s ' $$f; \
	  (cd .. && uv run python paper/figures/$$f.py >/dev/null 2>&1) \
	    && echo OK || echo FAILED; \
	done
	@echo "figures/svg/main/ holds the paper's four figures -- open them in Inkscape."
	@echo "figures/svg/ holds the panel sources they are assembled from."
	@echo "figures/svg/fig1_panels/ holds Figure 1 as one SVG per panel."

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
	@mkdir -p "$(dir $@)"
	inkscape "$<" --export-type=pdf --export-filename="$@"

$(PDF_DIR):
	mkdir -p $(PDF_DIR)

watch:
	$(LATEX) -pvc main.tex

clean:
	latexmk -C
	rm -rf $(PDF_DIR)
