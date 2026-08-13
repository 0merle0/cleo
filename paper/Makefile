.PHONY: pdf svg2pdf clean watch

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

pdf: svg2pdf
	$(LATEX) main.tex

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
