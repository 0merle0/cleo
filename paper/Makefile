.PHONY: pdf svg2pdf clean watch

LATEX = latexmk -pdf -interaction=nonstopmode -halt-on-error
SVG_DIR = figures/svg
PDF_DIR = figures/pdf

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
