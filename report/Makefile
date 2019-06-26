.PHONY: all indent clean images
graphs := $(patsubst %.gv,%.pdf,$(wildcard images/graphs/*.gv))
pgfs := $(patsubst %.py,%.pgf,$(wildcard images/*.py))
pgfs := $(filter-out images/plot_utils.pgf, $(pgfs))

all: images
	latexmk

images: $(graphs) $(pgfs)

images/graphs/%.pdf: images/graphs/%.gv
	dot -Tsvg -Gbgcolor=transparent -O $<
	inkscape -D -z --file=$<.svg --export-pdf=$@ --export-latex
	rm -f $<.svg

images/%.pgf: images/%.py
	python $<

indent:
	mkdir -p /tmp/thesis
	find . -type f \( -name '*.tex' -o -name '*.sty' -o -name '*.tikz' -o -name '*.bib' \) -print0 | parallel -P 0 -0 ./indent.sh

clean:
	git clean -fdX --exclude='!*.pdf' --exclude='!*.pdf_tex' --exclude='!*.pgf'
