.PHONY: all indent clean images
pgfs := $(patsubst %.py,%.pgf,$(wildcard images/*.py))
pgfs := $(filter-out images/plot_utils.pgf, $(pgfs))

all:
	latexmk

images: $(pgfs)

images/%.pgf: images/%.py
	python $<

indent:
	mkdir -p /tmp/thesis
	find . -type f \( -name '*.tex' -o -name '*.sty' -o -name '*.tikz' -o -name '*.bib' \) -print0 | parallel -P 0 -0 ./indent.sh

clean:
	git clean -fdX --exclude='!*.pgf'
