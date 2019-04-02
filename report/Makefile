.PHONY: all indent clean

all:
	latexmk

indent:
	mkdir -p /tmp/thesis
	find . -type f \( -name '*.tex' -o -name '*.sty' -o -name '*.tikz' -o -name '*.bib' \) -print0 | parallel -P 0 -0 ./indent.sh

clean:
	git clean -fdX --exclude='!thesis.pdf' .
