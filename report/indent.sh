#!/bin/bash

outfile="$(mktemp --suffix=".${1##*.}")"
latexindent --local=myyaml.yaml --cruft=/tmp/thesis --silent -o="$outfile" "$1"
cmp -s "$outfile" "$1" && (
    rm -f "$outfile"
    true
) || mv "$outfile" "$1"
