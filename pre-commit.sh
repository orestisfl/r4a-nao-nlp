#!/bin/bash
function check_fmt() {
    if [[ $1 =~ \.yaml$ ]]; then
        prettier --check --parser yaml || exit $?
    elif [[ $1 =~ \.json$ ]]; then
        prettier --check --parser json || exit $?
    elif [[ $1 =~ \.pyi?$ ]]; then
        tee >([[ "$(isort --check-only --multi-line=3 --trailing-comma --force-grid-wrap=0 \
            --use-parentheses --line-width=88 -q -df -)" ]]) |
            black --check --pyi --py36 -q - || exit $?
    elif [[ $1 =~ \.sh$ ]]; then
        shfmt -d -s -i 4 || exit $?
    fi
}

if ! [[ "$VIRTUAL_ENV" ]]; then
    echo 'No virtual env detected'
    source .venv/bin/activate || exit $?
fi

# lowercase 'd' filter means exclude deleted
git diff --cached --name-only --diff-filter=d | while read FILE; do
    echo "$FILE"
    git show ":$FILE" | check_fmt "$FILE" || exit $?
done || exit $?

chronic python -m pytest -vv -s $(git ls-files 'test*.py')
