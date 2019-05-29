#1/bin/bash -x

# curl -s 'https://course.fast.ai/setup/colab' | bash

function to_36() {
    # __future__.annotations not supported in 3.6
    sed -i '/__future__ import annotations/d' $(find . -name '*.py')
    sed -i 's/if TYPE_CHECKING/if True/' $(find . -name '*.py')
    sed -i 's/python_requires=">=3.7"/python_requires=">=3.6"/' setup.py
    sed -i '/assert TYPE_CHECKING/d' $(find . -name '*.py')

    # Replace types with their name as a string
    sed -i 's/^\(.*\) =.*/\1 = "\1"/' r4a_nao_nlp/typing.py
    # Do the same for `from … import …`, avoid cyclic imports.
    # Exclude `from typing` imports.
    sed -i '/^from typing/! s/^from.* import \(.*\)$/\1 = "\1"/' r4a_nao_nlp/typing.py
}

function start_corenlp() {
    if [[ ! -d stanford-corenlp-full-2018-10-05 ]]; then
        apt-get install -y openjdk-8-jdk-headless unzip &
        wget 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip'
        wait
        unzip -o stanford-corenlp-full-2018-10-05.zip
        rm stanford-corenlp-full-2018-10-05.zip
    fi

    # Use port 9001 by default
    sed -i 's/localhost:9000/localhost:9001/' $(find . -name '*.py')

    cd stanford-corenlp-full-2018-10-05

    nohup java -Dorg.slf4j.simpleLogger.defaultLogLevel=debug -mx9g -cp '*:lib/*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 100000 -threads 2 -port 9001 &
    disown

    cd ..
}

function download_spacy() {
    python -m spacy download en_core_web_md
    python -m spacy download en_core_web_sm
    python -m spacy download en
}

function install_apex() {
    pip show -qqq apex && return

    rm -rf apex
    git clone --depth 1 'https://github.com/nvidia/apex'
    cd apex
    pip install -v --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' .
}

python -c 'import sys; sys.exit(int(sys.version_info.major < 3))' || exit 1
python -c 'import sys; v = sys.version_info; sys.exit(int(v.major == 3 and v.minor < 7))' || to_36

# https://github.com/huggingface/neuralcoref#spacystringsstringstore-size-changed-error
pip install neuralcoref --no-binary neuralcoref
pip install -U '.[all]'
python -m snips_nlu download en &
download_spacy &

# Optional, fixes warning when loading pytorch_pretrained_bert:
# > Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.
install_apex &

start_corenlp

wait
