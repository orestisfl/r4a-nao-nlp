#!/usr/bin/env python
import os
from setuptools import setup, find_packages

if os.getenv("DOWNLOAD_NEURALCOREF"):
    import atexit

    def post_install():
        import sys
        import subprocess

        subprocess.call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz",
            ],
            env=os.environ.copy(),
        )

        # en_coref_md model is larger than spacy's en model, so we don't want to do this:
        # import spacy
        # spacy.cli.link("en_coref_md", "en", force=True)
        # spacy.cli.link("en_coref_md", "en_core_web_sm", force=True)

    atexit.register(post_install)

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")) as f:
    long_description = f.read()

setup(
    name="r4a-nao nlp",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "spacy<=2.0.13",  # https://github.com/explosion/spaCy/issues/2852
        "allennlp",
        "snips-nlu",
    ],
    package_data={"r4a_nao_nlp": ["engine.tar.gz"]},
    author="Orestis Floros",
    author_email="orestisf1993@gmail.com",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robotics-4-all/2017_B_NLP_robotics",
    classifiers=[
        "Programming Language :: Python :: 3.6",  # snips-nlu needs python < 3.7
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Linux",
    ],
    entry_points={
        "console_scripts": ["r4a_nao_nlp = r4a_nao_nlp.__main__:entry_point"]
    },
)
