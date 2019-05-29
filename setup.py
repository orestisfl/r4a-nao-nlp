#!/usr/bin/env python
import os

from setuptools import find_packages, setup


def extras_add_all(extras):
    extras["all"] = sorted(
        set(package for package_list in extras.values() for package in package_list)
    )
    return extras


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")) as f:
    long_description = f.read()

setup(
    name="r4a-nao nlp",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "spacy>=2.1.0,<2.1.4",  # https://github.com/huggingface/neuralcoref/issues/164
        "allennlp>=0.7.2",
        "snips-nlu>=0.19.1",
        "networkx",
    ],
    extras_require=extras_add_all(
        {
            "plots": ["matplotlib", "adjustText"],
            "ecore": ["pyecore", "PyYAML", "braceexpand"],
            "train": ["braceexpand"],  # snips-nlu-en
            "CoreNLP-server": ["requests"],
            "neuralcoref": ["neuralcoref >= 4.0"],
        }
    ),
    python_requires=">=3.7",
    package_data={"r4a_nao_nlp": ["engine.tar.gz", "transformations.json"]},
    author="Orestis Floros",
    author_email="orestisf1993@gmail.com",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robotics-4-all/2017_B_NLP_robotics",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Linux",
    ],
    entry_points={
        "console_scripts": [
            "r4a_nao_nlp = r4a_nao_nlp.__main__:enter_cli_main",
            "r4a_nao_nlp_train = r4a_nao_nlp.__main__:enter_train_main",
            "r4a_nao_nlp_generate_yaml = r4a_nao_nlp.__main__:enter_generate_yaml",
        ]
    },
)

# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
