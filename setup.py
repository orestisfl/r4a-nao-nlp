#!/usr/bin/env python
import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")) as f:
    long_description = f.read()

setup(
    name="r4a-nao nlp",
    version="0.0.1",
    author="Orestis Floros",
    author_email="orestisf1993@gmail.com",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robotics-4-all/2017_B_NLP_robotics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",  # snips-nlu needs python < 3.7
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Linux",
    ],
)
