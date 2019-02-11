# TODO: docstrings
# vim:ts=4:sw=4:expandtab:fo-=t
from __future__ import annotations

import json
import os
import shutil
import tarfile
from argparse import Namespace
from glob import glob
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterator, List
from typing.io import TextIO

from braceexpand import braceexpand
from r4a_nao_nlp import utils
from snips_nlu import SnipsNLUEngine, load_resources
from snips_nlu.dataset import Dataset

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import JsonDict

logger = utils.create_logger(__name__)


def main(argv: List[str]) -> None:
    arguments = parse_command_line(argv[1:])

    original = arguments.data
    converted = convert_data(original)
    dataset = load_dataset(original, converted)

    json_save(dataset["entities"], os.path.join(original, "entities.json"))
    json_save(dataset, os.path.join(converted, "dataset.json"))

    load_resources("en")
    engine = SnipsNLUEngine()
    engine.fit(dataset)
    save_engine(engine, arguments.out_engine)


def parse_command_line(argv: List[str]) -> Namespace:
    parser = utils.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default="data",
        help="Data folder which contains entities.json and intent files.",
    )
    parser.add_argument(
        "-o",
        "--out-engine",
        default="engine.tar.gz",
        help="Where to store the fitted Snips engine.",
    )
    arguments = parser.parse_args(argv)

    return arguments


def convert_data(src: str) -> str:
    src = src.strip("/")
    dest = src + "-converted"
    try:
        shutil.rmtree(dest)
        logger.info("Deleted existing destination directory %s", dest)
    except FileNotFoundError:
        logger.debug("Dest %s did not exist", dest)
    os.makedirs(dest)

    expand_braces(src, dest)
    return dest


def expand_braces(src: str, dest: str) -> None:
    intents = glob(os.path.join(src, "intent") + "*")
    for filename in intents:
        with open(filename) as f:
            result = "\n".join(expand_file(f))
        with open(os.path.join(dest, os.path.basename(filename)), "w") as f:
            print(result, file=f, end="")


def expand_file(f: TextIO) -> Iterator[str]:
    for line in f:
        for line in braceexpand(line):
            line = line.strip()
            if line:
                yield line


def load_dataset(original: str, converted: str) -> JsonDict:
    intents = glob(os.path.join(converted, "intent") + "*")
    dataset = Dataset.from_files("en", intents).json
    entities = os.path.join(original, "entities.json")
    with open(entities) as f:
        dataset["entities"].update(json.load(f))
    return dataset


def json_save(json_dict: JsonDict, filename: str) -> None:
    with open(filename, "w") as f:
        print(json.dumps(json_dict, indent=4, sort_keys=True), file=f)


def save_engine(engine: SnipsNLUEngine, path: str) -> None:
    with tarfile.open(path, "w:gz") as archive:
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "engine")
            engine.persist(path)
            archive.add(path, arcname="engine")
