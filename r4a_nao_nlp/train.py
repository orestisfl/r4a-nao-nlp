"""Script that prepares data and trains the snips nlu engine."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
from argparse import Namespace
from contextlib import contextmanager
from glob import glob
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterator, List, Optional
from typing.io import TextIO

from r4a_nao_nlp import utils

if TYPE_CHECKING:
    from snips_nlu import SnipsNLUEngine
    from r4a_nao_nlp.typing import JsonDict

logger = utils.create_logger(__name__)


def main(argv: List[str]) -> None:
    arguments = parse_command_line(argv[1:])

    with dest_context_manager(arguments.dest) as converted:
        logger.debug("Converting and saving data to %s", converted)
        convert_data(src=arguments.data, dest=converted)
        dataset = load_dataset(converted)

        json_save(dataset, os.path.join(converted, "dataset.json"))

    logger.debug("Loading Snips")
    import snips_nlu

    snips_nlu.load_resources("en")
    engine = snips_nlu.SnipsNLUEngine()

    logger.debug("Training engine")
    engine.fit(dataset)
    save_engine(engine, arguments.out_engine)

    logger.debug("Done")


def parse_command_line(argv: List[str]) -> Namespace:
    parser = utils.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default="data",
        help="Data folder which contains the intent files.",
    )
    parser.add_argument(
        "--out-converted",
        default=None,
        help=(
            "Where to save the final data used to train the Snips engine. By default, a "
            "temporary directory is used and the files are deleted after being used."
        ),
        dest="dest",
    )
    parser.add_argument(
        "-o",
        "--out-engine",
        default="engine.tar.gz",
        help="Where to store the fitted Snips engine.",
    )
    arguments = parser.parse_args(argv)

    return arguments


@contextmanager
def dest_context_manager(dest: Optional[str]) -> Iterator[str]:
    """Prepare destination directory.

    If `dest` is `None`, a new temporary directory is yielded, which will be deleted on
    context exit. If not, the specified directory is cleared and yielded. It will
    persist on context exit.
    """
    if dest is None:
        with TemporaryDirectory() as tmp:
            yield tmp
    else:
        try:
            shutil.rmtree(dest)
            logger.info("Deleted existing destination directory %s", dest)
        except FileNotFoundError:
            logger.debug("Dest %s did not exist", dest)
        os.makedirs(dest)

        yield dest


def convert_data(src: str, dest: str) -> str:
    """Copy intents, entities and utterances to destination folder.

    Only intents that have corresponding utterances_* files are copied to avoid training
    on intent files without any values. Utterances are brace-expanded. All entities are
    copied to the destination without being modified.
    """
    utterance_files = glob(os.path.join(src, "utterances_*"))
    for filename in utterance_files:
        intent_basename = "intent_{intent_name}.yaml".format(
            intent_name=os.path.basename(filename).split("_")[1]
        )
        with open(os.path.join(src, intent_basename)) as f:
            # XXX: assert that the intent is the last yaml "document" in the file and we
            # only need to append the "utterances:" key to it. Otherwise, we could use a
            # YAML library.
            base_yaml = f.read().strip() + "\nutterances:"
        with open(filename) as f:
            expanded_utterances = "\n".join(
                str('  - "') + line + '"' for line in expand_file(f)
            )
        with open(os.path.join(dest, intent_basename), "w") as f:
            print(base_yaml, file=f)
            print(expanded_utterances, file=f)

    # Copy entity yaml files to destination so that the dataset can be generated from
    # that folder only.
    for filename in glob(os.path.join(src, "entity_*")):
        shutil.copy(filename, os.path.join(dest, os.path.basename(filename)))

    return dest


def expand_file(f: TextIO) -> Iterator[str]:
    """Perform brace expansion on specified file object.

    Uses the `braceexpand` module to expand braces for each line read from the file."""
    from braceexpand import braceexpand

    for line in f:
        for line in braceexpand(line):
            line = line.strip()
            if line:
                yield line


def load_dataset(converted: str) -> JsonDict:
    from snips_nlu.dataset import Dataset

    filenames = glob(os.path.join(converted, "*.yaml"))
    return Dataset.from_yaml_files("en", filenames).json


def json_save(json_dict: JsonDict, filename: str) -> None:
    with open(filename, "w") as f:
        print(json.dumps(json_dict, indent=4, sort_keys=True), file=f)


def save_engine(engine: SnipsNLUEngine, path: str) -> None:
    """Save trained snips nlu engine in a tar.gz archive."""
    with tarfile.open(path, "w:gz") as archive:
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "engine")

            logger.debug("Saving engine to path %s", path)
            engine.persist(path)
            archive.add(path, arcname="engine")


if __name__ == "__main__":
    from r4a_nao_nlp import __main__

    __main__.entry_point(__spec__.name)

# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
