# TODO: docstrings
from __future__ import annotations

import datetime
import os
import tarfile
from functools import lru_cache
from operator import itemgetter
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Optional, Union

from r4a_nao_nlp import utils

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import JsonDict, Doc

logger = utils.create_logger(__name__)
# XXX: Can use pkg_resources to find distributed resources:
# https://setuptools.readthedocs.io/en/latest/pkg_resources.html
HERE = os.path.abspath(os.path.dirname(__file__))


class Shared:
    def __init__(self):
        logger.debug("Creating shared object")

        # TODO: typing, make some of them less optional
        self.engine = None
        self.srl_predictor = None
        self.coref_predictor = None
        self._spacy = None

        self.srl_cache = {}

    def init(
        self,
        snips_path: Optional[str] = os.path.join(HERE, "engine.tar.gz"),
        srl_predictor_path: Optional[
            str
        ] = "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz",
        coref_predictor_path: Optional[str] = None,
        spacy_lang: Optional[str] = "en",
        neural_coref_model: Optional[str] = "en_coref_md",
    ) -> None:
        logger.debug("Initializing shared resources")

        if snips_path:
            logger.debug("Loading snips engine from %s", snips_path)
            from snips_nlu import SnipsNLUEngine

            if os.path.isdir(snips_path):
                logger.debug("%s is a directory, loading directly", snips_path)
                self.engine = SnipsNLUEngine.from_path(snips_path)
            else:
                with tarfile.open(snips_path, "r:gz") as archive:
                    with TemporaryDirectory() as tmp:
                        archive.extractall(tmp)
                        logger.debug(
                            "Extracted to temporary dir %s, loading from there", tmp
                        )
                        self.engine = SnipsNLUEngine.from_path(
                            os.path.join(tmp, "engine")
                        )

        if srl_predictor_path:
            logger.debug("Loading allennlp srl model from %s", srl_predictor_path)
            from allennlp.predictors.predictor import Predictor
            from allennlp.common.file_utils import cached_path

            self.srl_predictor = Predictor.from_path(cached_path(srl_predictor_path))

        if coref_predictor_path and neural_coref_model:
            logger.warn(
                "Ignoring allennlp coref model %s because of neuralcoref model %s",
                coref_predictor_path,
                neural_coref_model,
            )
        elif coref_predictor_path:
            logger.debug("Loading allennlp coref model from %s", coref_predictor_path)
            from allennlp.predictors.predictor import Predictor
            from allennlp.common.file_utils import cached_path

            self.coref_predictor = Predictor.from_path(
                cached_path(coref_predictor_path)
            )

        if spacy_lang and neural_coref_model:
            logger.debug(
                "Skipping spacy model %s, will load neuralcoref model %s",
                spacy_lang,
                neural_coref_model,
            )
        elif spacy_lang:
            import spacy

            logger.debug("Loading spacy lang %s", spacy_lang)
            self._spacy = spacy.load(spacy_lang)

        if neural_coref_model:
            import spacy

            logger.debug("Loading spacy neuralcoref model %s", neural_coref_model)
            self._spacy = self.neuralcoref = spacy.load(neural_coref_model)

        logger.info("Done loading")

    @lru_cache(maxsize=1024)
    def parse(self, s: str) -> "SnipsResult":
        assert self.engine

        logger.debug("Passing '%s' to snips engine", s)
        result = SnipsResult(self.engine.parse(s))
        logger.debug("Result = '%s'", result)
        return result

    def srl(self, s: str) -> JsonDict:
        assert self.srl_predictor

        # TODO: just lru cache
        r = self.srl_predictor.predict(s)
        self.srl_cache[s] = r
        return r

    def spacy(self, s: str) -> Doc:
        assert self._spacy

        # TODO: find a better way to deal with problems with whitespace.
        # Eliminate newlines and multiple whitespace.
        s = " ".join(s.split())

        logger.debug("Passing '%s' to spacy", s)
        return self._spacy(s)

    def coref(self, s: str) -> Union[Doc, JsonDict]:
        if self.neuralcoref:
            return self.neuralcoref(s)
        else:
            assert self.coref_predictor

            return self.coref_predictor.predict(s)


shared = Shared()

# TODO: https://docs.python.org/3/library/dataclasses.html
class SnipsResult(tuple):
    """Immutable convenience object that holds the output of the SNIPS engine"""

    __slots__ = []

    def __new__(cls, parsed: Optional[JsonDict] = None):
        if (
            parsed is None
            or parsed["intent"] is None
            or parsed["intent"]["intentName"] is None
        ):
            score = 0.0
            name = None
            slots = tuple()
        else:
            score = parsed["intent"]["probability"]
            name = parsed["intent"]["intentName"]
            slots = tuple(SnipsSlot(slot) for slot in parsed["slots"])
        return tuple.__new__(cls, (score, name, slots))

    score = property(itemgetter(0))
    name = property(itemgetter(1))
    slots = property(itemgetter(2))

    # TODO
    def to_eobject(self):
        from r4a_nao_nlp import ecore

        return ecore.snips_dict_to_eobject(self)

    def __bool__(self):
        return self.name is not None

    def __float__(self):
        return float(self.score)

    def __iter__(self):
        return iter(self.slots)

    def __lt__(self, other: object):
        if not isinstance(other, SnipsResult):
            return NotImplemented

        return self.score < other.score

    def __str__(self):
        return "{intent}({args})".format(
            intent=self.name, args=",".join(str(slot) for slot in self.slots)
        )


class SnipsSlot(tuple):
    """Immutable convenience object that holds the snips output of a single slot"""

    __slots__ = []

    def __new__(cls, parsed: JsonDict):
        r = range(parsed["range"]["start"], parsed["range"]["end"])
        value = _resolve_value(parsed["value"])
        return tuple.__new__(cls, (r, value, parsed["entity"], parsed["slotName"]))

    range = property(itemgetter(0))
    value = property(itemgetter(1))
    entity = property(itemgetter(2))
    name = property(itemgetter(3))

    @property
    def start(self):
        return self.range.start

    @property
    def end(self):
        return self.range.stop

    def __str__(self):
        return "{}={}".format(self.name, self.value)


def _resolve_value(value: JsonDict) -> Union[datetime.timedelta, float, str]:
    # https://github.com/snipsco/snips-nlu-ontology#grammar-entity
    if value["kind"] == "Duration":
        return _resolve_duration(value)
    if value["kind"] in ("Number", "Ordinal", "Percentage"):
        return value["value"]

    # NOTE: Other kinds of entities should be added here when used.
    assert value["kind"] == "Custom"
    return value["value"]


def _resolve_duration(value: JsonDict) -> datetime.timedelta:
    """Convert the parsed value of a snips/duration entity to a datetime timedelta.

    Months and years are converted to their average length in seconds in the Gregorian
    calendar. This way, the same text will always produce models that use the same
    duration values in relevant arguments, regardless of the time of parsing.
    """
    return datetime.timedelta(
        weeks=value["weeks"],
        days=value["days"],
        seconds=value["seconds"]
        + 60 * (value["minutes"] + 15 * value["quarters"])
        + 3600 * value["hours"]
        # 30.436875 days * 24 hours * 3600 seconds
        + 2_629_746 * value["months"]
        # 365.2425 days * 24 hours * 3600 seconds
        + 31_557_600 * value["years"],
    )


# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
