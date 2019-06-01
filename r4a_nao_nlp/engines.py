# TODO: docstrings
from __future__ import annotations

import atexit
import copy
import datetime
import importlib
import json
import os
import tarfile
from dataclasses import dataclass
from functools import lru_cache, total_ordering
from math import isclose
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Union

from r4a_nao_nlp import utils

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import (
        Doc,
        EObject,
        JsonDict,
        Language,
        SnipsNLUEngine,
        Token,
    )
    from r4a_nao_nlp.utils import before_37

    if before_37():
        SnipsResult = "SnipsResult"
        SnipsSlot = "SnipsSlot"

    SlotValue = Union[datetime.timedelta, float, str]

logger = utils.create_logger(__name__)
# XXX: Can use pkg_resources to find distributed resources:
# https://setuptools.readthedocs.io/en/latest/pkg_resources.html
HERE = os.path.abspath(os.path.dirname(__file__))
QUOTE_STRING = "QUOTE"


class Shared:
    def __init__(self):
        logger.debug("Creating shared object")

        self._engine: Optional[SnipsNLUEngine] = None
        self._spacy: Optional[Language] = None
        self._core_nlp_server_url: Optional[str] = None

        self._transformations: JsonDict = {}

    @utils.timed
    def init(
        self,
        snips_path: Optional[str] = os.path.join(HERE, "engine.tar.gz"),
        transformations: Optional[str] = os.path.join(HERE, "transformations.json"),
        srl_predictor_path: Optional[
            str
        ] = "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz",
        spacy_lang: Optional[str] = "en_core_web_md",
        neuralcoref: bool = True,
        core_nlp_server_url: Optional[str] = "http://localhost:9000",
    ) -> None:
        logger.debug("Initializing shared resources")

        if core_nlp_server_url:
            logger.debug(
                "Connecting to stanford CoreNLP server %s", core_nlp_server_url
            )
            import requests

            try:
                requests.head(core_nlp_server_url).ok
                self._core_nlp_server_url = core_nlp_server_url
            except IOError:
                logger.exception("During HEAD request:")
                logger.warn(
                    "Failed to load CoreNLP server %s, make sure it is live and ready,"
                    " continuing without it.",
                    core_nlp_server_url,
                )

        if srl_predictor_path:
            logger.debug(
                "Initiating allennlp SRL server with model from %s", srl_predictor_path
            )
            from multiprocessing import Process, Queue

            self._srl_qi = Queue()
            self._srl_qo = Queue()
            self._srl_count = 0

            p = Process(
                target=_predictor_server,
                args=(srl_predictor_path, self._srl_qi, self._srl_qo),
                daemon=True,
            )
            p.start()

            @atexit.register
            def cleanup():
                logger.debug("Calling cleanup")
                self._srl_qi.put(None, timeout=0.5)
                self._srl_qi.close()
                self._srl_qo.close()
                p.join(timeout=5)
                if p.is_alive():
                    logger.error("Killing SRL server")
                    p.kill()

        if snips_path:
            logger.debug("Loading snips engine from %s", snips_path)
            from snips_nlu import SnipsNLUEngine

            if os.path.isdir(snips_path):
                logger.debug("%s is a directory, loading directly", snips_path)
                self._engine = SnipsNLUEngine.from_path(snips_path)
            else:
                with tarfile.open(snips_path, "r:gz") as archive:
                    with TemporaryDirectory() as tmp:
                        archive.extractall(tmp)
                        logger.debug(
                            "Extracted to temporary dir %s, loading from there", tmp
                        )
                        self._engine = SnipsNLUEngine.from_path(
                            os.path.join(tmp, "engine")
                        )

        if transformations:
            logger.debug("Loading transformations file from %s", transformations)

            with open(transformations) as f:
                self._transformations = json.load(f)

        if spacy_lang:
            logger.debug("Loading spacy lang %s", spacy_lang)
            try:
                module = importlib.import_module(spacy_lang)
            except ModuleNotFoundError:
                from spacy.cli.download import download

                download(spacy_lang)
                module = importlib.import_module(spacy_lang)

            self._spacy = module.load()

        if neuralcoref:
            if self._spacy is None:
                raise ValueError("neuralcoref is set but no spacy model is loaded")

            import neuralcoref

            neuralcoref.add_to_pipe(self._spacy)

        if self._spacy and self._core_nlp_server_url:
            from spacy.tokens import Token

            Token.set_extension("quote", default=None, force=True)

    @lru_cache(maxsize=1024)
    def _parse(self, s: str) -> JsonDict:
        logger.debug("Passing '%s' to snips engine", s)
        return self._transform(self._engine.parse(s))

    def parse(self, s: str) -> SnipsResult:
        assert self._engine

        result = SnipsResult.from_parsed(self._parse(s))
        logger.debug("Result = '%s'", result)
        return result

    def parse_tokens(self, tokens: Sequence[Token]) -> SnipsResult:
        s = " ".join(str(t) for t in tokens)  # TODO: better whitespace
        if not self._core_nlp_server_url or all(t._.quote is None for t in tokens):
            return self.parse(s)

        idx_to_token = {}
        idx = 0
        for token in tokens:
            start = idx
            idx += len(token)
            idx_to_token[(start, idx)] = token
            idx += 1

        # Copy since we modify what is returned by the lru_cache.
        result = copy.deepcopy(self._parse(s))
        # We don't bother to modify "input" so it will be invalid afterwards, make sure
        # we don't use it later.
        del result["input"]

        offset = 0
        for slot in result["slots"]:
            if slot["range"]["start"] < 0:
                continue

            raw = slot.get("rawValue", "")
            idx = slot["range"]["start"]

            slot["range"]["start"] += offset

            raw_tokens = []
            for raw_token in raw.split(" "):
                start = idx
                idx += len(raw_token)
                replacement = idx_to_token[(start, idx)]._.quote
                if replacement:
                    assert raw_token == QUOTE_STRING
                    raw_tokens.append(replacement)
                    offset += len(replacement) - len(raw_token)
                else:
                    raw_tokens.append(raw_token)
                idx += 1
            slot["rawValue"] = slot["value"]["value"] = " ".join(raw_tokens)

            slot["range"]["end"] += offset

        snips_result = SnipsResult.from_parsed(result)
        logger.debug("Result = '%s'", snips_result)
        return snips_result

    def _transform(self, parsed: JsonDict):
        transformation = self._transformations.get(parsed["intent"]["intentName"])
        if not transformation:
            return parsed

        parsed["intent"]["intentName"] = transformation["name"]
        for slot, value in transformation.get("slots", {}).items():
            name, entity = slot.split(":")

            if any(existing["slotName"] == name for existing in parsed["slots"]):
                logger.debug("Slot %s already exists", name)
                continue

            parsed["slots"].insert(
                0,  # Insert in beginning because of the invalid range.
                {
                    "slotName": name,
                    "entity": entity,
                    "range": {"start": -1, "end": -1},
                    "value": {"kind": "Custom", "value": value},
                },
            )

        return parsed

    def srl_put(self, s: str) -> None:
        logger.debug("SRL put: %s", s)
        self._srl_count += 1
        self._srl_qi.put_nowait(s)

    def srl_get(self) -> JsonDict:
        self._srl_count -= 1
        return self._srl_qo.get()

    def srl_clear(self) -> None:
        while self._srl_count > 0:
            self.srl_get()

    def spacy(self, s: str) -> Doc:
        assert self._spacy

        logger.debug("Passing '%s' to spacy", s)
        return self._spacy(s)

    def core_annotate(
        self, s: str, properties: Optional[Dict[str, str]] = None, **kwargs
    ) -> JsonDict:
        if not self._core_nlp_server_url:
            raise ValueError("CoreNLP server not configured")
        import requests

        properties = properties or {}
        properties.update(kwargs, outputFormat="json")
        if "annotators" in properties:
            annotators = properties["annotators"]
            if not isinstance(annotators, str):
                if not isinstance(annotators, Iterable):
                    raise TypeError("annotators should be Iterable or str")
                properties["annotators"] = ",".join(
                    str(annotator) for annotator in annotators
                )

        # TODO: requests session instead of Connection: close
        logger.debug("Sending data=%s, properties=%s to CoreNLP", s, properties)
        r = requests.post(
            self._core_nlp_server_url,
            data=s.encode(),
            params={"properties": str(properties)},
            headers={"Connection": "close"},
        )
        if r.ok:
            logger.debug("Got reply")
            return json.loads(r.text)
        raise RuntimeError(
            f"Failed to POST to nlp server {self._core_nlp_server_url}: "
            f"{r.status_code}{' - ' + r.text if r.text else ''}"
        )


def _predictor_server(path, qi, qo):
    # Using os.fork() (called from multiprocessing) and allowing cleanups to be run like
    # normal is dangerous because some filesystem-related cleanups might be called
    # twice. That's why we remove them first, without executing them in this process.
    atexit._clear()

    from allennlp.predictors.predictor import Predictor

    predictor = Predictor.from_path(path)
    while True:
        s = qi.get()
        if s is None:
            break
        qo.put_nowait(predictor.predict(s))

    # We need to manually call atexit callbacks here because the multiprocessing module
    # doesn't call them:
    # https://stackoverflow.com/a/34507557/
    # https://github.com/python/cpython/blob/49fd6dd887df6ea18dbb1a3c0f599239ccd1cb42/Lib/multiprocessing/popen_fork.py#L75
    # But if we don't call them, allennlp leaves extracted archives in the $TMPDIR:
    # https://github.com/allenai/allennlp/blob/fefc439035df87e3d2484eb2f53ca921c4c2e2fe/allennlp/models/archival.py#L176-L178
    logger.debug("atexit should call %d callbacks", atexit._ncallbacks())
    atexit._run_exitfuncs()


@total_ordering
@dataclass(order=False, frozen=True)
class SnipsResult:
    """Dataclass that holds the output of the SNIPS engine."""

    score: float = 0.0
    name: Optional[str] = None
    slots: Tuple[SnipsSlot] = ()
    input: Optional[str] = None

    @classmethod
    def from_parsed(cls, parsed: Optional[JsonDict] = None):
        if (
            parsed is None
            or parsed["intent"] is None
            or parsed["intent"]["intentName"] is None
        ):
            return cls()

        return cls(
            score=parsed["intent"]["probability"],
            name=parsed["intent"]["intentName"],
            slots=tuple(SnipsSlot.from_parsed(slot) for slot in parsed["slots"]),
            input=parsed.get("input", None),
        )

    def to_eobject(self) -> EObject:
        """Convert to an `EObject` of the corresponding `EClass`."""
        from r4a_nao_nlp import ecore

        return ecore.snips_result_to_eobject(self)

    def __bool__(self):
        return self.name is not None

    def __float__(self):
        return float(self.score)

    def __iter__(self):
        return iter(self.slots)

    @utils.other_isinstance
    def __lt__(self, other: object):
        return self.score < other.score

    @utils.other_isinstance
    def __eq__(self, other: object):
        return isclose(self.score, other.score, abs_tol=0.001)

    def __str__(self):
        return "{intent}({args})".format(
            intent=self.name, args=",".join(str(slot) for slot in self.slots)
        )


@dataclass(order=False, frozen=True)
class SnipsSlot:
    """Dataclass that holds the snips output of a single slot."""

    range: range
    value: SlotValue
    entity: str
    name: str

    @classmethod
    def from_parsed(cls, parsed: JsonDict):
        return cls(
            range=range(parsed["range"]["start"], parsed["range"]["end"]),
            value=_resolve_value(parsed["value"]),
            entity=parsed["entity"],
            name=parsed["slotName"],
        )

    @property
    def start(self):
        return self.range.start

    @property
    def end(self):
        return self.range.stop

    def __str__(self):
        return f"{self.name}={self.value}"


def _resolve_value(value: JsonDict) -> SlotValue:
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


shared = Shared()
# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
