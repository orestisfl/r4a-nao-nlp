# TODO: docstrings
import logging
import os
import tarfile
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import spacy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # TODO
# XXX: Can use pkg_resources to find distributed resources:
# https://setuptools.readthedocs.io/en/latest/pkg_resources.html
HERE = os.path.abspath(os.path.dirname(__file__))

JsonDict = Dict[str, Any]


class Shared:
    def __init__(self):
        logger.debug("Creating shared object")

        # TODO: typing, make some of them less optional
        self.engine = None
        self.srl_predictor = None
        self.coref_predictor = None
        self.spacy = None

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

        # TODO: disable parser? https://spacy.io/usage/linguistic-features
        if spacy_lang and neural_coref_model:
            logger.debug(
                "Skipping spacy model %s, will load neuralcoref model %s",
                spacy_lang,
                neural_coref_model,
            )
        elif spacy_lang:
            logger.debug("Loading spacy lang %s", spacy_lang)
            self.spacy = spacy.load(spacy_lang)

        if neural_coref_model:
            logger.debug("Loading spacy neuralcoref model %s", neural_coref_model)
            self.spacy = self.neuralcoref = spacy.load(neural_coref_model)

    def parse(self, s: str, use_cache: bool = True) -> dict:
        assert self.engine

        if use_cache:
            return self._parse(s)
        else:
            return self.engine.parse(s)

    @lru_cache(maxsize=1024)
    def _parse(self, s: str) -> JsonDict:
        logger.debug("Passing '%s' to snips engine", s)
        return self.engine.parse(s)

    def srl(self, s: str) -> JsonDict:
        assert self.srl_predictor

        # TODO: just lru cache
        r = self.srl_predictor.predict(s)
        self.srl_cache[s] = r
        return r

    def coref(self, s: str) -> Union[spacy.tokens.doc.Doc, JsonDict]:
        if self.neuralcoref:
            return self.neuralcoref(s)
        else:
            assert self.coref_predictor

            return self.coref_predictor.predict(s)


shared = Shared()


def parsed_score(parsed: JsonDict) -> float:
    return 0.0 if parsed["intent"] is None else parsed["intent"]["probability"]
