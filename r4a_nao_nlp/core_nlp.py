"""Contains functions used to communicate with the Stanford CoreNLP server.

The output of the following annotators is used:
- QuoteAnnotator - https://stanfordnlp.github.io/CoreNLP/quote.html
- CorefAnnotator - https://stanfordnlp.github.io/CoreNLP/coref.html
  statistical and neural models.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

from r4a_nao_nlp import utils
from r4a_nao_nlp.engines import shared

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import JsonDict

    TokenIdx = Tuple[int, int]  # (sentence index, token index)
    CharacterIdx = Tuple[int, int]  # (character offset start, character offset end)
    TokenToCharacter = Dict[TokenIdx, CharacterIdx]
    IdxTextTuple = Tuple[CharacterIdx, str]
    CorefDict = Dict[IdxTextTuple, List[IdxTextTuple]]  # TODO: Dict[str, …]

logger = utils.create_logger(__name__)

BASE_ANNOTATORS = ("tokenize", "ssplit", "pos", "lemma", "ner")
ANNOTATORS_STATISTICAL_COREF = ("depparse", "coref")
ANNOTATORS_NEURAL_COREF = ("parse", "coref")


def get_quotes(s: str) -> List[JsonDict]:
    # This is a quite fast pipeline because we don't use quote attribution.
    return shared.core_annotate(
        s,
        # NOTE: Need to use customAnnotatorClass because of CoreNLP bug
        properties={
            "annotators": "tokenize, ssplit, quote1",
            "customAnnotatorClass.quote1": "edu.stanford.nlp.pipeline.QuoteAnnotator",
            "quote1.attributeQuotes": "false",
        },
    )["quotes"]


# TODO: Doc
def annotate_coref(s: str) -> CorefDict:
    statistical_corefs = _statistical_coref(s)
    # XXX: use a thread here?
    neural_corefs = _neural_coref(s)
    # TODO: Doc
    # return _corefs(reply)
    # XXX: maybe this is too slow?
    return _merge_corefs(statistical_corefs, neural_corefs)


def _statistical_coref(s: str) -> CorefDict:
    return _corefs(
        shared.core_annotate(
            s, annotators=BASE_ANNOTATORS + ANNOTATORS_STATISTICAL_COREF
        )
    )


def _neural_coref(s: str) -> CorefDict:
    return (
        _corefs(
            shared.core_annotate(
                s,
                properties={"coref.algorithm": "neural"},
                annotators=BASE_ANNOTATORS + ANNOTATORS_NEURAL_COREF,
            )
        ),
    )


def _corefs(parsed: JsonDict) -> CorefDict:
    """Convert given dict containing "_corefs" from the CoreNLP annotator to our own
    format."""
    result = {}
    token_to_char_map = _token_to_char(parsed)
    for cluster in parsed["corefs"].values():
        if len(cluster) < 2:
            logger.error("Skipping cluster with length %s < 2", len(cluster))
            continue
        for mention in cluster:
            if mention["isRepresentativeMention"]:
                representative = _mention_to_idx_text_tuple(mention, token_to_char_map)
                break
        else:
            logger.error("Couldn't find representative mention")
            continue
        assert representative not in result
        result[representative] = [
            _mention_to_idx_text_tuple(mention, token_to_char_map)
            for mention in cluster
            if not mention["isRepresentativeMention"]
        ]
    return result


def _merge_corefs(dest: CorefDict, src: CorefDict) -> CorefDict:
    """Update destination coreference dict to include all mentions from the second
    dict.

    Values in existing representative mentions are updated with new unique ones and then
    sorted.
    """
    for k, v in src.items():
        target = dest.get(k)
        if target:
            target.extend(mention for mention in v if mention not in target)
            target.sort()
        else:
            dest[k] = v
    return dest


def _mention_to_idx_text_tuple(
    mention: JsonDict, token_to_char_map: TokenToCharacter
) -> IdxTextTuple:
    sent_idx = mention["sentNum"] - 1
    start_idx = token_to_char_map[(sent_idx, mention["startIndex"])][0]
    end_idx = token_to_char_map[(sent_idx, mention["endIndex"] - 1)][1]

    return ((start_idx, end_idx), mention["text"])


def _token_to_char(parsed: JsonDict) -> TokenToCharacter:
    result = {}
    for sent in parsed["sentences"]:
        tokens = sent["tokens"]
        sent_idx = sent["index"]
        for token in tokens:
            result[(sent_idx, token["index"])] = (
                token["characterOffsetBegin"],
                # XXX: characterOffsetEnd unused
                token["characterOffsetEnd"],
            )
    return result


# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
