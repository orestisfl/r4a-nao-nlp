"""Contains functions used to communicate with the Stanford CoreNLP server.

The output of the following annotators is used:
- QuoteAnnotator - https://stanfordnlp.github.io/CoreNLP/quote.html
- CorefAnnotator - https://stanfordnlp.github.io/CoreNLP/coref.html
  statistical and neural models.
"""
from __future__ import annotations

from collections import deque
from functools import reduce
from itertools import chain
from threading import Thread
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterable, List, Tuple

from r4a_nao_nlp import utils
from r4a_nao_nlp.engines import QUOTE_STRING, shared

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import JsonDict, Doc, Span, Token

    TokenIdx = Tuple[int, int]  # (sentence index, token index)
    CharacterIdx = Tuple[int, int]  # (character offset start, character offset end)
    TokenToCharacter = Dict[TokenIdx, CharacterIdx]
    IdxTextTuple = Tuple[CharacterIdx, str]
    CorefDict = Dict[IdxTextTuple, List[IdxTextTuple]]  # TODO: Dict[str, …]

logger = utils.create_logger(__name__)

BASE_ANNOTATORS = ("tokenize", "ssplit", "pos", "lemma", "ner")
ANNOTATORS_STATISTICAL_COREF = ("depparse", "coref")
ANNOTATORS_NEURAL_COREF = ("parse", "coref")


def replace_quotes(s: str) -> Tuple[str, Deque[str]]:
    assert QUOTE_STRING not in s
    quotes = _get_quotes(s)
    q = deque()

    if quotes:
        chars = list(s)
        for quote in quotes:
            start = quote["beginIndex"]
            end = quote["endIndex"]
            chars[start] = QUOTE_STRING
            for idx in range(start + 1, end):
                chars[idx] = ""
            q.append(quote["text"])
        s = "".join(chars)
    return s, q


def _get_quotes(s: str) -> List[JsonDict]:
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


class CorefThreads:
    def __init__(self, s: str, methods: Iterable[str] = ("statistical", "neural")):
        self._threads = tuple(
            Thread(
                target=getattr(self, f"_{name}_coref"),
                args=(s,),
                daemon=True,  # Does not prohibit the process from exiting.
                name=name,
            )
            for name in methods
        )
        for thread in self._threads:
            thread.start()
        self._joined = None

    def _statistical_coref(self, s: str) -> None:
        self.statistical = _corefs(
            shared.core_annotate(
                s, annotators=BASE_ANNOTATORS + ANNOTATORS_STATISTICAL_COREF
            )
        )

    def _neural_coref(self, s: str) -> None:
        # XXX: maybe this is too slow?
        self.neural = _corefs(
            shared.core_annotate(
                s,
                properties={"coref.algorithm": "neural"},
                annotators=BASE_ANNOTATORS + ANNOTATORS_NEURAL_COREF,
            )
        )

    def join(self) -> CorefDict:
        if self._joined is None:
            for thread in self._threads:
                thread.join()
            self._joined = reduce(
                _merge_corefs, (getattr(self, thread.name) for thread in self._threads)
            )
        return self._joined


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
                token["characterOffsetEnd"],
            )
    return result


def doc_mark_quotes(doc: Doc, replacements: Deque[str]) -> None:
    """Assign the `quote` extension attribute to all appropriate "QUOTE" tokens
    according to the given `replacements`."""
    if replacements:
        for token in doc:
            if str(token) == QUOTE_STRING:
                token._.quote = replacements.popleft()
        assert not replacements


def doc_enhance_corefs(doc: Doc, corefs: CorefDict) -> None:
    """Extend the coreference clusters annotated by neuralcoref with the given
    coreference dict.

    If the `coref_clusters` extension attribute doesn't exist, this function will create it.

    If the representative mention of a cluster is already found by neuralcoref, all new
    secondary mentions will be appended to its `Cluster.mentions` list. Otherwise, a new
    `MockCluster` object is created and the mentions are saved there.

    In either case, the `coref_clusters` list is updated for each token in the `doc`.

    After using this function, some methods provided by neuralcoref are not expected to
    work. The following should still work though:
    - `cluster.main`
    - iterating over a cluster
    - `doc._.coref_clusters`
    - `token._.coref_clusters`
    """
    from spacy.tokens import Doc, Token

    if not Doc.has_extension("coref_clusters"):
        Doc.set_extension("coref_clusters", default=[])
    if not Token.has_extension("coref_clusters"):
        Token.set_extension("coref_clusters", default=[])

    for ((main_start, main_end), main_text), references in corefs.items():
        mentions = [doc.char_span(start, end) for (start, end), _ in references]
        for cluster in doc._.coref_clusters:
            if (
                cluster.main.start_char == main_start
                and cluster.main.end_char == main_end
            ):
                logger.debug(
                    "Adding mentions %s to existing cluster %s", mentions, cluster
                )
                assert main_text == str(cluster.main)

                for mention in mentions:
                    if mention not in cluster:
                        for token in mention:
                            _token_add_cluster(token, cluster)
                        cluster.mentions.append(mention)
                # XXX: sort?
                break
        else:
            main_span = doc.char_span(main_start, main_end)
            cluster = MockCluster(main_span, mentions)
            for token in chain.from_iterable(cluster):
                _token_add_cluster(token, cluster)

            assert all(
                text == str(mention) for (_, text), mention in zip(references, cluster)
            )


class MockCluster:
    """Immitates a `neuralcoref.Cluster` object."""

    def __init__(self, main: Span, mentions: List[Span]):
        self.main = main
        self.mentions = mentions

    def __iter__(self):
        return iter(self.mentions)

    @property
    def i(self):
        raise NotImplementedError


def _token_add_cluster(token: Token, cluster: Any):
    if cluster not in token._.coref_clusters:
        token._.coref_clusters.append(cluster)


# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
